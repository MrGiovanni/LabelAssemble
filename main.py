import argparse

import datasets
import torch
from augmentation import Augmentation
import densenet121
from tqdm import tqdm
import numpy as np
from sklearn.metrics._ranking import roc_auc_score
import torchvision.transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from noise import *




class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset.get_labels(idx)
           

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



def get_arguments():

    parser = argparse.ArgumentParser(description="xxx")
    parser.add_argument("--dataset", type=str, default='assemble')
    parser.add_argument("--data_dir", type=str, default='./data/images')
    parser.add_argument("--data_dir2", type=str, default='xxx')
    parser.add_argument("--list_train", type=str, default='./ChestXray14_11March2021/clean_code/dataset/train_official.txt')
    parser.add_argument("--list_train2", type=str,
                        default='./ChestXray14_11March2021/clean_code/dataset/train_official.txt')

    parser.add_argument("--list_test", type=str, default='./ChestXray14_11March2021/clean_code/dataset/test_official.txt')
    parser.add_argument("--list_test2", type=str,
                        default='./ChestXray14_11March2021/clean_code/dataset/test_official.txt')

    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--test_interval", type=int, default=3)
    parser.add_argument("--LR", type=float, default=2e-4)

    return parser


def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    dataGT = np.array(dataGT)
    dataPRED = np.array(dataPRED)
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    return outAUROC


def test(model, args):
    augment = Augmentation(normalize="imagenet").get_augmentation(
        "{}_{}".format('full', 224), "valid")

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                normalize])

    if args.dataset=='nih':
        testset = datasets.NIH(args.data_dir, args.list_test, transform=transforms, num_class=args.num_class)
    elif args.dataset=='chexpert':
        testset = datasets.Chexpert(args.data_dir, args.list_test, transform=transforms, num_class=args.num_class)
    elif args.dataset=='CXR3':
        testset = datasets.CXR3('dataset', 'dataset', transforms, num_class=args.num_class, mode='test')
    elif args.dataset == 'assemble':
        testset = datasets.CXR3('dataset', 'dataset', transforms, num_class=args.num_class, mode='test')


    dataloader=torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)

    predict = []
    target = []
    model.eval()
    print(f'test now')


    for inputs, labels, _ in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs,fea=model(inputs)
            predict.append(outputs)
            target.append(labels)
    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()
    auc = computeAUROC(target, predict, args.num_class)
    print(f'\n {auc}  avg_auc: {np.average(auc)} \n')
    predict, target = predict[:, 0] > 0.5, target[:, 0]
    print(confusion_matrix(target, predict))
    print('acc:', accuracy_score(target, predict), 'precision:', precision_score(target, predict), 'sensitivity:', recall_score(target, predict))

    return np.average(auc), auc

def main():
    parser = get_arguments()
    args = parser.parse_args()

    if args.mode=='train':
        augment = Augmentation(normalize="imagenet").get_augmentation(
            "{}_{}".format('full', 224), "train")

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.Resize(256),
            AddPepperNoise(snr=0.9, p=0.1),
            AddGaussianNoise(p=0.3),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            normalize
        ])

        transforms_consistency = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            AddPepperNoise(snr=0.7, p=0.5),
            AddGaussianNoise(p=0.5),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.ToTensor(),
            normalize,
            torchvision.transforms.RandomErasing()
        ])
        if args.dataset=='nih':
            trainset=datasets.NIH(args.data_dir, args.list_train, transform=transforms, num_class=args.num_class, reduct_ratio=1)
        elif args.dataset=='chexpert':
            trainset = datasets.Chexpert(args.data_dir, args.list_train, transform=transforms, num_class=args.num_class)
        elif args.dataset=='c_c':
            trainset = datasets.c_c(args.data_dir,args.data_dir2,args.list_train,args.list_train2,transform1=transforms
                                    ,transform2=transforms, reduct_ratio=10, transform_consistency=transforms_consistency)
        elif args.dataset == 'assemble':
            trainset = datasets.Assemble(['dataset', 'images/images'], ['dataset', 'images/train.txt'], augment=[transforms, transforms_consistency], num_class=args.num_class)
        # dataloader=torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)

        dataloader=torch.utils.data.DataLoader(trainset, sampler=ImbalancedDatasetSampler(trainset), batch_size=32, num_workers=12, pin_memory=True)

        model = densenet121.densenet121(pretrained=True,num_classes=args.num_class)
        #model.classifier_me = torch.nn.Sequential(torch.nn.Linear(model.classifier_me.in_features, args.num_class), torch.nn.Sigmoid())
        model.train()
        model = model.cuda()

        criterion=torch.nn.BCELoss().cuda()

        best_auc=-1
        best_epoch=-1
        best_auc_full=None

        best_auc2=-1
        best_auc_full_2=None
        best_epoch2=-1
        lr_count=0

        optimizer = torch.optim.Adam(lr=args.LR, params=filter(lambda p: p.requires_grad, model.parameters()))
        #optimizer.add_param_group({'params':model.encodings,'lr':args.LR})

        # model=torch.load('best')
        # test(model,args)
        # # # model = torch.load('best2_6')
        # # # test(model, args)
        # # # model = torch.load('best2_7')
        # # # test(model, args)
        # return 0
        # model = torch.load('best_aug_nopseudo')
        # test(model, args)
        # return 0

        for i in range(args.epochs):
            print(f'epoch: {i}/{args.epochs}')
            for img, label, source, img_consistency in tqdm(dataloader):
 
                # break
                img, label, source, img_consistency=img.cuda(), label.cuda(), source.cuda(), img_consistency.cuda()
                output, dis_res=model(img)
                output_consistency,_ = model(img_consistency)

                loss = torch.tensor(0.).cuda()
                loss_consistency = torch.tensor(0.).cuda()



                if args.dataset=='c_c':
                    loss = torch.tensor(0.).cuda()

                    for i1 in range(len(img)):
                        for i2 in range(args.num_class):
                            if i2 in [2, 5]: #[8,9,2,6,7]:
                                loss += criterion(output[i1][i2], label[i1][i2])
                    loss = loss / len(img)
                elif args.dataset == 'assemble':
                    loss = torch.tensor(0.).cuda()
                    loss_consistency=torch.tensor(0.).cuda()
                    loss_pseudo = torch.tensor(0.).cuda()

                    # for i1 in range(len(img)):
                    #     for i2 in range(args.num_class):
                    #         if output[i1][i2]>0.7 and ((source[i1]==0 and i2 in [1,2]) or (source[i1]==1 and i2 in [18, 19])):

                    


                    for i1 in range(len(img)):
                        for i2 in range(0, args.num_class):
                            if source[i1] == 0:
                                #print(label[i1][i2])
                                loss += criterion(output[i1][i2], label[i1][i2])
                            elif source[i1] == 1:
                                #print(label[i1][i2])
                                loss += criterion(output[i1][i2], label[i1][i2])
#                             if output[i1][i2] > 0.5:
#                                 tmp = output[i1][i2]+(1-output[i1][i2])/4.0
#                             else:
#                                 tmp = output[i1][i2] - output[i1][i2] / 4.0
#                             loss_pseudo += F.mse_loss(output[i1][i2],tmp).cuda()
#                             loss_consistency += F.mse_loss(output_consistency[i1][i2],tmp).cuda()
                            
                                
                            # elif (source[i1]==0 and i2 in [2,18]) or (source[i1]==1 and i2 in [3,19]):
                            #     # if output[i1][i2]>0.5:
                            #     #     loss+=F.mse_loss(output[i1][i2],output[i1][i2]+(1-output[i1][i2])/4.0).cuda()
                            #     # else:
                            #     #     loss += F.mse_loss(output[i1][i2],
                            #     #                        output[i1][i2] - output[i1][i2] / 4.0).cuda()
                            #     if output[i1][i2]>0.5:
                            #         tmp=output[i1][i2]+(1-output[i1][i2])/1.0
                            #     else:
                            #         tmp=output[i1][i2]-output[i1][i2]/1.0
                            #     loss_consistency += torch.nn.MSELoss()(output_consistency[i1][i2], tmp)
                                #
                                # if output[i1][i2]>0.5:
                                #     #tmp = output[i1][i2] + (1 - output[i1][i2]) / 4.0
                                #     tmp=output[i1][i2]
                                #     loss += F.mse_loss(output_consistency[i1][i2], tmp).cuda()
                                # else:
                                #     #tmp = output[i1][i2]-output[i1][i2]/4.0
                                #     tmp=output[i1][i2]
                                #     loss += F.mse_loss(output_consistency[i1][i2], tmp).cuda()
                                # elif output[i1][i2]<0.2:
                                #     loss += criterion(output[i1][i2], torch.tensor(0.0).cuda())

                            # if i2 in [1,2,3,4,5,18,19]:
                            #     loss += criterion(output[i1][i2], label[i1][i2])


                    loss=loss/len(img)
                    # loss_pseudo=loss_pseudo/len(img)
                    # loss_consistency=loss_consistency/len(img)
                    #print(f'loss: {loss}')
                    # print(f'loss_consistency: {loss_consistency}')


                #loss_discriminator=F.cross_entropy(dis_res, source)
                #print(f'loss: {loss}  loss_discriminator: {loss_discriminator}')
                # loss+=loss_consistency
                # loss+=loss_pseudo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(model,'noiseCovid/f/epoch_%d' %i)


            if i%args.test_interval==0:

                if args.dataset=='assemble':
                    auc,auc_full=test(model, args)
                    if auc>best_auc:
                        best_auc=auc
                        best_auc_full=auc_full
                        best_epoch = i
                        torch.save(model,'noiseCovid/f/best')
                    else:
                        lr_count+=1
                    print(f'best auc: {best_auc}   {best_auc_full}, best epoch: {best_epoch}')
              


            if i!=0 and i%3==0:
                print('adjust learning rate now')
                for param_group in optimizer.param_groups:
                    param_group["lr"]/=2.0

if __name__=='__main__':
    main()
