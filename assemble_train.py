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

def get_arguments():

    parser = argparse.ArgumentParser(description="xxx")
    parser.add_argument("--dataset", type=str, default='nih')
    parser.add_argument("--data_dir", type=str, default='./data/images')
    parser.add_argument("--data_dir2", type=str, default='xxx')
    parser.add_argument("--list_train", type=str, default='./ChestXray14_11March2021/clean_code/dataset/train_official.txt')
    parser.add_argument("--list_train2", type=str,
                        default='./ChestXray14_11March2021/clean_code/dataset/train_official.txt')

    parser.add_argument("--list_test", type=str, default='./ChestXray14_11March2021/clean_code/dataset/test_official.txt')
    parser.add_argument("--list_test2", type=str,
                        default='./ChestXray14_11March2021/clean_code/dataset/test_official.txt')

    parser.add_argument("--num_class", type=int, default=14)
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
                                                torchvision.transforms.Scale(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                normalize])

    if args.dataset=='nih':
        testset = datasets.NIH(args.data_dir, args.list_test, transform=transforms, num_class=args.num_class)
    elif args.dataset=='chexpert':
        testset = datasets.Chexpert(args.data_dir, args.list_test, transform=transforms, num_class=args.num_class)
    elif args.dataset == 'c_c':
        testset1, testset2 = datasets.NIH(args.data_dir, args.list_test, transforms, 14), \
                             datasets.Chexpert(args.data_dir2, args.list_test2, transforms, 14)
        dataloader1 = torch.utils.data.DataLoader(testset1, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)
        dataloader2 = torch.utils.data.DataLoader(testset2, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)

        predict = []
        target = []
        model.eval()
        print(f'test now for nih')
        for inputs, labels, _ in tqdm(dataloader1):
            inputs = inputs.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs,_ = model(inputs)
                outputs=outputs[:,[2,18,1,4,5,3,19]]
                predict.append(outputs)
                target.append(labels[:,[1,6,0,9,2,8,7]])



            #     for j in range(inputs.shape[0]):
            #         outputs,_ = model(inputs[j])
            #         outputs = outputs[:, [1, 2, 5, 11, 13, 14, 18, 19, 3, 4, 6, 8, 16, 10]]
            #         #print(outputs)
            #         predict.append(outputs.mean(dim=0,keepdim=True))
            #         target.append(labels[j:j+1])


        predict = torch.cat(predict, dim=0).cpu().numpy()
        target = torch.cat(target, dim=0).cpu().numpy()
        auc=computeAUROC(target, predict, 7)
        print(f'\n {auc}  avg_auc: {np.average(auc)} \n')

        predict = []
        target = []
        model.eval()
        print(f'test now for chexpert')
        for inputs, labels, _ in tqdm(dataloader2):
            inputs = inputs.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs,_ = model(inputs)
                outputs=outputs[:,[2,18,1,4,5,3,19]]

                predict.append(outputs)
                target.append(labels[:,[2,7,8,5,10,6,9]])


                # for j in range(inputs.shape[0]):
                #     outputs = model(inputs[j])
                #     predict.append(outputs.mean(dim=0,keepdim=True))
                #     target.append(labels[j:j+1])

        predict = torch.cat(predict, dim=0).cpu().numpy()
        target = torch.cat(target, dim=0).cpu().numpy()
        auc2 = computeAUROC(target, predict, 7)
        print(f'\n {auc2}  avg_auc2: {np.average(auc2)} \n')


        return np.average(auc), auc, np.average(auc2), auc2

    dataloader=torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)

    predict = []
    target = []
    model.eval()
    print(f'test now')
    for inputs, labels, _, _ in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs,_=model(inputs)
            predict.append(outputs)
            target.append(labels)
            # for j in range(inputs.shape[0]):
            #     outputs = model(inputs[j])
            #     predict.append(outputs.mean(dim=0,keepdim=True))
            #     target.append(labels[j:j+1])

    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()

    auc = computeAUROC(target, predict, args.num_class)
    print(f'\n {auc}  avg_auc: {np.average(auc)} \n')

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
            torchvision.transforms.Scale(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            normalize
        ])

        transforms_consistency = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.Scale(256),
            torchvision.transforms.CenterCrop(256),
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
            trainset = datasets.assemble(['dataset', 'images/images'], ['dataset', 'images/train.txt'], augment=[transforms, transforms_consistency], num_class=3)
        dataloader=torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)

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
                img, label, source, img_consistency=img.cuda(), label.cuda(), source.cuda(), img_consistency.cuda()
                output, dis_res=model(img)
                output_consistency,_ = model(img_consistency)

                loss = torch.tensor(0.).cuda()
                loss_consistency = torch.tensor(0.).cuda()



                if args.dataset!='c_c':
                    loss = torch.tensor(0.).cuda()

                    for i1 in range(len(img)):
                        for i2 in range(args.num_class):
                            if i2 in [2, 5]: #[8,9,2,6,7]:
                                loss += criterion(output[i1][i2], label[i1][i2])
                    loss = loss / len(img)
                else:
                    loss = torch.tensor(0.).cuda()
                    loss_consistency=torch.tensor(0.).cuda()
                    loss_pseudo = torch.tensor(0.).cuda()
                    # for i1 in range(len(img)):
                    #     for i2 in range(args.num_class):
                    #         if output[i1][i2]>0.7 and ((source[i1]==0 and i2 in [1,2]) or (source[i1]==1 and i2 in [18, 19])):

                    for i1 in range(len(img)):
                        for i2 in range(args.num_class):
                            if source[i1] == 0:
                                #print(label[i1][i2])
                                loss += criterion(output[i1][i2], label[i1][i2])
                            elif source[i1] == 1:
                                #print(label[i1][i2])
                                loss += criterion(output[i1][i2], label[i1][i2])
                            if output[i1][i2] > 0.5:
                                tmp = output[i1][i2]+(1-output[i1][i2])/4.0
                            else:
                                tmp = output[i1][i2]+(1-output[i1][i2])/4.0
                            loss_pseudo += F.mse_loss(output[i1][i2],tmp).cuda()
                            loss_consistency += F.mse_loss(output_consistency[i1][i2],tmp).cuda()
                            
                                
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
                    loss_pseudo=loss_pseudo/len(img)
                    loss_consistency=loss_consistency/len(img)
                    #print(f'loss: {loss}')
                    # print(f'loss_consistency: {loss_consistency}')


                #loss_discriminator=F.cross_entropy(dis_res, source)
                #print(f'loss: {loss}  loss_discriminator: {loss_discriminator}')
                loss_discriminator=torch.tensor(0.).cuda()
                loss+=loss_discriminator
                loss+=loss_consistency
                loss+=loss_pseudo

                optimizer.zero_grad()
                loss.backward()



            if i%args.test_interval==0:

                if args.dataset!='c_c':
                    auc,auc_full=test(model, args)
                    if auc>best_auc:
                        best_auc=auc
                        best_auc_full=auc_full
                        best_epoch = i
                        torch.save(model,'best_7')
                    else:
                        lr_count+=1
                    print(f'best auc: {best_auc}   {best_auc_full}, best epoch: {best_epoch}')
                else:
                    auc,auc_full,auc2,auc_full_2=test(model,args)
                    if auc>best_auc:
                        best_auc=auc
                        best_auc_full=auc_full
                        best_epoch = i
                        torch.save(model,'best_dis')
                    if auc2>best_auc2:
                        best_auc2=auc2
                        best_auc_full_2=auc_full_2
                        best_epoch2=i
                        torch.save(model,'best_dis_2')
                    print(f'best auc: {best_auc}   {best_auc_full}, best epoch: {best_epoch}')
                    print(f'best auc2: {best_auc2}   {best_auc_full_2}, best epoch: {best_epoch2}')



            if i!=0 and i%3==0:
                print('adjust learning rate now')
                for param_group in optimizer.param_groups:
                    param_group["lr"]/=2.0

if __name__=='__main__':
    main()