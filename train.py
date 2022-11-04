import datasets
import torch
import densenet121
from tqdm import tqdm
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from noise import *
from aug import *
import os
from utils import *
from logger import Logger
# from loss import Loss





def train(args):
    # avoid rewrite pth file
    
    assert not os.path.exists(args.saveDir), 'This directory has already been created!'
    os.makedirs(args.saveDir, exist_ok=False)
    logger = Logger()

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    weak_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.Resize(256),
        AddPepperNoise(snr=0.9, p=0.1),
        AddGaussianNoise(p=0.3),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    strong_aug = torchvision.transforms.Compose([
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
    
    if args.datasetType == 'assemble':
        logger.info('Label training')


    #     trainset = datasets.Assemble([args.covidxTrainImagePath, args.chestImagePath], 
    # [args.covidxTrainFilePath, args.chestFilePath], augments=[weak_aug, strong_aug],
    # covidx_ratio=args.covidxRatio, chest_ratio=args.chestRatio, 
    # label_assembles=['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule','Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'], num_class=args.numClass)
    
    
    
    elif args.datasetType == 'covidx':
        # TODO: implement this part
        pass
    else:
        raise ValueError("No such dataset type: %s" % args.datasetType)
    
        # dataloader=torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)

    dataloader=torch.utils.data.DataLoader(trainset, batch_size=8, num_workers=12, pin_memory=True)

    model = densenet121.densenet121(pretrained=True,num_classes=args.numClass)
    model.train()
    model = model.cuda()


    best_auc=-1
    best_epoch=-1
    best_auc_full=None

    lr_count=0

    optimizer = torch.optim.Adam(lr=args.lr, params=filter(lambda p: p.requires_grad, model.parameters()))


    for i in range(args.epochs):
        print(f'epoch: {i}/{args.epochs}')
        cnt = 0
        for img, label, source, img_consistency in tqdm(dataloader):

            img, label, source, img_consistency=img.cuda(), label.cuda(), source.cuda(), img_consistency.cuda()
            output, dis_res=model(img)
            output_consistency,_ = model(img_consistency)
            criterion = Loss()
            if args.datasetType == 'assemble':        
                loss = criterion.cal_loss(output, output_consistency, label, source)
                loss=loss/len(img)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model,os.path.join(args.saveDir, 'epoch_%d' %i))
        if i%args.testInterval==0:
            if args.datasetType=='assemble':
                auc,auc_full=test(model, args)
                if auc>best_auc:
                    best_auc=auc
                    best_auc_full=auc_full
                    best_epoch = i
                    torch.save(model, os.path.join(args.saveDir, 'best'))
                else:
                    lr_count+=1
                print(f'best auc: {best_auc}   {best_auc_full}, best epoch: {best_epoch}')
          
        if i!=0 and i%3==0:
            print('adjust learning rate now')
            for param_group in optimizer.param_groups:
                param_group["lr"]/=2.0

if __name__=='__main__':
    main()
