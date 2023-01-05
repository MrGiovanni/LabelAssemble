import datasets
import torch
import densenet121
from tqdm import tqdm
import torchvision.transforms
import torch.nn.functional as F
from noise import *
import os
from utils import *
from loss import FullyLoss, SemiLoss
from logger import Logger
from config import *
from test import test

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
        logger.info('Label Assemble')
        covidx = datasets.COVIDX(COVIDXConfig, mode='train', source=0, augment=None, start_id=0, num_classes=2)
        # covidx = datasets.COVIDX(img_path=args.covidxTrainImagePath,
        #             file_path=args.covidxTrainFilePath,
        #             augment=None,
        #             extra_num_class=args.extraNumClass,
        #             select_num=args.covidxNum)
        chestxray14 = datasets.ChestXRay14(ChestXray14Config, mode='train', source=0, augment=None, start_id=1, num_classes=2)
        train_set = datasets.Assemble([covidx, chestxray14], augments=[weak_aug, strong_aug])


    
    elif args.datasetType == 'covidx':
        logger.info('COVIDx')
        train_set = datasets.COVIDX(img_path=args.covidxTrainImagePath,
                    file_path=args.covidxTrainFilePath,
                    augment=None,
                    extra_num_class=args.extraNumClass,
                    select_num=args.covidxNum)

    else:
        raise ValueError("No such dataset type: %s" % args.datasetType)
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=True, pin_memory=True)

    model = densenet121.densenet121(pretrained=True, num_classes=args.numClass)
    model.train()
    model = model.to(args.device)
    optimizer = torch.optim.Adam(lr=args.lr, params=filter(lambda p: p.requires_grad, model.parameters()))

    best_auc = -1
    best_epoch = -1
    best_auc_full = None

    lr_count=0

    for i in range(args.epochs):
        logger.info(f'epoch: {i}/{args.epochs}')
        cnt = 0
        for img, label, source, img_consistency in tqdm(dataloader):
            img, label, source, img_consistency=img.to(args.device), label.to(args.device), source.to(args.device), img_consistency.to(args.device)
            output, _ = model(img)
            output_consistency, _ = model(img_consistency)
            criterion = FullyLoss()
            if args.datasetType == 'assemble':        
                loss = criterion(output, label, source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model,os.path.join(args.saveDir, 'epoch_%d' %i))
        if i % args.testInterval==0:
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

