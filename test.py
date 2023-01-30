import datasets
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from noise import *
from metrics import *
from logger import Logger
from config import *
import copy
from build import build_dataset


def test(model, args):
    args_temp = copy.deepcopy(args)
    args_temp.mode = 'test'
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
    
    test_set = build_dataset(args_temp, weak_aug, strong_aug)
    logger = Logger()

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batchSize, shuffle=False, num_workers=12, pin_memory=True)

    predict = []
    target = []
    model.eval()
    logger.info('Testing starts!')

    for inputs, labels, source, img_consistency in tqdm(dataloader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs,fea=model(inputs)
            predict.append(outputs)
            target.append(labels)
    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()
    auc = compute_roc_auc(target, predict, args.numClass)
    logger.info(f'\n {auc}  avg_auc: {np.average(auc)} \n')
    results = compute_conf(target, predict)
    return np.average(auc), auc





