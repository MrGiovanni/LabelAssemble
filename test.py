import datasets
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from noise import *
from config import device
from metrics import *
from logger import Logger
from config import *

def test(model, args):

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                normalize])
    logger = Logger()
    
    test_set = dcovidx = datasets.COVIDX(COVIDXConfig, mode='test', source=0, augment=transforms, start_id=0, num_classes=2)

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batchSize, shuffle=True, num_workers=12, pin_memory=True)

    predict = []
    target = []
    model.eval()
    logger.info('Testing starts!')

    for inputs, labels, _ in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs,fea=model(inputs)
            predict.append(outputs)
            target.append(labels)
    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()
    logger.info(predict)
    auc = compute_roc_auc(target, predict, args.numClass)
    logger.info(f'\n {auc}  avg_auc: {np.average(auc)} \n')
    results = compute_conf(target, predict)
    return np.average(auc), auc





