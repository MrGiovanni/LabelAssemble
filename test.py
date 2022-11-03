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
import os




def get_arguments():

    parser = argparse.ArgumentParser(description="Assemble Label")
    parser.add_argument("--datasetType", type=str, default='assemble')
    parser.add_argument("--covidxTrainImagePath", type=str)
    parser.add_argument("--covidxTestImagePath", type=str)
    parser.add_argument("--chestImagePath", type=str)
    parser.add_argument('--covidxTrainFilePath', type=str)
    parser.add_argument('--covidxTestFilePath', type=str)
    parser.add_argument("--chestFilePath", type=str)
    parser.add_argument("--numClass", type=int, default=15)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--testInterval", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--covidxRatio", type=float, default=1)
    parser.add_argument("--chestRatio", type=float, default=1)
    parser.add_argument("--saveDir", type=str)

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

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                normalize])


    testset = datasets.COVIDX(img_path=args.covidxTestImagePath, file_path=args.covidxTestFilePath, augment=transforms, num_class=args.numClass)

    dataloader=torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=12, pin_memory=True)

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
    auc = computeAUROC(target, predict, args.numClass)
    print(f'\n {auc}  avg_auc: {np.average(auc)} \n')
    # print(predict[:10, :])
    # input(666)
    predict, target = predict[:, 0] > 0.0001, target[:, 0]
    print(confusion_matrix(target, predict))
    print('acc:', accuracy_score(target, predict), 'precision:', precision_score(target, predict), 'sensitivity:', recall_score(target, predict))

    return np.average(auc), auc



if __name__=='__main__':
    parser = get_arguments()
    args = parser.parse_args()
    base_url = '/home/PJLAB/zhuzengle/workstation/Assemble/AssembleCovid2/all/epoch_%d'
    for i in range(2):# range(len(os.listdir('/home/PJLAB/zhuzengle/workstation/Assemble/AssembleCovid2/all'))):
        model_url = base_url %i
        print(model_url)
        model = torch.load(model_url)
        test(model, args)

