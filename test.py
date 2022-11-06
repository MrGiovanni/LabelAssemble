import datasets
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from noise import *
from config import device
from metrics import *


def test(model, args):

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                normalize])
    
    test_set = datasets.COVIDX(img_path=args.covidxTestImagePath,
                    file_path=args.covidxTestFilePath,
                    augment=None,
                    extra_num_class=args.extraNumClass,
                    select_num=args.covidxNum)
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=12, pin_memory=True)

    predict = []
    target = []
    model.eval()
    print(f'test now')

    for inputs, labels, _ in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs,fea=model(inputs)
            predict.append(outputs)
            target.append(labels)
    predict = torch.cat(predict, dim=0).cpu().numpy()
    target = torch.cat(target, dim=0).cpu().numpy()
    auc = compute_roc_auc(target, predict, args.numClass)
    print(f'\n {auc}  avg_auc: {np.average(auc)} \n')
    # print(predict[:10, :])
    # input(666)

    return np.average(auc), auc





