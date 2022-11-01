from cProfile import label
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from imageio import imread
from PIL import Image
import random
from random import sample
from torchvision import transforms
import torchvision
from random import shuffle

class COVIDX(Dataset):
    def __init__(self, img_path, file_path, augment, num_class, ratio=1):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.num_class = num_class
        self.source = []
        with open(file_path, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(img_path, lineItems[1])
                    self.img_list.append(imagePath)
                    imageLabel = [0, ] * num_class
                    if lineItems[2] == 'positive':
                        imageLabel = [1, ] + [0, ] * num_class
                        # self.img_label.append(1)
                    else:
                        imageLabel = [0,] + [0, ] * num_class
                        # self.img_label.append(0)
                    self.img_label.append(imageLabel)

                    self.source.append(0)
        
        img_list = []
        source = []
        img_label = []


        total = len(self.img_list)
        index = list(range(len(self.source)))
        shuffle(index)
        for i, idx in enumerate(index):
            if i / total > ratio:
                break
            img_list.append(self.img_list[idx])
            source.append(self.source[idx])
            img_label.append(self.img_label[idx])
        
        self.img_list = img_list
        self.source = source
        self.img_label = img_label
        

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self):
        return len(self.img_list)
    def get_labels(self, idx):
        
        return self.img_label[idx].index(1)





class ChestXRay14(Dataset):

    def __init__(self, img_path, file_path, augment, num_class, label_assembles, ratio=1):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.source = []
        self.label_mappings = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4, 'Nodule': 5,
                       'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 9,
                       'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}
        for label in label_assembles:
            assert label in self.label_mappings.keys(), 'No such label!'

        label_assembles_mappings = [self.label_mappings[label_assemble] for label_assemble in label_assembles]
        with open(file_path, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(img_path, lineItems[0])
                    imageLabelTotal = lineItems[1:]
                    imageLabelTotal = [int(i) for i in imageLabelTotal]
                    # input(len(imageLabelTotal))
                    imageLabel = [imageLabelTotal[label_assemble_mapping] for label_assemble_mapping in label_assembles_mappings]
                    imageLabel = [0, ] + imageLabel                
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)
                    self.source.append(1)
        
        img_list = []
        source = []
        img_label =[]
        total = len(self.img_list)
        index = list(range(len(self.source)))
        shuffle(index)
        for i, idx in enumerate(index):
            if i / total > ratio:
                break
            img_list.append(self.img_list[idx])
            source.append(self.source[idx])
            img_label.append(self.img_label[idx])
        self.img_list = img_list
        self.img_label = img_label
        self.source = source
        

    def __getitem__(self, index):

        imagePath = self.img_list[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel, 0

    def __len__(self):

        return len(self.img_list)


class Assemble(Dataset):
    def __init__(self, img_paths, file_paths, augments, label_assembles, num_class, covidx_ratio=1, chest_ratio=1):

        cxr3 = COVIDX(img_paths[0], file_paths[0], augments[0], num_class=num_class, ratio=covidx_ratio)
        chexpert_ray14 = ChestXRay14(img_paths[1], file_paths[1], augments[0], num_class=num_class, label_assembles=label_assembles, ratio=chest_ratio)

        self.img_list = cxr3.img_list + chexpert_ray14.img_list
        self.img_label = cxr3.img_label + chexpert_ray14.img_label
        self.source = cxr3.source + chexpert_ray14.source
        self.augments = augments


    def __getitem__(self, index):

        imagePath = self.img_list[index]


        image1 = Image.open(imagePath).convert('RGB')
        image2 = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augments != None:
            image1 = self.augments[0](image1)
            image2 = self.augments[1](image2)

        return image1, imageLabel, self.source[index], image2



    def __len__(self):
        return len(self.img_list)
        
    def get_labels(self, idx):
        if self.source[idx] == 0:
            if self.img_label[idx][0] == 0:
                return 0
            else:
                return 1
        else:
            if sum(self.img_label[idx]) == 0:
                return 2
            else:
                return self.img_label[idx].index(1) + 2

        # return self.img_label[idx].index(1)

if __name__ == '__main__':
    assemble = Assemble(['/home/PJLAB/zhuzengle/workstation/Assemble/dataset', '/home/PJLAB/zhuzengle/workstation/Assemble/images/images/train'], 
    ['/home/PJLAB/zhuzengle/workstation/Assemble/dataset/train.txt', '/home/PJLAB/zhuzengle/workstation/Assemble/images/train.txt'], augments=[None, None],
    covidx_ratio=1, chest_ratio=0, label_assembles=['Edema'], num_class=3)
    print(len(assemble.img_list), len(assemble.img_label))
