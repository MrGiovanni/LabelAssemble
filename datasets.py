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
from utils import sample





class COVIDX(Dataset):
    def __init__(self, img_path:str, file_path:str, augment:transforms, extra_num_class:int, select_num:int=30000)->None:
        """ COVIDx dataset
        Args:
            img_path (str): the path of the image files
            file_path (str): the path of the train/test file list
            augment (transforms): augumentation
            extra_num_class (int): Label-Assemble classes nums
            select_num (int, optional): nums of images involved in training. Defaults to 30000.
        """        
        
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.source = []
        with open(file_path, "r") as fileDescriptor:
            lines = fileDescriptor.readlines()
        for line in lines:
            line = line.strip()
            lineItems = line.split()
            imagePath = os.path.join(img_path, lineItems[1])
            self.img_list.append(imagePath)
            if lineItems[2] == 'positive':
                imageLabel = [1, ] + [0, ] * extra_num_class
            else:
                imageLabel = [0,] + [0, ] * extra_num_class
            self.img_label.append(imageLabel)
            self.source.append(0)
        # sample from dataset
        self.img_list, self.img_label, self.source = sample(self.img_list, self.img_label, self.source, select_num)
        
    
    def __getitem__(self, index:int)->Tensor, list, int:
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self)->int:
        return len(self.img_list)

    def get_labels(self, idx:int)->int:
        return self.img_label[idx].index(1)





class ChestXRay14(Dataset):

    def __init__(self, img_path:str, file_path:str, augment:transforms, label_assembles:list, select_num=110000)->None:
        """ ChestXRay14 dataset
        Args:
            img_path (str): the path of the image files
            file_path (str): the path of the train/test file list
            augment (transforms): augumentation
            label_assembles (list): Label-Assemble classes
            select_num (int, optional): nums of images involved in training. Defaults to 110000.
        """        
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
            lines = fileDescriptor.readlines()
        for line in lines:
            line = line.strip()
            lineItems = line.split()
            imagePath = os.path.join(img_path, lineItems[0])
            imageLabelTotal = [int(i) for i in lineItems[1:]]
            imageLabel = [0, ] + [imageLabelTotal[label_assemble_mapping] for label_assemble_mapping in label_assembles_mappings]            
            self.img_list.append(imagePath)
            self.img_label.append(imageLabel)
            self.source.append(1)
        
        self.img_list, self.img_label, self.source = sample(self.img_list, self.img_label, self.source, select_num)
        

    def __getitem__(self, index:int)->torch.Tensor, list, int:
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self)->int:
        return len(self.img_list)


class Assemble(Dataset):
    def __init__(self, img_paths:list, file_paths:list, augments:list, label_assembles:list, covidx_num:int, chest_num:int)->None:
        covidx = COVIDX(img_paths[0], file_paths[0], None, extra_class=len(label_assembles), select_num=covidx_num)
        chestxray14 = ChestXRay14(img_paths[1], file_paths[1], None, label_assembles=label_assembles, select_num=chest_num)
        self.img_list = covidx.img_list + chestxray14.img_list
        self.img_label = covidx.img_label + chestxray14.img_label
        self.source = covidx.source + chestxray14.source
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
