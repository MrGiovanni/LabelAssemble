import torch
from torch.utils.data import Dataset
import os
from PIL import Image

from random import sample
from torchvision import transforms
from random import shuffle
from utils import sample
from torch import FloatTensor
from typing import Union, Tuple, List
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        
    
    def __getitem__(self, index:int)->Tuple[FloatTensor, list, int]:
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
        

    def __getitem__(self, index:int)->Tuple[torch.Tensor, list, int]:
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self)->int:
        return len(self.img_list)

class Assemble(Dataset):
    def __init__(self, datasets:list, augments:list)->None:
        """Assemble dataset

        Args:
            datasets (list): Assemble datasets
            augments (list): Dataset augments

        """
        self.img_list = []
        self.img_label = []
        self.source = []
        self.augments = augments
        for dataset in datasets:
            self.img_list.extend(dataset.img_list)
            self.img_label.extend(dataset.img_label)
            self.source.extend(dataset.source)

    def __getitem__(self, index:int)->Tuple[FloatTensor, FloatTensor, int, FloatTensor]:
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
        # TODO
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





if __name__ == '__main__':
    covidx = COVIDX(img_path='/Users/zenglezhu/code/dataset/COVIDX/train',
                    file_path='/Users/zenglezhu/code/dataset/COVIDX/train.txt',
                    augment=None,
                    extra_num_class=0,
                    select_num=10000)
    chest = ChestXRay14(img_path='/Users/zenglezhu/code/dataset/chestxray14/train',
                    file_path='/Users/zenglezhu/code/dataset/chestxray14/train_official.txt',
                    augment=None,
                    label_assembles=['Pneumonia'],
                    select_num=10000)
    assemble = Assemble([covidx, chest], None)
    print(len(assemble.source))


