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
from abc import abstractmethod
ImageFile.LOAD_TRUNCATED_IMAGES = True



class BaseDataset(Dataset):
    
    def __init__(self, dataset_config:dict, mode:str, num_classes:int, start_id:int, source:int, augment:transforms, is_sample=True):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.source = []
        self.label_mappings = self.label_mappings if self.label_mappings else {}
        img_dir, file_path = dataset_config['%s_img_path' % mode], dataset_config['%s_file_path' % mode]
        with open(file_path, "r") as fileDescriptor:
            lines = fileDescriptor.readlines()
        for line in lines:
            line = line.strip()
            img_path, img_label = self.parse_line(line, img_dir, start_id, num_classes)  
            self.img_list.append(img_path)
            self.img_label.append(img_label)
            self.source.append(source)
        self.filter(start_id, dataset_config['class_filter'], num_classes)
        # sample from dataset
        if is_sample:
            self.img_list, self.img_label, self.source = sample(self.img_list, self.img_label, self.source, dataset_config['using_num'])

    def __getitem__(self, index:int):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self)->int:
        return len(self.img_list)
    
    @abstractmethod
    def parse_line(self, line, img_dir, start_id, num_classes):
        raise NotImplemented
        
    @abstractmethod
    def filter(self, start_id, label_filters):
        raise NotImplemented


class COVIDX(BaseDataset):
    def __init__(self, dataset_config:dict, mode:str, num_classes:int, start_id:int, source:int, augment:transforms, is_sample=True):
        self.label_mappings = {'positive': 0, 'negative': 1}
        super().__init__(dataset_config, mode, num_classes, start_id, source, augment, is_sample)
        
    def parse_line(self, line, img_dir, start_id, num_classes):
        line_item = line.split(' ')
        img_path = os.path.join(img_dir, line_item[1])
        img_label = [0, ] * num_classes
        if line_item[2] == 'positive':
            img_label[start_id] = 1
        return img_path, img_label
    
    def filter(self, start_id, label_filters, num_classes):
        pass



class ChestXRay14(BaseDataset):
    
    def __init__(self, dataset_config:dict, mode:str, num_classes:int, start_id:int, source:int, augment:transforms, is_sample=True):
        self.label_mappings = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4, 'Nodule': 5,
                       'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 9,
                       'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}
        super().__init__(dataset_config, mode, num_classes, start_id, source, augment, is_sample)

    def parse_line(self, line, img_dir, start_id, num_classes):
        line_item = line.split(' ')
        img_path = os.path.join(img_dir, line_item[0])
        img_label = [int(i) for i in line_item[1:]]
        return img_path, img_label
    
    def filter(self, start_id, label_filters, num_classes):
        img_list_filter = []
        img_label_filter = []
        source_filter = []
        label_filter_mappings = [self.label_mappings[label_filter] for label_filter in label_filters]
        label_filter_mappings.sort()
        for i in range(len(self.source)):
            img_label = self.img_label[i]
            img_filter = [img_label[label_filter_mapping] for label_filter_mapping in label_filter_mappings] 
            if sum(img_filter) == 0:
                continue               
            img_label = [0, ] * start_id + [img_label[label_filter_mapping] for label_filter_mapping in label_filter_mappings] + [0, ] * (num_classes - len(label_filter_mappings) - start_id)
            img_label_filter.append(img_label)
            source_filter.append(self.source[i])
            img_list_filter.append(self.img_list[i])
        self.img_list, self.img_label, self.source = img_list_filter, img_label_filter, source_filter
        


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
        
