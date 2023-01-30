import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from utils import sample
from PIL import ImageFile
from abc import abstractmethod
from config import *
ImageFile.LOAD_TRUNCATED_IMAGES = True



class BaseDataset(Dataset):
    
    def __init__(self, dataset_config:dict, mode:str, source:int, augment:transforms, is_sample=True):
        self.augment = augment
        self.class_filter = dataset_config['class_filter']
        self.img_list, self.img_label, self.source = [], [], []
        img_dir, file_path = dataset_config['%s_img_path' % mode], dataset_config['%s_file_path' % mode]
        with open(file_path, "r") as fileDescriptor:
            lines = fileDescriptor.readlines()
        for line in lines:
            line = line.strip()
            img_path, img_label = self.parse_line(line, img_dir)  
            # single class
            if isinstance(img_label, str):
                img_label = [img_label]
            # multiple class
            if isinstance(img_label, list):
                for label in img_label:
                    if label in dataset_config['class_filter'] or label == 'health':     
                        self.img_list.append(img_path)
                        self.img_label.append(img_label)
                        self.source.append(source)
                        break
        # sample from dataset
        class_filter_mapping = [class_mapping[class_filter] for class_filter in self.class_filter]
        sorted_ids = sorted(range(len(class_filter_mapping)), key=lambda k: class_filter_mapping[k], reverse=False)
        self.label_mapping = {}
        self.labels = []
        for i, sorted_id in enumerate(sorted_ids):
            self.labels.append(self.class_filter[sorted_id])
            self.label_mapping[self.class_filter[sorted_id]] = i
        if is_sample:
            self.img_list, self.img_label, self.source = sample(self.img_list, self.img_label, self.source, dataset_config['using_num'])

    def __getitem__(self, index:int):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        gt = torch.FloatTensor([0, ] * len(self.class_filter))
        img_label = self.img_label[index]
        for label in img_label:
            if label == 'health':
                break
            if label not in self.label_mapping.keys():
                continue
            gt[self.label_mapping[label]] = 1.
        # imageLabel = torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            img = self.augment(img)
        return img, gt

    def __len__(self)->int:
        return len(self.img_list)
    
    @abstractmethod
    def parse_line(self, line, img_dir):
        raise NotImplemented
        


class COVIDX(BaseDataset):
    def __init__(self, dataset_config:dict, mode:str, source:int, augment:transforms, is_sample=True):
        super().__init__(dataset_config, mode, source, augment, is_sample)
        
    def parse_line(self, line, img_dir):
        line_item = line.split(' ')
        img_path = os.path.join(img_dir, line_item[1])
        if line_item[2] == 'positive':
            img_label = 'CovidPositive'
        else:
            img_label = 'health'
        return img_path, img_label

class ChestXRay14(BaseDataset):
    def __init__(self, dataset_config:dict, mode:str, source:int, augment:transforms, is_sample=True):
        super().__init__(dataset_config, mode, source, augment, is_sample)

    def parse_line(self, line, img_dir):
        gt2label = {0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Effusion', 
                          3: 'Infiltration', 4: 'Mass', 5: 'Nodule', 
                          6: 'Pneumonia', 7: 'Pneumothorax', 8: 'Consolidation', 
                          9: 'Edema', 10: 'Emphysema', 11: 'Fibrosis', 
                          12: 'Pleural_Thickening', 13: 'Hernia'}
        line_item = line.split(' ')
        img_path = os.path.join(img_dir, line_item[0])
        gts = [int(i) for i in line_item[1:]]
        img_label = []
        for i, gt in enumerate(gts):
            if gt:
                img_label.append(gt2label[i])
        return img_path, img_label
    
        
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
        self.class_filter = []
        for dataset in datasets:
            self.img_list.extend(dataset.img_list)
            self.img_label.extend(dataset.img_label)
            self.source.extend(dataset.source)
            self.class_filter.extend(dataset.class_filter)
        self.class_filter = list(set(self.class_filter))
        class_filter_mapping = [class_mapping[class_filter] for class_filter in self.class_filter]
        sorted_ids = sorted(range(len(class_filter_mapping)), key=lambda k: class_filter_mapping[k], reverse=False)
        self.label_mapping = {}
        self.labels = []
        for i, sorted_id in enumerate(sorted_ids):
            self.labels.append(self.class_filter[sorted_id])
            self.label_mapping[self.class_filter[sorted_id]] = i

    def __getitem__(self, index:int):
        img_path = self.img_list[index]
        image1 = Image.open(img_path).convert('RGB')
        image2 = Image.open(img_path).convert('RGB')
        gt = torch.FloatTensor([0, ] * len(self.class_filter))
        img_label = self.img_label[index]
        for label in img_label:
            if label == 'health':
                break
            if label not in self.label_mapping.keys():
                continue
            gt[self.label_mapping[label]] = 1.
        if self.augments != None:
            image1 = self.augments[0](image1)
            image2 = self.augments[1](image2)

        return image1, gt, self.source[index], image2

    def __len__(self):
        return len(self.img_list)
