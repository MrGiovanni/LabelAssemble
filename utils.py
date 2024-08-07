import torch
import torchvision.transforms
import numpy as np
from sklearn.metrics._ranking import roc_auc_score
from random import shuffle

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset.get_labels(idx)
           

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def sample(img_list:list, img_label:list, source:list, select_num:int)->list:
    """sample from the whole dataset

    Args:
        img_list (list): image list
        img_label (list): image label
        source (list): dataset source
        select_num (int): the number of images to be trained

    Returns:
        list: [sample_img_list, sample_img_label, sample_source]
    """    
    sample_img_list = []
    sample_source = []
    sample_img_label = []

    total = len(img_list)
    index = list(range(len(img_list)))
    shuffle(index)

    for i, idx in enumerate(index):
        if i  >= select_num:
            break
        sample_img_list.append(img_list[idx])
        sample_source.append(source[idx])
        sample_img_label.append(img_label[idx])
    return [sample_img_list, sample_img_label, sample_source]


import pandas as pd
