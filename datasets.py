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


class NIH_encod(Dataset):
    def __init__(self, img_path, list_path, transform, num_class):

        self.transform = transform
        self.img_path = img_path
        self.list_path=list_path
        self.num_class=num_class

        self.img_list = []
        self.img_label = []


        with open(self.list_path, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.img_path, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [float(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.img_pos = []
        self.label_pos = []
        self.encod_pos = []
        self.img_neg = []
        self.label_neg = []
        self.encod_neg = []
        self.img_unk = []
        self.label_unk = []
        self.encod_unk = []

        for i in range(len(self.img_list)):
            for j in range(self.num_class):
                tmp_encod = [0] * self.num_class
                tmp_encod[j] = 1
                if self.img_label[i][j] == 1:
                    self.img_pos.append(self.img_list[i])
                    self.label_pos.append(self.img_label[i][j])
                    self.encod_pos.append(tmp_encod)
                elif self.img_label[i][j] == 0:
                    self.img_neg.append(self.img_list[i])
                    self.label_neg.append(self.img_label[i][j])
                    self.encod_neg.append(tmp_encod)
                elif self.img_label[i][j] == -1:
                    self.img_unk.append(self.img_list[i])
                    self.label_unk.append(self.img_label[i][j])
                    self.encod_unk.append(tmp_encod)

        self.data = []
        self.labels = []
        self.encodings = []

        for x in self.img_pos:
            self.data.append(x)
        for x in self.img_neg:
            self.data.append(x)
        for x in self.img_unk:
            self.data.append(x)

        for x in self.label_pos:
            self.labels.append(x)
        for x in self.label_neg:
            self.labels.append(x)
        for x in self.label_unk:
            self.labels.append(x)

        for x in self.encod_pos:
            self.encodings.append(x)
        for x in self.encod_neg:
            self.encodings.append(x)
        for x in self.encod_unk:
            self.encodings.append(x)

        self.encodings = torch.tensor(self.encodings, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # if len(img.shape) == 3:
        #     img = np.moveaxis(img.numpy(), 0, 2)
        # elif len(img.shape) == 4:
        #     img = np.moveaxis(img.numpy(), 1, 3)


        label = self.labels[idx]
        encoding = self.encodings[idx]

        return img, label, encoding

    def __len__(self):
        return len(self.data)


class NIH(Dataset):
    def __init__(self, path_image, path_list, transform, num_class, reduct_ratio=1):
        self.path_list = path_list
        self.transform = transform
        self.path_image = path_image
        self.num_class=num_class

        self.img_list = []
        self.img_label = []

        #cnt=[55]*14

        with open(self.path_list, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [float(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)


                    # index=-1
                    # for id,i in enumerate(imageLabel):
                    #     k=float(i)
                    #     # if k>0:
                    #     #     cnt[int(id)]+=1
                    #     if k>0 and index==-1:
                    #         index=id
                    #     elif k>0 and index>=0:
                    #         index=-2
                    # if index>=0 and cnt[int(index)]>0:
                    #     cnt[int(index)]-=1
                    #
                    #     self.img_list.append(imagePath)
                    #     self.img_label.append(imageLabel)
        #print(cnt)

        random.seed(1)
        print(len(self.img_list))
        self.reduct_ratio = reduct_ratio
        self.img_list = np.array(self.img_list)
        self.img_label = np.array(self.img_label)
        index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        self.img_list = self.img_list[index]
        self.img_label = self.img_label[index]
        self.img_list = self.img_list.tolist()
        self.img_label = self.img_label.tolist()
        print(len(self.img_list))


        if len(self.img_list)>7000:
            cnt=400
            ind=3
            for i in range(len(self.img_label)):
                for j in range(14):
                    if j==ind and cnt>0 and self.img_label[i][j]==1:
                        cnt-=1
                    if j==ind and cnt==0 and self.img_label[i][j]==1:
                        self.img_label[i][j]=-1



        # cnt=[0]*14
        # for i in range(len(self.img_list)):
        #     for ind, j in enumerate(self.img_label[i]):
        #         if j==1:
        #             cnt[ind]+=1
        # print(cnt)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform is not None:

            img = self.transform(img)

        # if len(img.shape) == 3:
        #     img = np.moveaxis(img.numpy(), 0, 2)
        # elif len(img.shape) == 4:
        #     img = np.moveaxis(img.numpy(), 1, 3)
        label = torch.zeros((self.num_class),dtype=torch.float)

        for i in range(0, self.num_class):
            label[i] = self.img_label[idx][i]

        return img, label, 0

    def __len__(self):
        return len(self.img_list)

class Chexpert(Dataset):
    def __init__(self, path_image, path_list, transform, num_class, reduct_ratio=1):
        self.path_list = path_list
        self.transform = transform
        self.path_image = path_image
        self.num_class=num_class

        self.img_list = []
        self.img_label = []

        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        with open(self.path_list, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image, lineItems[0])

                    imageLabel = lineItems[5:5+14]

                    labels=[]
                    for idx,_ in enumerate(imageLabel):
                        # if idx not in [5,8,2,7]:
                        #     imageLabel[idx]=-1.0
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        imageLabel[idx]=self.dict[0][imageLabel[idx]]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)
                    #self.img_label.append(labels)


        # random.seed(1)
        # self.reduct_ratio = reduct_ratio
        # self.img_list = np.array(self.img_list)
        # self.img_label = np.array(self.img_label)
        # if len(self.img_list) > 100000:
        #     index = sample(range(len(self.img_list)), 7715)
        # else:
        #     index = sample(range(len(self.img_list)), len(self.img_list))
        # self.img_list = self.img_list[index]
        # self.img_label = self.img_label[index]
        # self.img_list = self.img_list.tolist()
        # self.img_label = self.img_label.tolist()
        #
        # cnt = [0] * 14
        # for i in range(len(self.img_list)):
        #     for ind, j in enumerate(self.img_label[i]):
        #         if j == 1:
        #             cnt[ind] += 1
        # print(cnt)
        # print(len(self.img_list))


    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform is not None:

            img = self.transform(img)


        # if len(img.shape) == 3:
        #     img = np.moveaxis(img.numpy(), 0, 2)
        # elif len(img.shape) == 4:
        #     img = np.moveaxis(img.numpy(), 1, 3)
        label = torch.zeros((self.num_class),dtype=torch.float)

        for i in range(0, self.num_class):
            label[i] = self.img_label[idx][i]

        return img, label, 0

    def __len__(self):
        return len(self.img_list)

'''
 NIH:(train:75312  test:25596)
 0:A 1:Cd 2:Ef 3:In 4:M 5:N 6:Pn 7:pnx 8:Co 9:Ed 10:Em 11:Fi 12:PT 13:H
 Chexpert:(train:223415 valid:235)
 0:NF 1:EC 2:Cd 3:AO 4:LL 5:Ed 6:Co 7:Pn 8:A 9:Pnx 10:Ef 11:PO 12:Fr 13:SD
 combined:
 0: Airspace Opacity(AO)	1: Atelectasis(A)	2:Cardiomegaly(Cd)	3:Consolidation(Co)
 4:Edema(Ed)	5:Effusion(Ef)	6:Emphysema(Em)	7:Enlarged Card(EC)	8:Fibrosis(Fi)	
 9:Fracture(Fr)	10:Hernia(H)	11:Infiltration(In)	12:Lung lession(LL)	13:Mas(M)	
 14:Nodule(N)	15:No finding(NF)	16:Pleural thickening(PT)	17:Pleural other(PO)	18:Pneumonia(Pn)	
 19:Pneumothorax(Pnx)	20:Support Devices(SD)
'''


class c_c(Dataset):
    def __init__(self, path_image_1, path_image_2, path_list_1, path_list_2, transform1, transform2, reduct_ratio=1):

        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        self.path_list_1 = path_list_1
        self.path_list_2 = path_list_2
        self.transform1 = transform1
        self.transform2 = transform2
        self.num_class=21

        self.img_list = []
        self.img_label = []
        self.source = []
        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.dict_nih2combine={0:1,1:2,2:5,3:11,4:13,5:14,6:18,7:19,8:3,9:4,10:6,11:8,12:16,13:10}
        self.dict_chex2combine={0:15,1:7,2:2,3:0,4:12,5:4,6:3,7:18,8:1,9:19,10:5,11:17,12:9,13:20}

        cnt=[55]*14

        with open(self.path_list_1, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(self.path_image_1, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]

                    # tmp_label=[-1]*21
                    # for i in range(14):
                    #     if i not in [2,8,7]: #[1,6,0,9,2,8,7]: #[2, 8,7]:
                    #         continue
                    #     tmp_label[self.dict_nih2combine[i]]=float(imageLabel[i])

                    index = -1
                    for id, i in enumerate(imageLabel):
                        k = float(i)
                        # if k>0:
                        #     cnt[int(id)]+=1
                        if k > 0 and index == -1:
                            index = id
                        elif k > 0 and index >= 0:
                            index = -2
                    if index >= 0 and cnt[int(index)] > 0:
                        cnt[int(index)] -= 1
                        self.img_list.append(imagePath)
                        tmp_label = [-1] * 21
                        tmp_label[self.dict_nih2combine[index]]=1
                        self.img_label.append(tmp_label)
                        self.source.append(0)

        random.seed(1)
        self.reduct_ratio = reduct_ratio
        self.img_list = np.array(self.img_list)
        self.img_label = np.array(self.img_label)
        self.source=np.array(self.source)
        index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        self.img_list = self.img_list[index]
        self.img_label = self.img_label[index]
        self.source = self.source[index]
        self.img_list = self.img_list.tolist()
        self.img_label = self.img_label.tolist()
        self.source=self.source.tolist()


        # index=sample(range(166739), len(self.img_list)*3)
        # cnt=-1

        cnt2=[55]*14

        with open(self.path_list_2, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                #cnt+=1
                if line  :#and cnt in index:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image_2, lineItems[0])
                    imageLabel = lineItems[5:5+14]


                    tmp_label=[-1]*21
                    index=-1
                    for idx,_ in enumerate(imageLabel):
                        if idx not in [3,1,12,4,11]:
                            continue
                        # if idx not in [5,8,2,7]: #[2,7,8,5,10,6,9]: #[5,8,2,7]:
                        #     continue
                        # if idx in [5,8]:
                        #     imageLabel[idx]=self.dict[0][imageLabel[idx]]
                        # elif idx in [2,6,10]:
                        #     imageLabel[idx]=self.dict[1][imageLabel[idx]]
                        # labels.append(float(imageLabel[idx]))
                        k=self.dict[0][imageLabel[idx]]
                        #tmp_label[self.dict_chex2combine[idx]]=self.dict[0][imageLabel[idx]]




                            # if k>0:
                            #     cnt[int(id)]+=1
                        if k > 0 and index == -1:
                            index = idx
                        elif k > 0 and index >= 0:
                            index = -2
                        if index>=0 and cnt2[index]>0:
                            cnt2[index]-=1
                        # if k > 0 and index == -1:
                        #     index = idx
                        # elif k > 0 and index >= 0:
                        #     index = -2
                        # if index>=0:
                        #     cnt2[index]+=1
                            self.img_list.append(imagePath)
                            tmp_label[self.dict_chex2combine[idx]] = k
                            self.img_label.append(tmp_label)
                            self.source.append(1)
        print(cnt2)
        self.img_label=torch.tensor(self.img_label)
        self.source=torch.tensor(self.source)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')

        if self.transform1 is not None:

            img = self.transform1(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        #
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]

        return img, self.img_label[idx], self.source[idx]

    def __len__(self):
        return len(self.img_list)

from glob import glob

class Covid_19(Dataset):
    def __init__(self, path, path_normal_txt, path_normal_img, transform, args, reduct_ratio=1):
        self.transform = transform
        self.img_list = []
        self.img_label = []

        filenames = glob(path + '/COVID/' + '*.png')
        tmp = [0]*(args.num_class)
        tmp[-1] = 1
        for i in filenames:
            self.img_list.append(i)
            self.img_label.append(tmp)

        # filenames = glob(path + '/Normal/' + '*.png')
        # for i in filenames:
        #     self.img_list.append(i)
        #     self.img_label.append(0)

        with open(path_normal_txt, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(path_normal_img, lineItems[0])
                    imageLabel = lineItems[1:14 + 1]
                    imageLabel = [int(i) for i in imageLabel]

                    if args.num_class==2:
                        if any(imageLabel[j]==1 for j in range(14)):
                            imageLabel = [1]
                        else:
                            imageLabel = [0]
                    if args.num_class==1:
                        imageLabel = []

                    imageLabel.append(0)
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        random.seed(1)
        self.reduct_ratio = reduct_ratio
        self.img_list = np.array(self.img_list)
        self.img_label = np.array(self.img_label)
        index = sample(range(len(self.img_list)), len(self.img_list) // reduct_ratio)
        self.img_list = self.img_list[index]
        self.img_label = self.img_label[index]
        self.img_list = self.img_list.tolist()
        self.img_label = self.img_label.tolist()

        cnt = 0
        for i in range(len(self.img_label)):
            if self.img_label[i][-1]==1:
                cnt+=1
        print(cnt)
        print(len(self.img_list)-cnt)

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.img_label[idx],dtype=torch.float)

        return img, label, 0








class CXR3(Dataset):
    def __init__(self, pathImageRoot, pathDatasetRoot, augment, num_class=16, mode='train', ratio=1):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.num_class = num_class
        self.source = []
        pathImageRoot = os.path.join(pathImageRoot, mode)

        pathDatasetRoot = os.path.join(pathDatasetRoot, mode + '.txt')
        pos_cnt, neg_cnt = 0, 0
        with open(pathDatasetRoot, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageRoot, lineItems[1])
                    # imageLabel = lineItems[1:num_class + 1]
                    # imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    imageLabel = [0, ] * num_class
                    if lineItems[2] == 'positive':
                        imageLabel = [1, 0]# [1, ] + [0, ] * 14
                        # self.img_label.append(1)
                    else:
                        imageLabel = [0, 0] # [0,] + [0, ] * 14
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

        
        # print(pos_cnt, neg_cnt)
        

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel = torch.zeros(self.num_class)
        # imageLabel[self.img_label[index]] = 1
        imageLabel = torch.FloatTensor(self.img_label[index])
        # torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self):
        return len(self.img_list)
    def get_labels(self, idx):
        
        return self.img_label[idx].index(1)


    




class CovidTest1(Dataset):
    def __init__(self, pathImageRoot, pathDatasetRoot, augment, num_class=3, mode='test'):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.num_class = num_class
        pathImageRoot = os.path.join(pathImageRoot, mode)

        pathDatasetRoot = os.path.join(pathDatasetRoot, mode + '.txt')
        pos_cnt, neg_cnt = 0, 0
        with open(pathDatasetRoot, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    # imagePath = os.path.join(pathImageRoot, lineItems[0])
                    # imageLabel = lineItems[1:num_class + 1]
                    # imageLabel = [int(i) for i in imageLabel]
                    if 'jpg' in lineItems[0] or 'png' in lineItems[0] or 'jpeg' in lineItems[0] or 'JPG' in lineItems[0] or 'PNG' in lineItems[0]:
                        self.img_list.append(lineItems[0])
                        imageLabel = [0, ] * num_class
                        imageLabel[int(lineItems[1])] = 1
                        self.img_label.append(imageLabel)
                    else:
                        self.img_list.append(lineItems[0] + ' ' + lineItems[1])
                        imageLabel = [0, ] * num_class
                        imageLabel[int(lineItems[2])] = 1
                        self.img_label.append(imageLabel)

        

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel = torch.zeros(self.num_class)
        # imageLabel[self.img_label[index]] = 1
        imageLabel = torch.FloatTensor(self.img_label[index])
        # torch.FloatTensor(self.img_label[index])
        if self.augment != None:
            imageData = self.augment(imageData)
        return imageData, imageLabel, 0

    def __len__(self):
        return len(self.img_list)
    def get_labels(self, idx):
        
        return self.img_label[idx].index(1)


class ChestX_ray14(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=16, anno_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.source = []

        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()

                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    imageLabel = [0, ] + imageLabel
                    # if sum(imageLabel) == 0:
                    #     continue
                    # imageLabel = [int(i) for i in imageLabel]
                    # self.img_list.append(imagePath)
                    # self.img_label.append(imageLabel)
                    # if sum(imageLabel) == 0:
                    #     continue
                    # if not (sum(imageLabel) == 0):
                    if imageLabel[11]:
                       self.img_list.append(imagePath)
                       self.img_label.append([0, 1])
                    else:
                       self.img_list.append(imagePath)
                       self.img_label.append([0, 0])
                    # imageLabel = [0, 0] + imageLabel
                    # self.img_list.append(imagePath)
                    # self.img_label.append(imageLabel)
                    self.source.append(1)

                    # Atelectasis, Edema, Effusion, Consolidation, Pneumothorax
# ['Atelectasis 0', 'Cardiomegaly 1', 'Effusion 2', 'Infiltration 3', 'Mass 4', 'Nodule 5',
#                        'Pneumonia 6', 'Pneumothorax 7', 'Consolidation 8', 'Edema 9',
#                        'Emphysema 10', 'Fibrosis 11', 'Pleural_Thickening 12', 'Hernia 13']

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(
                self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

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
    def __init__(self, pathImageRoots, pathDatasetRoots, augment, num_class=16, mode='train', ratio=1):

        cxr3 = CXR3(pathImageRoots[0], pathDatasetRoots[0], augment, mode=mode, num_class=num_class)
        chexpert_ray14 = ChestX_ray14(pathImageRoots[1], pathDatasetRoots[1], augment)
        length = len(chexpert_ray14.img_list)
        self.img_list = cxr3.img_list + chexpert_ray14.img_list[:int(length * ratio)]
        self.img_label = cxr3.img_label + chexpert_ray14.img_label[:int(length * ratio)]
        self.source = cxr3.source + chexpert_ray14.source
        self.augment = augment


    def __getitem__(self, index):

        imagePath = self.img_list[index]


        image1 = Image.open(imagePath).convert('RGB')
        image2 = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            image1 = self.augment[0](image1)
            image2 = self.augment[1](image2)


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
    assemble = Assemble(['dataset', 'images/images'], ['dataset', 'images/train.txt'], augment=None)
    print(len(assemble.img_list), len(assemble.img_label))
