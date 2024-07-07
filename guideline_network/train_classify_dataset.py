'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class Prostate_lesionDataset_public(Dataset):

    def __init__(self, n_classes, phase='train'):

        self.phase = phase
        self.classes = n_classes
        self.root_dir = 'prostate/feature_input60'
        train_file = open("prostate/public_train.txt")
        test_file = open("prostate/public_test.txt")
        inf_file = open("prostate/public_test.txt")
        self.train_nimage_list = []
        self.train_label_list = []
        self.test_nimage_list = []
        self.test_label_list = []
        self.inf_nimage_list = []
        self.inf_label_list = []

        for _train in train_file:
            self.train_nimage_list.append(_train[:-1].split(" ")[0])
            self.train_label_list.append(int(_train[:-1].split(" ")[1]))
        for _test in test_file:
            self.test_nimage_list.append(_test[:-1].split(" ")[0])
            self.test_label_list.append(int(_test[:-1].split(" ")[1]))
        for _inf in inf_file:
            self.inf_nimage_list.append(_inf[:-1].split(" ")[0])
            self.inf_label_list.append(int(_inf[:-1].split(" ")[1]))
        
    
    def __len__(self):
        if self.phase == "train":
            return len(self.train_nimage_list)
        elif self.phase == "test":
            return len(self.test_nimage_list)
        else:
            return len(self.inf_nimage_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            case_name = self.train_nimage_list[idx].split('-Target')[0]
            target_id = self.train_nimage_list[idx].split('-Target')[1]
            # T2W_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_T2W_Target" +  target_id + ".npy"
            # ADC_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_ADC_Target" +  target_id + ".npy"
            # DWI_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_DWI_Target" +  target_id + ".npy"

            T2W_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_T2W_Target" +  target_id + ".npy"
            ADC_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_ADC_Target" +  target_id + ".npy"
            DWI_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_DWI_Target" +  target_id + ".npy"

            # T2W1 = np.load(T2W_name1)
            # ADC1 = np.load(ADC_name1)
            # DWI1 = np.load(DWI_name1)
            # T2W1 = torch.tensor(T2W1)
            # ADC1 = torch.tensor(ADC1)
            # DWI1 = torch.tensor(DWI1)
            T2W2 = np.load(T2W_name2)
            ADC2 = np.load(ADC_name2)
            DWI2 = np.load(DWI_name2)
            T2W2 = torch.tensor(T2W2)
            ADC2 = torch.tensor(ADC2)
            DWI2 = torch.tensor(DWI2)

            # T2W = T2W1 + T2W2
            # ADC = ADC1 + ADC2
            # DWI = DWI1 + DWI2

            # T2W = T2W[:, -1, :]
            # ADC = ADC[:, -1, :]
            # DWI = DWI[:, -1, :]
            # inp = torch.cat([T2W, ADC, DWI], 1)
            T2W2 = T2W2[:, -1, :]
            ADC2 = ADC2[:, -1, :]
            DWI2 = DWI2[:, -1, :]
            inp = torch.cat([T2W2, ADC2, DWI2], 1)

            target = self.train_label_list[idx] - 1
            return inp, target
        
        elif self.phase == "test":
            # read image
            case_name = self.test_nimage_list[idx].split('-Target')[0]
            target_id = self.test_nimage_list[idx].split('-Target')[1]
            # T2W_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_T2W_Target" +  target_id + ".npy"
            # ADC_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_ADC_Target" +  target_id + ".npy"
            # DWI_name1 = self.root_dir + "/" + case_name + "/feature15_" + case_name + "_DWI_Target" +  target_id + ".npy"

            T2W_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_T2W_Target" +  target_id + ".npy"
            ADC_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_ADC_Target" +  target_id + ".npy"
            DWI_name2 = self.root_dir + "/" + case_name + "/feature_" + case_name + "_DWI_Target" +  target_id + ".npy"

            # T2W1 = np.load(T2W_name1)
            # ADC1 = np.load(ADC_name1)
            # DWI1 = np.load(DWI_name1)
            # T2W1 = torch.tensor(T2W1)
            # ADC1 = torch.tensor(ADC1)
            # DWI1 = torch.tensor(DWI1)
            T2W2 = np.load(T2W_name2)
            ADC2 = np.load(ADC_name2)
            DWI2 = np.load(DWI_name2)
            T2W2 = torch.tensor(T2W2)
            ADC2 = torch.tensor(ADC2)
            DWI2 = torch.tensor(DWI2)

            # T2W = T2W1 + T2W2
            # ADC = ADC1 + ADC2
            # DWI = DWI1 + DWI2
            T2W2 = T2W2[:, -1, :]
            ADC2 = ADC2[:, -1, :]
            DWI2 = DWI2[:, -1, :]
            # inp = torch.cat([T2W, ADC, DWI], 1)
            inp = torch.cat([T2W2, ADC2, DWI2], 1)
            
            target = self.test_label_list[idx] - 1


            return inp, target

if __name__ == '__main__':
    # aa = torch.randn((1,4096,256))
    # print(aa.shape)
    # bb = F.avg_pool1d(aa, kernel_size=1, stride=2)
    # print(bb.shape)


    dataset = Prostate_lesionDataset_public(5,"train")
    print(len(dataset.train_nimage_list))
    print(len(dataset.train_label_list))
    a = 0
    b = 0
    c = 0
    for data in dataset:
        aaaaa = data[0].view(1, -1)
        print(aaaaa.shape, data[1])
        # print(type(data[0]))
        # if a < data[0].shape[1]:
        #     a = data[0].shape[1]
        # if b < data[1].shape[1]:
        #     b = data[1].shape[1]
        # if c < data[2].shape[1]:
        #     c = data[2].shape[1]
