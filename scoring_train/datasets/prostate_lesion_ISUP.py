'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

class Prostate_lesionDataset_ISUP(Dataset):

    def __init__(self, sets, phase='train'):
        self.root_dir = sets.root_path
        self.input_D = sets.sample_duration
        self.input_H = sets.sample_size
        self.input_W = sets.sample_size
        self.phase = phase
        self.classes = sets.n_classes
        self.pca_flag = sets.pca_flag
        # self.input_D = 14
        # self.input_H = 112
        # self.input_W = 112
        # self.phase = "train"

        train_file = open("/public/xiaofzhao4/ttzhang/3D-ResNet/data/train_PI-RADS_ISUP_manxi.txt")
        test_file = open("/public/xiaofzhao4/ttzhang/3D-ResNet/data/test_PI-RADS_ISUP_manxi.txt")
        self.train_nimage_list = []
        self.train_label_list = []
        self.test_nimage_list = []
        self.test_label_list = []
        self.data_type = sets.data_name
        for _train in train_file:
            self.train_nimage_list.append(_train[:-1].split(" ")[0])
            self.train_label_list.append(int(_train[:-1].split(" ")[2]))
        for _test in test_file:
            self.test_nimage_list.append(_test[:-1].split(" ")[0])
            self.test_label_list.append(int(_test[:-1].split(" ")[2]))

    def __nii2tensorarray__(self, T2W_array, ADC_array, DWI_array):
        assert T2W_array.shape == ADC_array.shape == DWI_array.shape
        new_data = np.stack((T2W_array, ADC_array, DWI_array), axis=0)
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        if self.phase == "train":
            return len(self.train_nimage_list)
        else:
            return len(self.test_nimage_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            case_name = self.train_nimage_list[idx].split('_')[0]
            T2W_name = self.root_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.train_nimage_list[idx] + ".nii.gz"
            ADC_name = self.root_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.train_nimage_list[idx] + ".nii.gz"
            DWI_name = self.root_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.train_nimage_list[idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            # if self.train_label_list[idx] == 0:
            #     target = 0
            # elif self.train_label_list[idx] == 1:
            #     target = 1
            # else:
            #     target = 2
            if self.classes == 3:
                if self.train_label_list[idx] == 0:
                    target = 0
                elif self.train_label_list[idx] == 1:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:
                if self.pca_flag:
                    if self.train_label_list[idx] == 0:
                        target = 0
                    else:
                        target = 1
                else:
                    if self.train_label_list[idx] == 0 or self.train_label_list[idx] == 1:
                        target = 0
                    else:
                        target = 1
            else:
                target = self.train_label_list[idx]


            # target = self.train_label_list[idx] - 1 #################class编号要从0开始  所以是0-4

            T2W_array, ADC_array, DWI_array = self.__training_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)


            return img_array, target
        
        elif self.phase == "test":
            # read image
            case_name = self.test_nimage_list[idx].split('_')[0]
            T2W_name = self.root_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            ADC_name = self.root_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            DWI_name = self.root_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            # if self.test_label_list[idx] == 0:
            #     target = 0
            # elif self.test_label_list[idx] == 1:
            #     target = 1
            # else:
            #     target = 2
            if self.classes == 3:
                if self.test_label_list[idx] == 0:
                    target = 0
                elif self.test_label_list[idx] == 1:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:
                if self.pca_flag:
                    if self.test_label_list[idx] == 0:
                        target = 0
                    else:
                        target = 1
                else:
                    if self.test_label_list[idx] == 0 or self.test_label_list[idx] == 1:
                        target = 0
                    else:
                        target = 1
            else:
                target = self.test_label_list[idx]


            # data processing
            T2W_array, ADC_array, DWI_array = self.__testing_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)

            return img_array, target

        else:
            # read image
            case_name = self.test_nimage_list[idx].split('_')[0]
            T2W_name = self.root_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            ADC_name = self.root_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            DWI_name = self.root_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            # if self.test_label_list[idx] == 0:
            #     target = 0
            # elif self.test_label_list[idx] == 1:
            #     target = 1
            # else:
            #     target = 2
            if self.classes == 3:
                if self.test_label_list[idx] == 0:
                    target = 0
                elif self.test_label_list[idx] == 1:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:
                if self.pca_flag:
                    if self.test_label_list[idx] == 0:
                        target = 0
                    else:
                        target = 1
                else:
                    if self.test_label_list[idx] == 0 or self.test_label_list[idx] == 1:
                        target = 0
                    else:
                        target = 1
            else:
                target = self.test_label_list[idx]

            # data processing
            T2W_array, ADC_array, DWI_array = self.__testing_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)
            return img_array, self.test_nimage_list[idx], target

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        # out_random = np.random.normal(0, 1, size = volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        
        return data, label

    def __training_data_process__(self, T2W, ADC, DWI):
        # crop data according net input size
        T2W = T2W.get_data()
        ADC = ADC.get_data()
        DWI = DWI.get_data()

        T2W = np.transpose(T2W, [2, 0, 1])
        ADC = np.transpose(ADC, [2, 0, 1])
        DWI = np.transpose(DWI, [2, 0, 1])


        # resize data
        T2W = self.__resize_data__(T2W)
        ADC = self.__resize_data__(ADC)
        DWI = self.__resize_data__(DWI)

        # normalization datas
        T2W = self.__itensity_normalize_one_volume__(T2W)
        ADC = self.__itensity_normalize_one_volume__(ADC)
        DWI = self.__itensity_normalize_one_volume__(DWI)

        return T2W, ADC, DWI


    def __testing_data_process__(self, T2W, ADC, DWI):
        # crop data according net input size
        T2W = T2W.get_data()
        ADC = ADC.get_data()
        DWI = DWI.get_data()

        T2W = np.transpose(T2W, [2, 0, 1])
        ADC = np.transpose(ADC, [2, 0, 1]) 
        DWI = np.transpose(DWI, [2, 0, 1])

        # resize data
        T2W = self.__resize_data__(T2W)
        ADC = self.__resize_data__(ADC)
        DWI = self.__resize_data__(DWI)

        # normalization datas
        T2W = self.__itensity_normalize_one_volume__(T2W)
        ADC = self.__itensity_normalize_one_volume__(ADC)
        DWI = self.__itensity_normalize_one_volume__(DWI)

        return T2W, ADC, DWI

if __name__ == '__main__':
    dataset = Prostate_lesionDataset('/public/ttzhang9/dataset/prostate/Case_input', sets)
    print(len(dataset.train_nimage_list))
    print(len(dataset.train_label_list))
    for data in dataset:
        print(data[0].shape, data[1])
