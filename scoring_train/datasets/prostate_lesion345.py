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

class Prostate_lesionDataset345(Dataset):

    def __init__(self, sets, phase='train'):
        self.root_dir = sets.root_path
        self.root_test_dir = sets.root_test_path
        self.input_D = sets.sample_duration
        self.input_H = sets.sample_size
        self.input_W = sets.sample_size
        self.phase = phase
        self.classes = sets.n_classes
        self.PI_RADS_type = sets.PI_RADS_type
        self.center_crop = sets.center_crop
        self.flip = sets.flip
        self.rot = sets.rot
        self.resize_select = sets.resize_select
        
        # self.input_D = 14
        # self.input_H = 112
        # self.input_W = 112
        # self.phase = "train"

        train_file = open("/media/ttzhang/5TDISK/prostate/xiaofzhao_server/ttzhang/3D-ResNet/data/train_1121_new.txt")
        test_file = open("/media/ttzhang/5TDISK/prostate/xiaofzhao_server/ttzhang/3D-ResNet/data/test_1121_seg.txt")
        inf_file = open("/media/ttzhang/5TDISK/prostate/xiaofzhao_server/ttzhang/3D-ResNet/data/test_1121_seg.txt")
        self.train_nimage_list = []
        self.train_label_list = []
        self.test_nimage_list = []
        self.test_label_list = []
        self.inf_nimage_list = []
        self.inf_label_list = []
        self.inf_slice_list = []
        self.data_type = sets.data_name
        for _train in train_file:
            self.train_nimage_list.append(_train[:-1].split(" ")[0])
            self.train_label_list.append(int(_train[:-1].split(" ")[1]))
        for _test in test_file:
            self.test_nimage_list.append(_test[:-1].split(" ")[0])
            self.test_label_list.append(int(_test[:-1].split(" ")[1]))
        for _inf in inf_file:
            self.inf_nimage_list.append(_inf[:-1].split(" ")[0])
            self.inf_label_list.append(int(_inf[:-1].split(" ")[1]))
            # self.inf_slice_list.append(int(_inf[:-1].split(" ")[3]))
        

    def __nii2tensorarray__(self, T2W_array, ADC_array, DWI_array):
        assert T2W_array.shape == ADC_array.shape == DWI_array.shape
        new_data = np.stack((T2W_array, ADC_array, DWI_array), axis=0)
        new_data = new_data.astype("float32")
            
        return new_data
    
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
            # print(self.train_nimage_list[idx])
            case_name = self.train_nimage_list[idx].split('_')[0]
            T2W_name = self.root_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.train_nimage_list[idx] + ".nii.gz"
            ADC_name = self.root_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.train_nimage_list[idx] + ".nii.gz"
            DWI_name = self.root_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.train_nimage_list[idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            if self.classes == 3:
                if self.train_label_list[idx] == 3:
                    target = 0
                elif self.train_label_list[idx] == 4:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:
                if self.train_label_list[idx] == 4 or self.train_label_list[idx] == 5:
                    target = 1
                else:
                    target = 0
            elif self.classes == 4:
                if self.train_label_list[idx] == -1:
                    target = 0
                elif self.train_label_list[idx] == 3:
                    target = 1
                elif self.train_label_list[idx] == 4:
                    target = 2
                else:
                    target = 3

            else:
                raise ValueError("error")


            # target = self.train_label_list[idx] - 1 #################classç¼–å·è¦ä»Ž0å¼€å§‹  æ‰€ä»¥æ˜¯0-4

            T2W_array, ADC_array, DWI_array = self.__training_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)


            return img_array, target
        
        elif self.phase == "test":
            # read image
            case_name = self.test_nimage_list[idx].split('_')[0]
            T2W_name = self.root_test_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            ADC_name = self.root_test_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"
            DWI_name = self.root_test_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.test_nimage_list[
                idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            if self.classes == 3:
                if self.test_label_list[idx] == 3:
                    target = 0
                elif self.test_label_list[idx] == 4:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:

                    if self.test_label_list[idx] == 4 or self.test_label_list[idx] == 5:
                        target = 1
                    else:
                        target = 0
            elif self.classes == 4:
                if self.test_label_list[idx] == -1:
                    target = 0
                elif self.test_label_list[idx] == 3:
                    target = 1
                elif self.test_label_list[idx] == 4:
                    target = 2
                else:
                    target = 3

            else:
                raise ValueError("error")

            # data processing
            T2W_array, ADC_array, DWI_array = self.__testing_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)

            return img_array, target
        else:
            # read image
            case_name = self.inf_nimage_list[idx].split('_')[0]
            T2W_name = self.root_dir + "/" + case_name + "/" + case_name + "_T2W" + "/" + self.inf_nimage_list[
                idx] + ".nii.gz"
            ADC_name = self.root_dir + "/" + case_name + "/" + case_name + "_ADC" + "/" + self.inf_nimage_list[
                idx] + ".nii.gz"
            DWI_name = self.root_dir + "/" + case_name + "/" + case_name + "_DWI" + "/" + self.inf_nimage_list[
                idx] + ".nii.gz"

            T2W = nibabel.load(T2W_name)
            ADC = nibabel.load(ADC_name)
            DWI = nibabel.load(DWI_name)

            if self.classes == 3:
                if self.inf_label_list[idx] == 3:
                    target = 0
                elif self.inf_label_list[idx] == 4:
                    target = 1
                else:
                    target = 2
            elif self.classes == 2:

                if self.inf_label_list[idx] == 4 or self.inf_label_list[idx] == 5:
                    target = 1
                else:
                    target = 0
            elif self.classes == 4:
                if self.inf_label_list[idx] == -1:
                    target = 0
                elif self.inf_label_list[idx] == 3:
                    target = 1
                elif self.inf_label_list[idx] == 4:
                    target = 2
                else:
                    target = 3

            else:
                raise ValueError("error")

            # data processing
            # T2W_array, ADC_array, DWI_array = self.__inference_data_process__(T2W, ADC, DWI, self.inf_slice_list[idx])
            T2W_array, ADC_array, DWI_array = self.__inference_data_process__(T2W, ADC, DWI)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(T2W_array, ADC_array, DWI_array)

            return img_array, self.inf_nimage_list[idx], target

    def __random_center_crop__(self, data, rzmin, rzmax, rymin, rymax, rxmin, rxmax):
        # from random import random
        """
        Random crop
    
        """
        label= np.zeros_like(data)
        [img_d, img_h, img_w] = data.shape
        # print([img_d, img_h, img_w])
        label[:, int(img_h*0.25): int(img_h*0.75), int(img_w*0.25): int(img_w*0.75)] = 1
        # print(label.shape)

        target_indexs = np.where(label > 0)
        
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        # print([target_depth, target_height, target_width])
        # Z_min = int((min_D - target_depth * 1.0 / 2) * rzmin)
        Y_min = int((min_H - target_height * 1.0 / 2) * rymin)
        X_min = int((min_W - target_width * 1.0 / 2) * rxmin)

        # Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * rzmax))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * rymax))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * rxmax))

        # Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        # Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        # Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        # Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[:, Y_min: Y_max, X_min: X_max]

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

    def __resize_data__(self, data, resize_select=False, begin_slice=0):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        if resize_select:
            if depth > self.input_D:
                data = data[begin_slice:begin_slice+self.input_D, :, :]



        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, rzmin, rzmax, rymin, rymax, rxmin, rxmax):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data= self.__random_center_crop__ (data, rzmin, rzmax, rymin, rymax, rxmin, rxmax)
        
        return data

    def __training_data_process__(self, T2W, ADC, DWI):
        # crop data according net input size
        from random import random
        from random import choice
        T2W = T2W.get_fdata()
        ADC = ADC.get_fdata()
        DWI = DWI.get_fdata()


        if self.flip:
            if random() > 0.5:
                T2W = np.fliplr(T2W)
                ADC = np.fliplr(ADC)
                DWI = np.fliplr(DWI)
            if random() > 0.5:
                T2W = np.flipud(T2W)
                ADC = np.flipud(ADC)
                DWI = np.flipud(DWI)
        if self.rot:
            rot_list = [1,2,3]
            degree = choice(rot_list)
            if random() > 0.5:
                T2W = np.rot90(T2W, degree)
                ADC = np.rot90(ADC, degree)
                DWI = np.rot90(DWI, degree)
            

        T2W = np.transpose(T2W, [2, 0, 1])
        ADC = np.transpose(ADC, [2, 0, 1])
        DWI = np.transpose(DWI, [2, 0, 1])

        if self.center_crop:

            if random() > 0.5:
                rzmin = random()
                rzmax = random()
                rymin = random()
                rymax = random()
                rxmin = random()
                rxmax = random()

                # center crop

                T2W = self.__crop_data__(T2W, rzmin, rzmax, rymin, rymax, rxmin, rxmax)
                ADC = self.__crop_data__(ADC, rzmin, rzmax, rymin, rymax, rxmin, rxmax)
                DWI = self.__crop_data__(DWI, rzmin, rzmax, rymin, rymax, rxmin, rxmax)


        # resize data
        if self.resize_select:
            [depth, height, width] = T2W.shape
            if depth > self.input_D:
                begin_slice = np.random.randint(low=0, high=depth-self.input_D+1)  # [low, high)
            else:
                begin_slice = 0
            T2W = self.__resize_data__(T2W, self.resize_select, begin_slice)
            ADC = self.__resize_data__(ADC, self.resize_select, begin_slice)
            DWI = self.__resize_data__(DWI, self.resize_select, begin_slice)
        else:

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
        T2W = T2W.get_fdata()
        ADC = ADC.get_fdata()
        DWI = DWI.get_fdata()

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
    def __inference_data_process__(self, T2W, ADC, DWI, slice=None):
        # crop data according net input size
        T2W = T2W.get_fdata()
        ADC = ADC.get_fdata()
        DWI = DWI.get_fdata()

        T2W = np.transpose(T2W, [2, 0, 1])
        ADC = np.transpose(ADC, [2, 0, 1]) 
        DWI = np.transpose(DWI, [2, 0, 1])

        # resize data
        if slice is not None:
            T2W = self.__resize_data__(T2W, resize_select=True, begin_slice=slice)
            ADC = self.__resize_data__(ADC, resize_select=True, begin_slice=slice)
            DWI = self.__resize_data__(DWI, resize_select=True, begin_slice=slice)
        else:
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
