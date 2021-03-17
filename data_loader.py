from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import sys
from random import sample
import math
import re

from cvtorchvision import cvtransforms

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


class NSP_RUL_WANG_DATALOAD(Dataset):
    def __init__(self, input_path, idx_list, raw_fpt, num_timestep):
        self.data_dir = []
        self.label = []
        input_dir = sorted_aphanumeric(os.listdir(input_path))

        for num, sub_dir in enumerate(input_dir):
            if not num+1 in idx_list:
                continue
            
            
            target_dir = os.path.join(input_path, sub_dir)
            target_image_list = os.listdir(target_dir)
            data_len = len(target_image_list)

            temp_dir = sorted_aphanumeric(target_image_list)

            fpt = raw_fpt[num]
            
            for i in range(num_timestep,data_len-num):
                time_dir = []
                if i <= fpt:
                    count = 1
                    # self.label.append(1.0)
                    # for j in range(num_timestep):
                    #     time_dir.append(os.path.join(input_path,sub_dir,temp_dir[i+j]))
                    # self.data_dir.append(time_dir)

                else:
                    self.label.append((data_len-i)/(data_len-fpt))

                    for j in range(num_timestep):
                        time_dir.append(os.path.join(input_path,sub_dir,temp_dir[i-100+j]))

                    self.data_dir.append(time_dir)

        
        self.label = np.array(self.label)
        self.label = self.label.astype('float16')



    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        train = []

        for j in range(len(self.data_dir[i])):
            NSP = cv2.imread(self.data_dir[i][j])

            NSP = np.array(NSP)
            NSP = np.transpose(NSP,(2,0,1))
            train.append(NSP)

        train = np.array(train)

        train = torch.from_numpy(train).float()
        return train, self.label[i]

class Dataload_Step1(Dataset):
    def __init__(self, input_path,condition, idx_list, ratio=0.05):
        self.data_dir = []
        self.label = []
        self.bearing_label = []

        for num in idx_list:
            target_dir = input_path + '/Bearing'+str(condition)+ '_' + str(num)+'/'
            
            target_csv_list = os.listdir(target_dir)
            target_csv_list = [filename for filename in target_csv_list if filename.startswith('acc')]

            dir_len = len(target_csv_list)

            cut_idx = int(dir_len * ratio)

            temp_dir = sorted(target_csv_list)
            
            temp_dir = temp_dir[:cut_idx]

            for filename in temp_dir:
                self.label.append(0)
                self.data_dir.append(os.path.join(target_dir,filename))
                self.bearing_label.append(num-1)

            temp_dir = sorted(target_csv_list)
            temp_dir = temp_dir[-(cut_idx)-1:-1]

            for filename in temp_dir:
                self.label.append(1)
                self.data_dir.append(os.path.join(target_dir,filename))
                self.bearing_label.append(num-1)

        self.label = np.asarray(self.label)
        self.bearing_label = np.asarray(self.bearing_label)

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        data = np.genfromtxt(self.data_dir[i], delimiter=',')

        data = data[:,4:]
        x = np.asarray(data[:,0])
        y = np.asarray(data[:,1])
        
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y, self.label[i], self.bearing_label[i]

class Dataload_XJTU_Step1(Dataset):
    def __init__(self, input_path,target_condition, num_bearings, idx_list, input_dimension, length, ratio=0.05):
        self.data_dir = []
        self.label = []
        self.bearing_label = []
        self.data_idx = []
        self.input_dimension = input_dimension
        self.length = length
        condition_list = [1, 2]
        
        for condition in condition_list:
            for num in range(1,num_bearings+1):
                if (condition == target_condition and num not in idx_list) or (len(idx_list)==1 and condition != target_condition):
                    continue

                # target_dir = input_path + '/Bearing'+str(condition)+ '_' + str(num)+'/'
                target_dir = input_path + '/condition'+str(condition) + '/Bearing'+str(condition)+ '_' + str(num)+'/'
                print(target_dir)
                target_csv_list = os.listdir(target_dir)
                # target_csv_list = [filename for filename in target_csv_list if filename.startswith('acc')]

                dir_len = len(target_csv_list)

                cut_idx = int(dir_len * ratio)

                temp_dir = sorted_aphanumeric(target_csv_list)
                            
                temp_dir = temp_dir[:cut_idx]

                for filename in temp_dir:
                    for i in range(length):
                        self.label.append(0)
                        self.data_dir.append(os.path.join(target_dir,filename))
                        # self.bearing_label.append(num-1)
                        self.bearing_label.append((condition-1)*num_bearings + num-1)
                        self.data_idx.append(i)

                temp_dir = sorted(target_csv_list)
                temp_dir = temp_dir[-(cut_idx)-1:-1]

                for filename in temp_dir:
                    for i in range(length):
                        self.label.append(1)
                        self.data_dir.append(os.path.join(target_dir,filename))
                        self.bearing_label.append((condition-1)*num_bearings + num-1)
                        # print((condition-1)*num_bearings + num-1)
                        self.data_idx.append(i)

        self.label = np.asarray(self.label)
        self.bearing_label = np.asarray(self.bearing_label)
        self.data_idx = np.asarray(self.data_idx)

        print(len(self.data_dir))

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        data = np.genfromtxt(self.data_dir[i], delimiter=',')

        x = np.asarray(data[:,0])
        y = np.asarray(data[:,1])

        x = x[ (self.data_idx[i]*self.input_dimension+1) : ((self.data_idx[i]+1)*self.input_dimension+1)]
        y = y[ (self.data_idx[i]*self.input_dimension+1) : ((self.data_idx[i]+1)*self.input_dimension+1)]

        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y, self.label[i], self.bearing_label[i]

class Dataload_Step1_FPT(Dataset):
    def __init__(self, input_path,transform, target_condition, num_bearings, idx_list, ratio=0.05):
        self.data_dir = []
        self.label = []
        
        # condition_list = [1, 2]
        condition_list = [target_condition]
        
        for condition in condition_list:
            for num in range(1,num_bearings+1):
                if (condition == target_condition and num not in idx_list) or (len(idx_list)==1 and condition != target_condition):
                    continue

            # for num in idx_list:
                target_dir = input_path + str(condition)+ '_' + str(num) + '/'
                # target_dir = input_path + str(num) + '/'

                print(target_dir)
                target_image_list = os.listdir(target_dir)
                data_len = len(target_image_list)

                sorted_image_list = sorted_aphanumeric(target_image_list)

                cut_idx = int(data_len * ratio)

                temp_dir = sorted_image_list[:cut_idx]

                for filename in temp_dir:
                    self.label.append(0)
                    self.data_dir.append(os.path.join(target_dir,filename))

                temp_dir = sorted_image_list[-(cut_idx)-1:-1]
                for filename in temp_dir:
                    self.label.append(1)
                    self.data_dir.append(os.path.join(target_dir,filename))
        
        self.transform = transform
        self.label = np.asarray(self.label)
        
    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        
        # print(self.data_dir[i], self.label[i])
        input_image = cv2.imread(self.data_dir[i])
        input_image = self.transform(input_image)
        input_label = self.label[i]
        
        return input_image, input_label

class Dataload_Step2(Dataset):
    def __init__(self, input_path, condition, idx_list, fpt_list, num_sequence):
        self.data_dir = []
        self.label = []
        self.bearing_label = []
        for num in idx_list:
            target_dir = input_path + '/Bearing'+str(condition)+ '_' + str(num)+'/'
            
            target_csv_list = os.listdir(target_dir)
            target_csv_list = [filename for filename in target_csv_list if filename.startswith('acc')]

            data_len = len(target_csv_list)
            temp_dir = sorted_aphanumeric(target_csv_list)

            fpt = fpt_list[num-1]

            for index in range(fpt,data_len,num_sequence):
                if index < data_len - num_sequence:
                    dir_seq = []
                    for seq in range(num_sequence):
                        dir_seq.append(os.path.join(target_dir,temp_dir[index+seq]))
                        
                    self.label.append((data_len-index)/(data_len-fpt))
                    self.data_dir.append(dir_seq)
                    self.bearing_label.append(num-1)
                    # self.data_dir.append(os.path.join(target_dir,temp_dir[index]))
                
        self.label = np.asarray(self.label)
        self.bearing_label = np.asarray(self.bearing_label)
        
        self.num_sequence=num_sequence

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        x_sequence = []
        y_sequence = []
        bearing_label_seq = []
        for data_idx in range(self.num_sequence):
            data = np.genfromtxt(self.data_dir[i][data_idx], delimiter=',')
            data = data[:,4:]
            x = np.asarray(data[:,0])
            y = np.asarray(data[:,1])        
            x = x.reshape(1,-1)
            y = y.reshape(1,-1)
            x_sequence.append(x)
            y_sequence.append(y)
            bearing_label_seq.append(self.bearing_label[i])
        
        x_sequence = np.array(x_sequence)
        y_sequence = np.array(y_sequence)
        bearing_label_seq = np.array(bearing_label_seq)

        x_sequence = torch.from_numpy(x_sequence).float()
        y_sequence = torch.from_numpy(y_sequence).float()
        bearing_label_seq = torch.from_numpy(bearing_label_seq).float()

        return x_sequence, y_sequence, self.label[i], bearing_label_seq

class Dataload_XJTU_Step2(Dataset):
    def __init__(self, input_path, target_condition, num_bearings, idx_list, fpt_list, num_sequence, input_dimension, length):
        self.data_dir = []
        self.label = []
        self.bearing_label = []
        self.data_idx = []
        self.input_dimension = input_dimension
        self.length = length

        condition_list = [1, 2]
        
        for condition in condition_list:
            for num in range(1,num_bearings+1):
                if (condition == target_condition and num not in idx_list) or (len(idx_list)==1 and condition != target_condition):
                    continue

                # target_dir = input_path + '/Bearing'+str(condition)+ '_' + str(num)+'/'
                target_dir = input_path + '/condition'+str(condition) + '/Bearing'+str(condition)+ '_' + str(num)+'/'
                print(target_dir)
                
            # for num in idx_list:
                # target_dir = input_path + '/Bearing'+str(condition)+ '_' + str(num)+'/'
                
                target_csv_list = os.listdir(target_dir)
                
                data_len = len(target_csv_list)
                temp_dir = sorted_aphanumeric(target_csv_list)

                fpt = fpt_list[condition-1][num-1]
                fpt_file = int(fpt / length)
                fpt_seq = int(fpt % length)
                
                for index in range(fpt_file,data_len):
                    if index == fpt_file:
                        start_seq = fpt_seq
                    else:
                        start_seq = 0

                    for i in range(start_seq, length, num_sequence):
                        if i <= length - num_sequence:
                            dir_seq = []
                            data_idx_seq = []
                            for seq in range(num_sequence):
                                dir_seq.append(os.path.join(target_dir,temp_dir[index]))
                                data_idx_seq.append(i+seq)
                            
                            act_rul = ( (data_len * length - (index * length + i) ) / (data_len * length - fpt) )
                            self.label.append(act_rul)
                            self.data_dir.append(dir_seq)
                            self.bearing_label.append((condition-1)*num_bearings + num-1)
                            self.data_idx.append(data_idx_seq)
                
        self.label = np.asarray(self.label)
        self.bearing_label = np.asarray(self.bearing_label)
        self.data_idx = np.asarray(self.data_idx)
        
        self.num_sequence=num_sequence

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        x_sequence = []
        y_sequence = []
        bearing_label_seq = []
        for data_idx in range(self.num_sequence):
            data = np.genfromtxt(self.data_dir[i][data_idx], delimiter=',')
            
            x = np.asarray(data[:,0])
            y = np.asarray(data[:,1])        

            x = x[ (self.data_idx[i][data_idx]*self.input_dimension+1) : ((self.data_idx[i][data_idx]+1)*self.input_dimension+1)]
            y = y[ (self.data_idx[i][data_idx]*self.input_dimension+1) : ((self.data_idx[i][data_idx]+1)*self.input_dimension+1)]

            x = x.reshape(1,-1)
            y = y.reshape(1,-1)
            x_sequence.append(x)
            y_sequence.append(y)
            bearing_label_seq.append(self.bearing_label[i])
        
        x_sequence = np.array(x_sequence)
        y_sequence = np.array(y_sequence)
        bearing_label_seq = np.array(bearing_label_seq)

        x_sequence = torch.from_numpy(x_sequence).float()
        y_sequence = torch.from_numpy(y_sequence).float()
        bearing_label_seq = torch.from_numpy(bearing_label_seq).float()

        return x_sequence, y_sequence, self.label[i], bearing_label_seq

class Dataload_Step2_FPT(Dataset):
    def __init__(self, input_path,transform, target_condition, num_bearings, idx_list, fpt_list, num_sequence):
        self.data_dir = []
        self.label = []
        # self.bearing_label = []

        # condition_list = [1, 2]
        condition_list = [target_condition]
        
        for condition in condition_list:
            for num in range(1,num_bearings+1):
                if (condition == target_condition and num not in idx_list) or (len(idx_list)==1 and condition != target_condition):
                    continue

            # for num in idx_list:
                target_dir = input_path + str(condition)+ '_' + str(num) + '/'
                # target_dir = input_path + str(num) + '/'

                print(target_dir)

        # for num in idx_list:
            # target_dir = input_path + str(num) + '/'
            
                target_image_list = os.listdir(target_dir)
                data_len = len(target_image_list)

                sorted_image_list = sorted_aphanumeric(target_image_list)
                
                fpt = fpt_list[condition-1][num-1]
                
                for index in range(fpt,data_len,num_sequence):
                    if index < data_len - num_sequence:
                        dir_seq = []
                        for seq in range(num_sequence):
                            dir_seq.append(os.path.join(target_dir,sorted_image_list[index+seq]))
                        
                        self.label.append((data_len-index)/(data_len-fpt))
                        self.data_dir.append(dir_seq)
                        # self.bearing_label.append(num-1)
            
        self.transform = transform
        self.label = np.asarray(self.label)
        # self.bearing_label = np.asarray(self.bearing_label)
        self.num_sequence=num_sequence
        
    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        input_images = []
        # bearing_label_seq = []
        for data_idx in range(self.num_sequence):
            input_image = cv2.imread(self.data_dir[i][data_idx])
            input_image = self.transform(input_image)
            input_image = input_image.numpy()
            input_images.append(input_image)
            # bearing_label_seq.append(self.bearing_label[i])

        input_images = np.array(input_images)
        # bearing_label_seq = np.array(bearing_label_seq)
        
        
        input_images = torch.from_numpy(input_images).float()
        # input_images = torch.FloatTensor(input_images)
        # bearing_label_seq = torch.from_numpy(bearing_label_seq).float()

        return input_images, self.label[i]


class RUL_CNN_DATALOAD(Dataset):
    def __init__(self, input_path, transform, idx_list, kurtosis_fpt, num_feature_map):
        self.data_dir = []
        self.label = []
        for num in idx_list:
            target_dir = input_path + str(num) + '/'
            print(target_dir)
            target_image_list = os.listdir(target_dir)
            data_len = len(target_image_list)

            temp_dir = sorted_aphanumeric(target_image_list)

            fpt = kurtosis_fpt[num-1]

            for index in range(fpt,data_len):
                self.label.append((data_len-index)/(data_len-fpt))
                self.data_dir.append(os.path.join(target_dir,temp_dir[index]))
                
        print (len(self.data_dir))
        self.label = np.asarray(self.label)

        self.transform = transform
        self.num_feature_map = num_feature_map

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, i):
        feature_image = []
        foldername= os.path.dirname(self.data_dir[i])
        filename = os.path.basename(self.data_dir[i])
        index = int(os.path.splitext(filename)[0])

        nextfilename = foldername + '/' + '%04d'%(index+self.num_feature_map-1) + os.path.splitext(filename)[1]
        

        #print(foldername, filename, index, nextfilename, str(os.path.exists(nextfilename)))

        if i < (len(self.data_dir) - self.num_feature_map) and os.path.exists(nextfilename):
            for img_cnt in range(i, i+self.num_feature_map):
                input_image = cv2.imread(self.data_dir[img_cnt])
                input_image = self.transform(input_image)
                feature_image.append(input_image)
            #feature_label = self.label[i+2]
        else:
            for img_cnt in range(i-self.num_feature_map+1, i+1):
                input_image = cv2.imread(self.data_dir[img_cnt])
                input_image = self.transform(input_image)
                feature_image.append(input_image)
            #feature_label = self.label[i-2]
        
        feature_label = self.label[i]

        return feature_image, feature_label
