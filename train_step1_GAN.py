import numpy as np
import argparse
import os
import sys

import re
import pandas as pd
import cv2

import torch.nn as nn
from torch import optim
import torch.autograd as autograd

import matplotlib.image as im
import torch
import time
import csv
from unet_model import UNet_L3, UNet_L4, UNet_L5, Classifier_FCs, CNN_HS, Discriminator, DomainDiscriminator

from data_loader import Dataload_Step1, Dataload_Step1_FPT, Dataload_XJTU_Step1
from create_NSP import scatterPlot, toRGB

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
from evaluate import evaluate_result

from cvtorchvision import cvtransforms

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class TrainOps_step1(object):
    def __init__(self, train_bearing, test_bearing, num_bearings, condition, device, epochs, batch_size, lr, discriminator_lr, lambda_unet, lambda_domain, n_channels, dropout, resize_width, resize_height, dataset):
        self.train_bearing = train_bearing
        self.test_bearing = test_bearing
        self.num_bearings = num_bearings
        self.condition = condition
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.discriminator_lr = discriminator_lr
        self.lambda_unet = lambda_unet
        self.lambda_domain = lambda_domain
        self.n_channels = n_channels
        self.dropout = dropout
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.dataset = dataset

        self.net3_model_save_path = './save_model/GAN_unet_l3_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.net4_model_save_path = './save_model/GAN_unet_l4_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.net5_model_save_path = './save_model/GAN_unet_l5_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'

        self.classfier_fcs_model_save_path = './save_model/GAN_classifier_FCs_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.discriminator_model_save_path = './save_model/GAN_discriminator_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        
        self.cnnhs_model_save_path = './save_model/cnnhs_model_C'+ str(self.condition) + 'B' + str(self.test_bearing) + '_' + str(self.n_channels) + '_dropout_' + str(self.dropout) +'.pth'
        self.input_dimension = 2560
        self.num_data_per_file = 12

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1))).to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = D(interpolates)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] 
        gradients = gradients.view(gradients.size(0), -1) + 1e-16
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_Discriminator(self, discriminator_model, discriminator_optimizer, input_data, recon_input, lambda_gp, criterion_domain, bearing_label):
        #reconstructed input
        d_recon, domain_recon = discriminator_model(recon_input)
        #original input
        d_original, domain_original = discriminator_model(input_data)
        
        gradient_penalty = self.compute_gradient_penalty(discriminator_model, input_data, recon_input)
        discriminator_loss = -torch.mean(d_original) + torch.mean(d_recon) + lambda_gp * gradient_penalty

        domain_d_loss_recon = criterion_domain(domain_recon, bearing_label)
        domain_d_loss_original = criterion_domain(domain_original, bearing_label)
        domain_d_loss = 0.5 * domain_d_loss_recon + 0.5 * domain_d_loss_original

        discriminator_loss = discriminator_loss + self.lambda_domain * domain_d_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        return discriminator_loss.item()    

    def train_Reconstructor(self, unet_optimizer, classifier_model, classifier_optimizer, discriminator_model, criterion, criterion_domain, bearing_label, recon_input, label):
        d_recon, domain_recon = discriminator_model(recon_input)
        generator_loss = -torch.mean(d_recon)
        domain_loss = criterion_domain(domain_recon, bearing_label)

        outputs = classifier_model(recon_input)
        classification_loss = criterion(outputs, label)
        
        label = torch.reshape(label, (-1,1))

        total_loss = generator_loss + classification_loss * self.lambda_unet - domain_loss * self.lambda_domain
    
        unet_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        unet_optimizer.step()
        classifier_optimizer.step()

        return total_loss.item(), generator_loss.item(), classification_loss.item(), domain_loss.item()
    
    def evaluate_Unet(self, unet_model, classifier_model, criterion, input_data, label):
        feature = unet_model(input_data)
        output = classifier_model(feature)
        
        _, preds = torch.max(output, 1)
        
        loss = criterion(output, label)    
        
        corr_sum = torch.sum(preds == label)
        
        return loss.item(), corr_sum.item()

    def train_GAN_merge(self):
        
        print('training unet with 2 channels merged input')
        
        if self.dataset == 'femto':
            target_dir = '/mnt/nas2/data/fault_diagnosis/FEMTO/condition'+str(self.condition)
            train_dataset = Dataload_Step1(target_dir, self.condition, self.train_bearing, 0.05)
            train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
            test_dataset = Dataload_Step1(target_dir, self.condition, [self.test_bearing], 0.05)
            test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        else:
            # target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY/condition'+str(self.condition)
            target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY'
            train_dataset = Dataload_XJTU_Step1(target_dir, self.condition, self.num_bearings, self.train_bearing, 2560, self.num_data_per_file, 0.05)
            train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
            test_dataset = Dataload_XJTU_Step1(target_dir, self.condition, self.num_bearings, [self.test_bearing], 2560, self.num_data_per_file, 0.05)
            test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        print(self.train_bearing, self.test_bearing)
        
        net_3 = UNet_L3(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_4 = UNet_L4(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_5 = UNet_L5(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)

        self.domain_num = 2*self.num_bearings #len(self.train_bearing) + len([self.test_bearing])
        classifier_fcs = Classifier_FCs(n_channels=self.n_channels, n_classes=2, input_dimension= 2560)
        discriminator = DomainDiscriminator(in_channels=2, input_dimension=self.input_dimension, n_domain=self.domain_num)
        
        net_3.to(self.device,dtype=torch.float64)
        net_4.to(self.device,dtype=torch.float64)
        net_5.to(self.device,dtype=torch.float64)
        classifier_fcs.to(self.device,dtype=torch.float64)
        discriminator.to(self.device,dtype=torch.float64)
        
        # def weight_init(m):
        #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #         nn.init.zeros_(m.bias)

        # net_3.apply(weight_init)
        # net_4.apply(weight_init)
        # net_5.apply(weight_init)


        optimizer_L3 = torch.optim.Adam(net_3.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_L4 = torch.optim.Adam(net_4.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_L5 = torch.optim.Adam(net_5.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_Classifier = torch.optim.Adam(classifier_fcs.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_Discriminator = torch.optim.Adam(discriminator.parameters(), lr = self.discriminator_lr, betas=(0.5, 0.99), weight_decay=1e-5)

        cuda = True if torch.cuda.is_available() else False
        
        criterion = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()

        if cuda:
            criterion.cuda()
            criterion_domain.cuda()
        
        lambda_gp = 10.0

        best_epoch = 0
        best_valid_loss = 9999
        start_time = time.time()

        print('length of train_loader', len(train_loader))
        for epoch in range(self.epochs):
            epoch_loss_L3 = 0
            epoch_loss_L4 = 0
            epoch_loss_L5 = 0
            epoch_loss_D_L3 = 0
            epoch_loss_D_L4 = 0
            epoch_loss_D_L5 = 0
            epoch_loss_C_L3 = 0
            epoch_loss_C_L4 = 0
            epoch_loss_C_L5 = 0
            
            net_3.train()
            net_4.train()
            net_5.train()
            classifier_fcs.train()
            discriminator.train()
            net_3.requires_grad_(True)
            net_4.requires_grad_(True)
            net_5.requires_grad_(True)        
            classifier_fcs.requires_grad_(True)      
            discriminator.requires_grad_(True)
            
            step = 0
            for x,y,label, bearing_label in train_loader:
                step+=1
                input_signal = torch.cat((x,y), dim=1)
                input_signal = input_signal.to(device=self.device, dtype=torch.float64)
                label = label.to(device=self.device)
                bearing_label = bearing_label.to(device=self.device)
                
                #Training Discriminator
                net_3.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L3 = net_3(input_signal)
                
                loss_D_L3 = self.train_Discriminator(discriminator, optimizer_Discriminator, input_signal, recon_L3, lambda_gp, criterion_domain, bearing_label)
                epoch_loss_D_L3 = epoch_loss_D_L3 + loss_D_L3
                discriminator.requires_grad_(False)
                
                net_4.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L4 = net_4(input_signal)
                loss_D_L4 = self.train_Discriminator(discriminator, optimizer_Discriminator, input_signal, recon_L4, lambda_gp, criterion_domain, bearing_label)
                epoch_loss_D_L4 = epoch_loss_D_L4 + loss_D_L4
                discriminator.requires_grad_(False)

                net_5.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L5 = net_5(input_signal)
                loss_D_L5 = self.train_Discriminator(discriminator, optimizer_Discriminator, input_signal, recon_L5, lambda_gp, criterion_domain, bearing_label)
                epoch_loss_D_L5 = epoch_loss_D_L5 + loss_D_L5
                discriminator.requires_grad_(False)

                #Training Reconstructor
                if step % 5 == 0:

                    # Reconstructor L3
                    loss_R_L3, g_loss_L3, c_loss_L3, domain_loss_L3 = self.train_Reconstructor(optimizer_L3, classifier_fcs, optimizer_Classifier, discriminator, criterion, criterion_domain, bearing_label, recon_L3, label)
                    epoch_loss_L3 = epoch_loss_L3 + loss_R_L3
                    epoch_loss_C_L3 = epoch_loss_C_L3 + c_loss_L3
                    net_3.requires_grad_(False)

                    # Reconstructor L4
                    loss_R_L4, g_loss_L4, c_loss_L4, domain_loss_L4 = self.train_Reconstructor(optimizer_L4, classifier_fcs, optimizer_Classifier, discriminator, criterion, criterion_domain, bearing_label, recon_L4, label)
                    epoch_loss_L4 = epoch_loss_L4 + loss_R_L4
                    epoch_loss_C_L4 = epoch_loss_C_L4 + c_loss_L4
                    net_4.requires_grad_(False)

                    # Reconstructor L4
                    loss_R_L5, g_loss_L5, c_loss_L5, domain_loss_L5 = self.train_Reconstructor(optimizer_L5, classifier_fcs, optimizer_Classifier, discriminator, criterion, criterion_domain, bearing_label, recon_L5, label)
                    epoch_loss_L5 = epoch_loss_L5 + loss_R_L5
                    epoch_loss_C_L5 = epoch_loss_C_L5 + c_loss_L5
                    net_5.requires_grad_(False)

            epoch_loss_L3 = epoch_loss_L3 / step
            epoch_loss_L4 = epoch_loss_L4 / step
            epoch_loss_L5 = epoch_loss_L5 / step

            epoch_loss_D_L3 = epoch_loss_D_L3 / step
            epoch_loss_D_L4 = epoch_loss_D_L4 / step
            epoch_loss_D_L5 = epoch_loss_D_L5 / step

            epoch_loss_C_L3 /= step
            epoch_loss_C_L4 /= step
            epoch_loss_C_L5 /= step

            net_3.requires_grad_(False)
            net_4.requires_grad_(False)
            net_5.requires_grad_(False)
            classifier_fcs.requires_grad_(False)

            net_3.eval()
            net_4.eval()
            net_5.eval()

            classifier_fcs.eval()

            valid_loss_L3 = 0
            valid_loss_L4 = 0
            valid_loss_L5 = 0
            step_acc_L3 = 0
            step_acc_L4 = 0
            step_acc_L5 = 0

            total_inputs_len = 0
            step = 0
            with torch.no_grad():
                valid_loss = 0
                for x,y,label, bearing_label in test_loader:
                    step+=1
                    x = x.to(device=self.device, dtype=torch.float64)
                    y = y.to(device=self.device, dtype=torch.float64)
                    label = label.to(device=self.device)
                    
                    input_signal = torch.cat((x,y), dim=1)
                    input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                    loss_L3, corr_sum = self.evaluate_Unet(net_3, classifier_fcs, criterion, input_signal, label)
                    valid_loss_L3 = valid_loss_L3 + loss_L3
                    step_acc_L3 = step_acc_L3 + corr_sum

                    loss_L4, corr_sum = self.evaluate_Unet(net_4, classifier_fcs, criterion, input_signal, label)
                    valid_loss_L4 = valid_loss_L4 + loss_L4
                    step_acc_L4 = step_acc_L4 + corr_sum

                    loss_L5, corr_sum = self.evaluate_Unet(net_5, classifier_fcs, criterion, input_signal, label)
                    valid_loss_L5 = valid_loss_L5 + loss_L5
                    step_acc_L5 = step_acc_L5 + corr_sum

                    total_inputs_len += x.size(0)

            valid_loss_L3 /= step
            valid_loss_L4 /= step
            valid_loss_L5 /= step

            step_acc_L3 /= total_inputs_len
            step_acc_L4 /= total_inputs_len
            step_acc_L5 /= total_inputs_len
        
            valid_loss = valid_loss_L3 + valid_loss_L4 + valid_loss_L5 + epoch_loss_C_L3 + epoch_loss_C_L4 + epoch_loss_C_L5
            if valid_loss <= best_valid_loss and epoch > 5:
                best_epoch = epoch
                best_valid_loss = valid_loss

                print('Best Valid Loss={:.4f} epoch_loss_C_L3={:.4f} epoch_loss_C_L4={:.4f} epoch_loss_C_L5={:.4f}'.format(
                    best_valid_loss, epoch_loss_C_L3, epoch_loss_C_L4, epoch_loss_C_L5
                ))
                torch.save(net_3.state_dict(), self.net3_model_save_path)
                torch.save(net_4.state_dict(), self.net4_model_save_path)
                torch.save(net_5.state_dict(), self.net5_model_save_path)
                torch.save(classifier_fcs.state_dict(), self.classfier_fcs_model_save_path)
                torch.save(discriminator.state_dict(), self.discriminator_model_save_path)
                            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} time={:.2f}s L3_train_loss={:.4f} L4_train_loss={:.4f} L5_train_loss={:.4f} L3_D_loss={:.4f} L4_D_loss={:.4f} L5_D_loss={:.4f} L3_test_loss={:.4f} L4_test_loss={:.4f} L5_test_loss={:.4f} L3_test_acc={:.4f} L4_test_acc={:.4f} L5_test_acc={:.4f}'.format(
                                    epoch +1, self.epochs,
                                    elapsed_time,
                                    epoch_loss_L3, epoch_loss_L4, epoch_loss_L5,
                                    epoch_loss_D_L3, epoch_loss_D_L4, epoch_loss_D_L5,
                                    valid_loss_L3, valid_loss_L4, valid_loss_L5,
                                    step_acc_L3, step_acc_L4, step_acc_L5))

        print('Best Epoch {} Best valid loss {:.4f}'.format(best_epoch+1, best_valid_loss))

    def sorted_aphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def make_nsp(self):
        net_3 = UNet_L3(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)
        net_4 = UNet_L4(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)
        net_5 = UNet_L5(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)

        net_3.load_state_dict(torch.load(self.net3_model_save_path))
        net_3.to(self.device,dtype=torch.float64)
        net_3.requires_grad = False
        
        net_4.load_state_dict(torch.load(self.net4_model_save_path))
        net_4.to(self.device,dtype=torch.float64)
        net_4.requires_grad = False
        
        net_5.load_state_dict(torch.load(self.net5_model_save_path))
        net_5.to(self.device,dtype=torch.float64)
        net_5.requires_grad = False
        
        net_3.eval()
        net_4.eval()
        net_5.eval()

        for condition in range(1,3):
            for bearing_idx in range(self.num_bearings):
                if self.dataset == 'femto':
                    target_dir = '/mnt/nas2/data/fault_diagnosis/FEMTO/condition'+str(condition)+ '/Bearing'+str(condition)+ '_' + str(bearing_idx+1)+'/'
                else:
                    target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY/condition'+str(condition)+ '/Bearing'+str(condition)+ '_' + str(bearing_idx+1)+'/'

                if self.dropout:
                    save_dir = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(condition)+ '_' + str(bearing_idx+1)+'/'
                else:
                    save_dir = './MultiScale_NSP_2/Bearing'+str(condition)+ '_' + str(bearing_idx+1)+'/'

                print('Start processing Bearing', bearing_idx+1)

                def _mkdir_recursive(path):
                    sub_path = os.path.dirname(path)
                    if not os.path.exists(sub_path):
                        _mkdir_recursive(sub_path)
                    if not os.path.exists(path):
                        os.mkdir(path)

                if not os.path.exists(save_dir):
                    _mkdir_recursive(save_dir)
                    print("Directory " , save_dir ,  " Created ")

                target_csv_list = os.listdir(target_dir)
                target_csv_list = self.sorted_aphanumeric(target_csv_list)
                print(save_dir)
                cnt=0
                with torch.no_grad():
                    for filename in target_csv_list:
                        if self.dataset == 'femto' and filename[:3]!='acc':
                            continue
                        
                        if self.dataset == 'femto':
                            data = np.genfromtxt(os.path.join(target_dir,filename), delimiter=',')

                            data = data[:,4:]

                            x = np.asarray(data[:,0])
                            y = np.asarray(data[:,1])
                            
                            x = x.reshape(1,1, -1)
                            y = y.reshape(1,1,-1)
                            
                            x = torch.from_numpy(x).float()
                            y = torch.from_numpy(y).float()
                                
                            input_signal = torch.cat((x,y), dim=1)
                            input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                            image = self.make_nsp_image_from_vib_signal(input_signal, net_3, net_4, net_5)
                            save_filename = save_dir + '%04d'%cnt + '.png'
                            # print(save_filename)

                            cv2.imwrite(save_filename, image)
                            cnt+=1
                                
                        else:
                            for data_idx in range(self.num_data_per_file):
                                
                                data = np.genfromtxt(os.path.join(target_dir,filename), delimiter=',')
                                x = np.asarray(data[:,0])
                                y = np.asarray(data[:,1])

                                x = x[ (data_idx*self.input_dimension+1) : ((data_idx+1)*self.input_dimension+1)]
                                y = y[ (data_idx*self.input_dimension+1) : ((data_idx+1)*self.input_dimension+1)]

                                x = x.reshape(1,1,-1)
                                y = y.reshape(1,1,-1)
                                
                                x = torch.from_numpy(x).float()
                                y = torch.from_numpy(y).float()
                                    
                                input_signal = torch.cat((x,y), dim=1)
                                input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                                image = self.make_nsp_image_from_vib_signal(input_signal, net_3, net_4, net_5)
                                save_filename = save_dir + '%04d'%cnt + '.png'
                                # print(save_filename)

                                cv2.imwrite(save_filename, image)
                                cnt+=1

                print('processing Done Bearing', bearing_idx+1)

    def make_nsp_image_from_vib_signal(self, input_signal, net_3, net_4, net_5):
        extractor_net3 = net_3.extractor(input_signal).cpu().numpy()
        extractor_net3 = extractor_net3.reshape(2, -1)
        extractor_net3_x = extractor_net3[0,:]
        extractor_net3_y = extractor_net3[1,:]

        extractor_net4 = net_4.extractor(input_signal).cpu().numpy()
        extractor_net4 = extractor_net4.reshape(2, -1)
        extractor_net4_x = extractor_net4[0,:]
        extractor_net4_y = extractor_net4[1,:]

        extractor_net5 = net_5.extractor(input_signal).cpu().numpy()
        extractor_net5 = extractor_net5.reshape(2, -1)
        extractor_net5_x = extractor_net5[0,:]
        extractor_net5_y = extractor_net5[1,:]

        red, red_count = scatterPlot(extractor_net3_x, extractor_net3_y, -2, 2, 128)
        green,green_count = scatterPlot(extractor_net4_x, extractor_net4_y, -2, 2, 128)
        blue, blue_count = scatterPlot(extractor_net5_x, extractor_net5_y, -2, 2, 128)

        # print(red_count, green_count, blue_count)
        image = toRGB(red, green, blue)
        image = cv2.resize(image,(128,128),interpolation=cv2.INTER_NEAREST)
        return image

    def save_intermediate_imageset(self):
        net_3 = UNet_L3(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_4 = UNet_L4(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_5 = UNet_L5(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        
        net_3.load_state_dict(torch.load(self.net3_model_save_path))
        net_3.to(self.device,dtype=torch.float64)
        net_3.requires_grad = False
        
        net_4.load_state_dict(torch.load(self.net4_model_save_path))
        net_4.to(self.device,dtype=torch.float64)
        net_4.requires_grad = False
        
        net_5.load_state_dict(torch.load(self.net5_model_save_path))
        net_5.to(self.device,dtype=torch.float64)
        net_5.requires_grad = False
        
        net_3.eval()
        net_4.eval()
        net_5.eval()

        himg_list = []

        for condition in range(1,3):
            for bearing_idx in range(self.num_bearings):
                if self.dataset == 'femto':
                    target_dir = '/mnt/nas2/data/fault_diagnosis/FEMTO/condition'+str(condition)+ '/Bearing'+str(condition)+ '_' + str(bearing_idx+1)+'/'
                else:
                    target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY/condition'+str(condition)+ '/Bearing'+str(condition)+ '_' + str(bearing_idx+1)+'/'
                
                print('Start processing Bearing', bearing_idx+1)
                
                print(target_dir)
                target_csv_list = os.listdir(target_dir)
                target_csv_list = self.sorted_aphanumeric(target_csv_list)
                
                cnt=0
                image_list = []

                total_file_cnt=0
                if self.dataset == 'femto':
                    for filename in target_csv_list:
                        if filename[:3]!='acc':
                            continue
                        total_file_cnt += 1
                else:
                    total_file_cnt = len(target_csv_list)*self.num_data_per_file

                interval = int(total_file_cnt/30)
                print(interval)

                with torch.no_grad():
                    for filename in target_csv_list:
                        if self.dataset == 'femto' and filename[:3]!='acc':
                            continue
                        
                        if self.dataset == 'femto':
                            if cnt % interval == 0 and cnt > 0:
                                
                                data = np.genfromtxt(os.path.join(target_dir,filename), delimiter=',')

                                data = data[:,4:]

                                x = np.asarray(data[:,0])
                                y = np.asarray(data[:,1])
                                
                                x = x.reshape(1,1, -1)
                                y = y.reshape(1,1,-1)
                                
                                x = torch.from_numpy(x).float()
                                y = torch.from_numpy(y).float()
                                    
                                input_signal = torch.cat((x,y), dim=1)
                                input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                                image = self.make_nsp_image_from_vib_signal(input_signal, net_3, net_4, net_5)
                                
                                image_list.append(image)
                            cnt+=1
                        else:
                            for data_idx in range(self.num_data_per_file):
                                if cnt % interval == 0 and cnt > 0:
                                    data = np.genfromtxt(os.path.join(target_dir,filename), delimiter=',')
                                    x = np.asarray(data[:,0])
                                    y = np.asarray(data[:,1])

                                    x = x[ (data_idx*self.input_dimension+1) : ((data_idx+1)*self.input_dimension+1)]
                                    y = y[ (data_idx*self.input_dimension+1) : ((data_idx+1)*self.input_dimension+1)]

                                    x = x.reshape(1,1, -1)
                                    y = y.reshape(1,1,-1)
                                    
                                    x = torch.from_numpy(x).float()
                                    y = torch.from_numpy(y).float()
                                        
                                    input_signal = torch.cat((x,y), dim=1)
                                    input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                                    image = self.make_nsp_image_from_vib_signal(input_signal, net_3, net_4, net_5)
                                    
                                    image_list.append(image)
                                cnt+=1

                print('processing Done Bearing', bearing_idx+1)

                h_img = image_list[0]
                image_list = np.array(image_list)
                print(image_list.shape)

                for j in range(1,30):
                    h_img = cv2.hconcat([h_img, image_list[j]])

                himg_list.append(h_img)


        himg_list = np.array(himg_list)
        print(himg_list.shape)
        v_img = himg_list[0]
        for i in range (1,3):
            for n in range(0,self.num_bearings):
                if i==1 and n==0:
                    continue
                v_img = cv2.vconcat([v_img, himg_list[(i-1)*self.num_bearings + n]])

        save_image_path = './NSP_imageset/' + self.dataset + '_Step1_NSP_C'+str(self.condition) + 'B' + str(self.test_bearing) + '_dropout_ ' + str(self.dropout) + '_' + str(self.n_channels) +'.png'
        cv2.imwrite(save_image_path,v_img)


    def train_cnn_hs(self):
        if self.n_channels == 0:
            input_path='./MultiScale_NSP_original/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        elif self.n_channels == 1:
            input_path='./MultiScale_NSP/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        elif self.n_channels == 2:
            if self.dropout:
                input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'
                # input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path = './MultiScale_NSP_2_Dropout/Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path='./MultiScale_NSP_2_Dropout/Bearing' + str(self.condition) + '_'
            else:
                input_path='./MultiScale_NSP_2/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        else:
            input_path='./NSP/' + self.dataset + '/Bearing' 
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor(),
            ])
        
        print(input_path)
        train_dataset = Dataload_Step1_FPT(input_path, transform, self.condition, self.num_bearings, self.train_bearing, 0.05)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
        test_dataset = Dataload_Step1_FPT(input_path, transform, self.condition, self.num_bearings, [self.test_bearing], 0.05)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        model = CNN_HS(input_dimension=(self.resize_height,self.resize_width,3), n_classes=2)
        model.to(device=self.device)

        optimizer_CNNHS = torch.optim.Adam(model.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
            
        criterion = nn.CrossEntropyLoss()
        
        best_epoch = 0
        best_valid_loss = 9999
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = 0

            model.train()
            model.requires_grad_(True)

            step = 0
            for image,label in train_loader:
                step+=1
                
                image = image.to(device=self.device)
                label = label.to(device=self.device)
                
                predict = model(image)

                loss = criterion(predict, label)
                
                optimizer_CNNHS.zero_grad()
                loss.backward()
                optimizer_CNNHS.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / step

            model.requires_grad_(False)
            model.eval()

            valid_loss = 0
            step_acc = 0

            total_inputs_len = 0
            step = 0
            with torch.no_grad():
                valid_loss = 0
                for image,label in test_loader:
                    step+=1
                    image = image.to(device=self.device)
                    label = label.to(device=self.device)
                    
                    output = model(image)
                    
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, label)

                    corr_sum = torch.sum(preds == label).item()
                    total_inputs_len += image.size(0)

                    valid_loss = valid_loss + loss.item()
                    step_acc = step_acc + corr_sum
                    
            valid_loss /= step

            step_acc /= total_inputs_len

            if valid_loss < best_valid_loss and epoch > 5:
                best_epoch = epoch
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.cnnhs_model_save_path)
                            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} time={:.2f}s train_loss={:.4f} test_loss={:.4f} test_acc={:.4f}'.format(
                                    epoch +1, self.epochs,
                                    elapsed_time,
                                    epoch_loss, valid_loss, step_acc))

    
    def get_fpt(self):
        
        if self.n_channels == 0:
            input_path='./MultiScale_NSP_original/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        elif self.n_channels == 1:
            input_path='./MultiScale_NSP/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        elif self.n_channels == 2:
            if self.dropout:
                input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'
                # input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path = './MultiScale_NSP_2_Dropout/Step1/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path='./MultiScale_NSP_2_Dropout/Bearing' + str(self.condition) + '_'
            else:
                input_path='./MultiScale_NSP_2/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        else:
            input_path='./NSP/' + self.dataset + '/Bearing' 
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor(),
            ])
        
        # test_dataset = Dataload_Step1_FPT(input_path, transform, [self.test_bearing], 0.5)
        # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8, pin_memory=True)

        model = CNN_HS(input_dimension=(self.resize_height,self.resize_width,3), n_classes=2)
        model.load_state_dict(torch.load(self.cnnhs_model_save_path))
        model.to(device=self.device)
        model.requires_grad = False
        model.eval()
        
        FPT_list = [] 
        total_length = []   
        condition_list = [1, 2]
        for condition in condition_list:
            FPT_condition_list = []
            condition_total_length = []
            for bearing_idx in range(1, self.num_bearings+1):
                test_dataset = Dataload_Step1_FPT(input_path, transform, condition, self.num_bearings, [bearing_idx], 0.5)
                test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8, pin_memory=True)
                
                results = []
                for image,label in test_loader:
                    image = image.to(device=self.device)
                    
                    output = model(image)
                    _, preds = torch.max(output, 1)

                    predict = preds.detach().cpu().numpy()
                    predict = predict.reshape(-1)
                    results.append(predict)

                fpt = self.FIND_FPT(results)
                print('Condition: {} Bearing: {} FPT: {}/{}'.format(condition, bearing_idx, fpt, (len(test_loader))))
                FPT_condition_list.append(fpt)
                condition_total_length.append(len(results))

                plt.cla()
                plt.clf()
                plt.plot(range(len(results)),results, color='y', label='Predict')
                plt.legend()
                plot_file_name = './plot/'+ self.dataset + '_FPTPredcition_C'+ str(condition)+'B'+str(bearing_idx)+ '_' + str(self.n_channels) + '_dropout_' + str(self.dropout) +'.png'
                plt.savefig(plot_file_name)
            FPT_list.append(FPT_condition_list)
            total_length.append(condition_total_length)

        testresultname = './testresult/' + 'FPT_C' + str(self.condition) + 'B'+str(self.test_bearing)+ '_' + str(self.n_channels) + '_dropout_' + str(self.dropout) + '.csv'
        f=open(testresultname, 'w',
                encoding='utf-8', newline='')
        wr = csv.writer(f)
        for condition in condition_list:
            for idx in range(self.num_bearings):
                wr.writerow([total_length[condition-1][idx], FPT_list[condition-1][idx]])
        f.close()
        
        print(total_length, FPT_list)
        return FPT_list

    def FIND_FPT(self, results):
        ''' 
        Find first predicting time using CNN-HS model results 
        args:
            results = Healthy or Unhealthy results obtained through CNN-HS
        return:
            fpt = The fastest spot observed unhealthy 3 times in results
        '''
        count=0
        fpt = len(results)
        for i in range(len(results)):
            for j in range(5):
                if(i+j == len(results)-1):
                        print("\n\ncould not find FPT\n\n")
                        fpt = len(results)
                        return fpt
                if (results[i+j]==1):
                    count+=1
                    if(count==5):
                        fpt = i
                        return fpt
            count = 0   
        return fpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="GPU number")
    parser.add_argument("--condition", type=int, default=1, help="condition")
    parser.add_argument("--num_bearings", type=int, default=7, help="the number of bearings")
    parser.add_argument("--test_bearing", type=int, default=7, help="test_bearing")
    parser.add_argument("--epoch", type=int, default=40, help="epoch")
    parser.add_argument("--n_channels", type=int, default=2, help="n_channels, '0:original','1:1channel','2:2channels','3:manualNSP'")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--discriminator_lr", type=float, default=1e-4, help="learning_rate of discriminator")
    parser.add_argument("--lambda_unet", type=float, default=20.0, help="lambda for loss function")
    parser.add_argument("--lambda_domain", type=float, default=1.0, help="lambda3 for loss function")
    parser.add_argument("--mode", type=str, default='train_unet', help=" 'train_unet', 'make_nsp', 'train_cnn_hs', 'get_fpt'")
    parser.add_argument("--resize_width", type=int, default=128)
    parser.add_argument("--resize_height", type=int, default=128)
    parser.add_argument("--dropout", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='femto', help=" 'femto', 'xjtu'")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device : ")
    print(device)
    # device = torch.device("cuda:%s" % args.gpu)

    if args.dataset == 'femto':
        train = [1,2,3,4,5,6,7]
    else:
        train = [1,2,3,4,5]
    train.remove(args.test_bearing)

    print('condition:', args.condition)
    print('test_bearing:', args.test_bearing)
    args.dropout = bool(args.dropout)
    print (args.dropout)

    train = TrainOps_step1(train_bearing=train, test_bearing=args.test_bearing, num_bearings=args.num_bearings, condition=args.condition, device=device, epochs=args.epoch, batch_size=args.batch_size, 
                            lr=args.lr, discriminator_lr=args.discriminator_lr, 
                            lambda_unet=args.lambda_unet, lambda_domain=args.lambda_domain, n_channels=args.n_channels, dropout=args.dropout, resize_width=args.resize_width, resize_height=args.resize_height, dataset=args.dataset)

    if args.mode == 'train_GAN':
        train.train_GAN_merge()

    elif args.mode == 'make_nsp':
        train.save_intermediate_imageset()
        train.make_nsp()
        train.train_cnn_hs()
        train.get_fpt()

    elif args.mode == 'train_cnn_hs':
        train.train_cnn_hs()
        train.get_fpt()
    
    elif args.mode == 'get_fpt':
        train.get_fpt()

    elif args.mode == 'intermediate':
        train.save_intermediate_imageset()
    
    elif args.mode == 'all':
        print('train GAN merge')
        train.train_GAN_merge()
        print('save intermediate imageset')
        train.save_intermediate_imageset()
        print('make nsp')
        train.make_nsp()
        print('train cnn hs')
        train.train_cnn_hs()
        print('get fpt')
        train.get_fpt()