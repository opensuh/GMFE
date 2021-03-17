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
import torch
import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/RULUNet_' + str(int(time.time())))

import matplotlib.image as im
import csv
from unet_model import UNet_L3, UNet_L4, UNet_L5, Classifier_LSTM, CNN_HS, DomainDiscriminator, Classifier_NSP_LSTM

from data_loader import Dataload_Step2, Dataload_XJTU_Step2, Dataload_Step1_FPT, Dataload_Step2_FPT
from create_NSP import scatterPlot, toRGB

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
from evaluate import evaluate_result

from cvtorchvision import cvtransforms

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class TrainOps_step2(object):

    def __init__(self, train_bearing, test_bearing, num_bearings, condition, device, epochs, batch_size, sequence_length, lr, discriminator_lr, lambda_1, lambda_2, lambda_3, lambda_domain, n_channels, dropout, resize_width, resize_height, dataset, FPT):
        self.train_bearing = train_bearing
        self.test_bearing = test_bearing
        self.num_bearings = num_bearings
        self.condition = condition
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length= sequence_length
        self.lr = lr
        self.discriminator_lr = discriminator_lr
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_domain = lambda_domain
        self.n_channels = n_channels
        self.dropout = dropout
        self.resize_width = resize_width
        self.resize_height = resize_height

        self.dataset = dataset
        self.FPT = FPT

        self.net3_model_path = './save_model/GAN_unet_l3_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.net4_model_path = './save_model/GAN_unet_l4_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.net5_model_path = './save_model/GAN_unet_l5_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'
        self.cnnhs_model_save_path = './save_model/cnnhs_model_C'+ str(self.condition) + 'B' + str(self.test_bearing) + '_' + str(self.n_channels) + '_dropout_' + str(self.dropout) +'.pth'
        self.discriminator_model_save_path = './save_model/GAN_discriminator_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '.pth'

        self.net3_RUL_model_save_path = './save_model/'+self.dataset+'_RUL_unet_l3_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.FPT) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '.pth' 
        self.net4_RUL_model_save_path = './save_model/'+self.dataset+'_RUL_unet_l4_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.FPT) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '.pth' 
        self.net5_RUL_model_save_path = './save_model/'+self.dataset+'_RUL_unet_l5_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.FPT) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '.pth' 
        self.LSTM_model_save_path = './save_model/'+self.dataset+'_RUL_classifier_LSTM_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.FPT) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '.pth'
        self.discriminator_RUL_model_save_path = './save_model/'+self.dataset+'_RUL_discriminator_C'+str(self.condition)+ 'B'+ str(self.test_bearing) + '_dropout_' + str(self.dropout) + '_' + str(self.FPT) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '.pth'
        
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
        # domain_d_loss_original = criterion_domain(domain_original, bearing_label)
        # domain_d_loss = 0.5 * domain_d_loss_recon + 0.5 * domain_d_loss_original
        domain_d_loss = domain_d_loss_recon

        discriminator_loss = discriminator_loss + self.lambda_domain * domain_d_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        return discriminator_loss.item()

    def train_Reconstructor_RUL(self, unet_optimizer, classifier_model, classifier_optimizer, discriminator_model, criterion_MAE, criterion_MSE, criterion_domain, bearing_label, recon_input, label):
        d_recon, domain_recon = discriminator_model(recon_input)
        generator_loss = -torch.mean(d_recon)
        domain_loss = criterion_domain(domain_recon, bearing_label)

        recon_input_sequence = torch.reshape(recon_input, (-1, self.sequence_length, 2, self.input_dimension))
        # print('recon_input_sequence_size:',recon_input_sequence.size())
        outputs = classifier_model(recon_input_sequence)                
        label = torch.reshape(label, (-1,1))

        classification_loss_MAE = criterion_MAE(outputs, label)
        classification_loss_MSE = torch.sqrt(criterion_MSE(outputs, label))
        
        classification_loss_MAPE = torch.mean(torch.abs((label - outputs) / label))    

        total_loss = generator_loss + classification_loss_MAE * self.lambda_1 + classification_loss_MSE * self.lambda_2 + classification_loss_MAPE * self.lambda_3 - domain_loss * self.lambda_domain
        
        unet_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        total_loss.backward()
        unet_optimizer.step()
        classifier_optimizer.step()

        return total_loss.item(), generator_loss.item(), classification_loss_MAE.item(), classification_loss_MSE.item(), classification_loss_MAPE.item(), domain_loss.item()

    def evaluate_Unet_RUL(self, unet_model, classifier_model, criterion_MAE, criterion_MSE, input_data, label):
        total_input_data = torch.reshape(input_data, (-1, input_data.size()[2], input_data.size()[3] ))         # (BATCHSIZE x SEQUENCE LENGTH, 2, input dimension)
        
        feature = unet_model(total_input_data)
        feature_sequence = torch.reshape(feature, (-1, self.sequence_length, 2, self.input_dimension))
        output = classifier_model(feature_sequence)
        
        label = torch.reshape(label, (-1,1))

        MAE_loss = criterion_MAE(output, label)
        MSE_loss = torch.sqrt(criterion_MSE(output, label))
        MAPE_loss = torch.mean(torch.abs((label - output) / label))    
        
        return output, MAE_loss.item(), MSE_loss.item(), MAPE_loss.item()

    def train_GAN_LSTM(self, FPT_list):
        
        print('training unet with 2 channels merged input')

        if self.dataset == 'femto':
            target_dir = '/mnt/nas2/data/fault_diagnosis/FEMTO/condition'+str(self.condition)
            # target_dir = '/home/opensuh/data/fault_diagnosis/FEMTO/condition'+str(self.condition)
            train_dataset = Dataload_Step2(target_dir, self.condition, self.train_bearing, FPT_list, self.sequence_length)
            train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
            test_dataset = Dataload_Step2(target_dir, self.condition, [self.test_bearing], FPT_list, self.sequence_length)
            test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        else:
            target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY'
            # target_dir = '/mnt/nas2/data/fault_diagnosis/XJTU_SY/condition'+str(self.condition)
            train_dataset = Dataload_XJTU_Step2(target_dir, self.condition, self.num_bearings, self.train_bearing, FPT_list, self.sequence_length, 2560, self.num_data_per_file)
            train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
            test_dataset = Dataload_XJTU_Step2(target_dir, self.condition, self.num_bearings, [self.test_bearing], FPT_list, self.sequence_length, 2560, self.num_data_per_file)
            test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        print(self.train_bearing, self.test_bearing)
        print(len(train_dataset), len(test_dataset))
        
        # torch.cuda.empty_cache()
        net_3 = UNet_L3(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_4 = UNet_L4(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)
        net_5 = UNet_L5(n_channels=self.n_channels, n_classes=2, input_dimension= self.input_dimension, dropout=self.dropout)

        # net_3.load_state_dict(torch.load(self.net3_model_path))
        net_3.to(self.device,dtype=torch.float64)
        
        # net_4.load_state_dict(torch.load(self.net4_model_path))
        net_4.to(self.device,dtype=torch.float64)
            
        # net_5.load_state_dict(torch.load(self.net5_model_path))
        net_5.to(self.device,dtype=torch.float64)
        
        self.domain_num = 2*self.num_bearings #len(self.train_bearing) + len([self.test_bearing])
        classifier_LSTM = Classifier_LSTM(n_channels=2, input_dimension= self.input_dimension, sequence_len=self.sequence_length)
        discriminator = DomainDiscriminator(in_channels=2, input_dimension=self.input_dimension, n_domain=self.domain_num)

        # discriminator.load_state_dict(torch.load(self.discriminator_model_save_path))

        net_3.to(self.device,dtype=torch.float64)
        net_4.to(self.device,dtype=torch.float64)
        net_5.to(self.device,dtype=torch.float64)
        classifier_LSTM.to(self.device,dtype=torch.float64)
        discriminator.to(self.device,dtype=torch.float64)

        
        optimizer_L3 = torch.optim.Adam(net_3.parameters(), lr = self.lr, betas=(0.5, 0.99))#, weight_decay=1e-5)
        optimizer_L4 = torch.optim.Adam(net_4.parameters(), lr = self.lr, betas=(0.5, 0.99))
        optimizer_L5 = torch.optim.Adam(net_5.parameters(), lr = self.lr, betas=(0.5, 0.99))
        optimizer_Classifier = torch.optim.Adam(classifier_LSTM.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_Discriminator = torch.optim.Adam(discriminator.parameters(), lr = self.discriminator_lr, betas=(0.5, 0.99))

        cuda = True if torch.cuda.is_available() else False
        
        criterion_MSE = nn.MSELoss()
        criterion_MAE = nn.L1Loss()
        criterion_domain = nn.CrossEntropyLoss()

        if cuda:
            criterion_MSE.cuda()
            criterion_MAE.cuda()
            criterion_domain.cuda()
        
        lambda_gp = 10.0

        best_epoch = 0
        best_valid_loss = 9999
        start_time = time.time()
        
        print('length of train_loader', len(train_loader))
        total_step = 0
        for epoch in range(self.epochs):
            epoch_loss_L3 = 0
            epoch_loss_L4 = 0
            epoch_loss_L5 = 0
            epoch_loss_D_L3 = 0
            epoch_loss_D_L4 = 0
            epoch_loss_D_L5 = 0
            
            net_3.train()
            net_4.train()
            net_5.train()
            classifier_LSTM.train()
            discriminator.train()
            net_3.requires_grad_(True)
            net_4.requires_grad_(True)
            net_5.requires_grad_(True)        
            classifier_LSTM.requires_grad_(True)      
            discriminator.requires_grad_(True)

            d_loss_L3_per_batch = []
            d_loss_L4_per_batch = []
            d_loss_L5_per_batch = []
            recon_loss_L3_per_batch = []
            recon_loss_L4_per_batch = []
            recon_loss_L5_per_batch = []
            g_loss_L3_per_batch = []
            g_loss_L4_per_batch = []
            g_loss_L5_per_batch = []
            MAE_loss_L3_per_batch = []
            MAE_loss_L4_per_batch = []
            MAE_loss_L5_per_batch = []
            MSE_loss_L3_per_batch = []
            MSE_loss_L4_per_batch = []
            MSE_loss_L5_per_batch = []
            MAPE_loss_L3_per_batch = []
            MAPE_loss_L4_per_batch = []
            MAPE_loss_L5_per_batch = []
            domain_loss_L3_per_batch = []
            domain_loss_L4_per_batch = []
            domain_loss_L5_per_batch = []

            step = 0
            
            for x,y,label, bearing_label in train_loader:
                step+=1
                total_step+=1
                input_signal = torch.cat((x,y), dim=2)
                input_signal = input_signal.to(device=self.device, dtype=torch.float64)
                label = label.to(device=self.device, dtype=torch.float64)
                bearing_label = bearing_label.to(device=self.device, dtype=torch.int64)            
                
                #Training Discriminator
                total_input_signal = torch.reshape(input_signal, (-1, input_signal.size()[2], input_signal.size()[3] ))         # (BATCHSIZE x SEQUENCE LENGTH, 2, input dimension)
                total_bearing_label = torch.reshape(bearing_label, (-1,1))
                total_bearing_label = total_bearing_label.view(-1)
                
                net_3.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L3 = net_3(total_input_signal)
                loss_D_L3 = self.train_Discriminator(discriminator, optimizer_Discriminator, total_input_signal, recon_L3, lambda_gp, criterion_domain, total_bearing_label)
                epoch_loss_D_L3 = epoch_loss_D_L3 + loss_D_L3
                discriminator.requires_grad_(False)
                d_loss_L3_per_batch.append(loss_D_L3)
                
                net_4.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L4 = net_4(total_input_signal)
                loss_D_L4 = self.train_Discriminator(discriminator, optimizer_Discriminator, total_input_signal, recon_L4, lambda_gp, criterion_domain, total_bearing_label)
                epoch_loss_D_L4 = epoch_loss_D_L4 + loss_D_L4
                discriminator.requires_grad_(False)
                d_loss_L4_per_batch.append(loss_D_L4)

                net_5.requires_grad_(True)
                discriminator.requires_grad_(True)
                recon_L5 = net_5(total_input_signal)
                loss_D_L5 = self.train_Discriminator(discriminator, optimizer_Discriminator, total_input_signal, recon_L5, lambda_gp, criterion_domain, total_bearing_label)
                epoch_loss_D_L5 = epoch_loss_D_L5 + loss_D_L5
                discriminator.requires_grad_(False)
                d_loss_L5_per_batch.append(loss_D_L5)
                
                #Training Reconstructor
                if total_step % 5 == 0:
                    classifier_LSTM.requires_grad_(True)
                    # Reconstructor L3
                    loss_R_L3, g_loss_L3, MAE_loss_L3, MSE_loss_L3, MAPE_loss_L3, domain_loss_L3 = self.train_Reconstructor_RUL(optimizer_L3, classifier_LSTM, optimizer_Classifier, discriminator, criterion_MAE, criterion_MSE, criterion_domain, total_bearing_label, recon_L3, label)
                    epoch_loss_L3 = epoch_loss_L3 + loss_R_L3
                    net_3.requires_grad_(False)
                    recon_loss_L3_per_batch.append(loss_R_L3)
                    g_loss_L3_per_batch.append(g_loss_L3)
                    MAE_loss_L3_per_batch.append(MAE_loss_L3)
                    MSE_loss_L3_per_batch.append(MSE_loss_L3)
                    MAPE_loss_L3_per_batch.append(MAPE_loss_L3)
                    domain_loss_L3_per_batch.append(domain_loss_L3)

                    # Reconstructor L4
                    loss_R_L4, g_loss_L4, MAE_loss_L4, MSE_loss_L4, MAPE_loss_L4, domain_loss_L4 = self.train_Reconstructor_RUL(optimizer_L4, classifier_LSTM, optimizer_Classifier, discriminator, criterion_MAE, criterion_MSE, criterion_domain, total_bearing_label, recon_L4, label)
                    epoch_loss_L4 = epoch_loss_L4 + loss_R_L4
                    net_4.requires_grad_(False)
                    recon_loss_L4_per_batch.append(loss_R_L4)
                    g_loss_L4_per_batch.append(g_loss_L4)
                    MAE_loss_L4_per_batch.append(MAE_loss_L4)
                    MSE_loss_L4_per_batch.append(MSE_loss_L4)
                    MAPE_loss_L4_per_batch.append(MAPE_loss_L4)
                    domain_loss_L4_per_batch.append(domain_loss_L4)

                    # Reconstructor L4
                    loss_R_L5, g_loss_L5, MAE_loss_L5, MSE_loss_L5, MAPE_loss_L5, domain_loss_L5 = self.train_Reconstructor_RUL(optimizer_L5, classifier_LSTM, optimizer_Classifier, discriminator, criterion_MAE, criterion_MSE, criterion_domain, total_bearing_label, recon_L5, label)
                    epoch_loss_L5 = epoch_loss_L5 + loss_R_L5
                    net_5.requires_grad_(False)
                    recon_loss_L5_per_batch.append(loss_R_L5)
                    g_loss_L5_per_batch.append(g_loss_L5)
                    MAE_loss_L5_per_batch.append(MAE_loss_L5)
                    MSE_loss_L5_per_batch.append(MSE_loss_L5)
                    MAPE_loss_L5_per_batch.append(MAPE_loss_L5)
                    domain_loss_L5_per_batch.append(domain_loss_L5)

                    classifier_LSTM.requires_grad_(False)
                
                # if step % ( int(len(train_loader)/10) ) == 0 and step != 0:
                if step % 10 == 0 and step != 0:    
                    writer.add_scalar('train_loss/D_loss_L3', np.mean(d_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/D_loss_L4', np.mean(d_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/D_loss_L5', np.mean(d_loss_L5_per_batch), total_step)

                    writer.add_scalar('train_loss/Recon_loss_L3', np.mean(recon_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/Recon_loss_L4', np.mean(recon_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/Recon_loss_L5', np.mean(recon_loss_L5_per_batch), total_step)
                    writer.add_scalar('train_loss/G_loss_L3', np.mean(g_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/G_loss_L4', np.mean(g_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/G_loss_L5', np.mean(g_loss_L5_per_batch), total_step)
                    writer.add_scalar('train_loss/MAE_loss_L3', np.mean(MAE_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/MAE_loss_L4', np.mean(MAE_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/MAE_loss_L5', np.mean(MAE_loss_L5_per_batch), total_step)
                    writer.add_scalar('train_loss/MSE_loss_L3', np.mean(MSE_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/MSE_loss_L4', np.mean(MSE_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/MSE_loss_L5', np.mean(MSE_loss_L5_per_batch), total_step)
                    writer.add_scalar('train_loss/MAPE_loss_L3', np.mean(MAPE_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/MAPE_loss_L4', np.mean(MAPE_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/MAPE_loss_L5', np.mean(MAPE_loss_L5_per_batch), total_step)
                    writer.add_scalar('train_loss/domain_loss_L3', np.mean(domain_loss_L3_per_batch), total_step)
                    writer.add_scalar('train_loss/domain_loss_L4', np.mean(domain_loss_L4_per_batch), total_step)
                    writer.add_scalar('train_loss/domain_loss_L5', np.mean(domain_loss_L5_per_batch), total_step)
                
                # print('Epoch {} SaveStep {} Step {} Time {}'.format(epoch+1, save_step, step, time.time() - start_time))

            epoch_loss_L3 = epoch_loss_L3 / step
            epoch_loss_L4 = epoch_loss_L4 / step
            epoch_loss_L5 = epoch_loss_L5 / step

            epoch_loss_D_L3 = epoch_loss_D_L3 / step
            epoch_loss_D_L4 = epoch_loss_D_L4 / step
            epoch_loss_D_L5 = epoch_loss_D_L5 / step

            net_3.requires_grad_(False)
            net_4.requires_grad_(False)
            net_5.requires_grad_(False)
            classifier_LSTM.requires_grad_(False)

            writer.add_scalar('train_loss/D_loss_L3', np.mean(d_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/D_loss_L4', np.mean(d_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/D_loss_L5', np.mean(d_loss_L5_per_batch), total_step)

            writer.add_scalar('train_loss/Recon_loss_L3', np.mean(recon_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/Recon_loss_L4', np.mean(recon_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/Recon_loss_L5', np.mean(recon_loss_L5_per_batch), total_step)
            writer.add_scalar('train_loss/G_loss_L3', np.mean(g_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/G_loss_L4', np.mean(g_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/G_loss_L5', np.mean(g_loss_L5_per_batch), total_step)
            writer.add_scalar('train_loss/MAE_loss_L3', np.mean(MAE_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/MAE_loss_L4', np.mean(MAE_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/MAE_loss_L5', np.mean(MAE_loss_L5_per_batch), total_step)
            writer.add_scalar('train_loss/MSE_loss_L3', np.mean(MSE_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/MSE_loss_L4', np.mean(MSE_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/MSE_loss_L5', np.mean(MSE_loss_L5_per_batch), total_step)
            writer.add_scalar('train_loss/MAPE_loss_L3', np.mean(MAPE_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/MAPE_loss_L4', np.mean(MAPE_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/MAPE_loss_L5', np.mean(MAPE_loss_L5_per_batch), total_step)
            writer.add_scalar('train_loss/domain_loss_L3', np.mean(domain_loss_L3_per_batch), total_step)
            writer.add_scalar('train_loss/domain_loss_L4', np.mean(domain_loss_L4_per_batch), total_step)
            writer.add_scalar('train_loss/domain_loss_L5', np.mean(domain_loss_L5_per_batch), total_step)
            

            # if dropout==False:
            net_3.eval()
            net_4.eval()
            net_5.eval()

            classifier_LSTM.eval()

            valid_MAE_loss_L3 = 0
            valid_MAE_loss_L4 = 0
            valid_MAE_loss_L5 = 0
            valid_MSE_loss_L3 = 0
            valid_MSE_loss_L4 = 0
            valid_MSE_loss_L5 = 0
            valid_MAPE_loss_L3 = 0
            valid_MAPE_loss_L4 = 0
            valid_MAPE_loss_L5 = 0
            
            total_plot_predict_L3=[]
            total_plot_predict_L4=[]
            total_plot_predict_L5=[]
            total_plot_label = []

            step = 0
            with torch.no_grad():
                valid_loss = 0
                for x,y,label,bearing_label in test_loader:
                    step+=1
                    x = x.to(device=self.device, dtype=torch.float64)
                    y = y.to(device=self.device, dtype=torch.float64)
                    label = label.to(device=self.device, dtype=torch.float64)
                    
                    input_signal = torch.cat((x,y), dim=2)
                    input_signal = input_signal.to(device=self.device, dtype=torch.float64)

                    predict, MAE_loss_L3, MSE_loss_L3, MAPE_loss_L3 = self.evaluate_Unet_RUL(net_3, classifier_LSTM, criterion_MAE, criterion_MSE, input_signal, label)
                    valid_MAE_loss_L3 += MAE_loss_L3
                    valid_MSE_loss_L3 += MSE_loss_L3
                    valid_MAPE_loss_L3 += MAPE_loss_L3

                    total_plot_predict_L3 = total_plot_predict_L3 + predict.detach().cpu().numpy().tolist()
                    
                    predict, MAE_loss_L4, MSE_loss_L4, MAPE_loss_L4 = self.evaluate_Unet_RUL(net_4, classifier_LSTM, criterion_MAE, criterion_MSE, input_signal, label)
                    valid_MAE_loss_L4 += MAE_loss_L4
                    valid_MSE_loss_L4 += MSE_loss_L4
                    valid_MAPE_loss_L4 += MAPE_loss_L4

                    total_plot_predict_L4 = total_plot_predict_L4 + predict.detach().cpu().numpy().tolist()
                    
                    predict, MAE_loss_L5, MSE_loss_L5, MAPE_loss_L5 = self.evaluate_Unet_RUL(net_5, classifier_LSTM, criterion_MAE, criterion_MSE, input_signal, label)
                    valid_MAE_loss_L5 += MAE_loss_L5
                    valid_MSE_loss_L5 += MSE_loss_L5
                    valid_MAPE_loss_L5 += MAPE_loss_L5

                    total_plot_predict_L5 = total_plot_predict_L5 + predict.detach().cpu().numpy().tolist()
                    total_plot_label = total_plot_label + label.cpu().numpy().tolist()
                    

            valid_MAE_loss_L3 /= step
            valid_MAE_loss_L4 /= step
            valid_MAE_loss_L5 /= step
            valid_MSE_loss_L3 /= step
            valid_MSE_loss_L4 /= step
            valid_MSE_loss_L5 /= step
            valid_MAPE_loss_L3 /= step
            valid_MAPE_loss_L4 /= step
            valid_MAPE_loss_L5 /= step

            valid_loss = self.lambda_1 * (valid_MAE_loss_L3 + valid_MAE_loss_L4 + valid_MAE_loss_L5) + self.lambda_2 * (valid_MSE_loss_L3 + valid_MSE_loss_L4 + valid_MSE_loss_L5) + self.lambda_3 * (valid_MAPE_loss_L3 + valid_MAPE_loss_L4 + valid_MAPE_loss_L5) #+ epoch_loss_D_L3 + epoch_loss_D_L4 + epoch_loss_D_L5
            writer.add_scalar('valid_loss/valid_loss', valid_loss, (epoch+1))
            writer.add_scalar('valid_loss/MAE_loss_L3', valid_MAE_loss_L3, (epoch+1))
            writer.add_scalar('valid_loss/MAE_loss_L4', valid_MAE_loss_L4, (epoch+1))
            writer.add_scalar('valid_loss/MAE_loss_L5', valid_MAE_loss_L5, (epoch+1))
            writer.add_scalar('valid_loss/MSE_loss_L3', valid_MSE_loss_L3, (epoch+1))
            writer.add_scalar('valid_loss/MSE_loss_L4', valid_MSE_loss_L4, (epoch+1))
            writer.add_scalar('valid_loss/MSE_loss_L5', valid_MSE_loss_L5, (epoch+1))
            writer.add_scalar('valid_loss/MAPE_loss_L3', valid_MAPE_loss_L3, (epoch+1))
            writer.add_scalar('valid_loss/MAPE_loss_L4', valid_MAPE_loss_L4, (epoch+1))
            writer.add_scalar('valid_loss/MAPE_loss_L5', valid_MAPE_loss_L5, (epoch+1))

            if valid_loss < best_valid_loss and epoch > 9:
                best_epoch = epoch
                best_valid_loss = valid_loss
                torch.save(net_3.state_dict(), self.net3_RUL_model_save_path)
                torch.save(net_4.state_dict(), self.net4_RUL_model_save_path)
                torch.save(net_5.state_dict(), self.net5_RUL_model_save_path)
                torch.save(classifier_LSTM.state_dict(), self.LSTM_model_save_path)
                torch.save(discriminator.state_dict(), self.discriminator_RUL_model_save_path)

                total_plot_label = np.array(total_plot_label)
                total_plot_label = total_plot_label.reshape(-1)
                total_plot_predict_L3 = np.array(total_plot_predict_L3)
                total_plot_predict_L3 = total_plot_predict_L3.reshape(-1)
                total_plot_predict_L4 = np.array(total_plot_predict_L4)
                total_plot_predict_L4 = total_plot_predict_L4.reshape(-1)
                total_plot_predict_L5 = np.array(total_plot_predict_L5)
                total_plot_predict_L5 = total_plot_predict_L5.reshape(-1)

                plt.cla()
                plt.clf()
                x_axis = range(FPT_list[self.condition-1][self.test_bearing-1], FPT_list[self.condition-1][self.test_bearing-1] + len(total_plot_label)*self.sequence_length, self.sequence_length)

                plt.plot(x_axis,total_plot_label, '--', color='b', label='Actual RUL')
                plt.legend()
                plt.plot(x_axis,total_plot_predict_L3, color='y', label='L3 Estimation')
                plt.plot(x_axis,total_plot_predict_L4, color='r', label='L4 Estimation')
                plt.plot(x_axis,total_plot_predict_L5, color='g', label='L5 Estimation')
                plt.legend()
                plot_file_name = './RUL_Plot/UnetRUL_'+ '_C' + str(self.condition)+'_'+str(self.test_bearing)+ 'FPT_'+ str(FPT_list[self.condition-1][self.test_bearing-1]) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '_' + str(epoch+1)+'.png'
                plt.savefig(plot_file_name)
            if (epoch % 5 == 0) or (epoch == (self.epochs-1)):
                plt.cla()
                plt.clf()
                x_axis = range(FPT_list[self.condition-1][self.test_bearing-1], FPT_list[self.condition-1][self.test_bearing-1] + len(total_plot_label)*self.sequence_length, self.sequence_length)
                plt.plot(x_axis,total_plot_label, '--', color='b', label='Actual RUL')
                plt.legend()
                plt.plot(x_axis,total_plot_predict_L3, color='y', label='L3 Estimation')
                plt.plot(x_axis,total_plot_predict_L4, color='r', label='L4 Estimation')
                plt.plot(x_axis,total_plot_predict_L5, color='g', label='L5 Estimation')
                plt.legend()
                plot_file_name = './RUL_Plot/UnetRUL5_'+ '_C' + str(self.condition)+'_'+str(self.test_bearing)+ 'FPT_'+ str(FPT_list[self.condition-1][self.test_bearing-1]) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '_' + str(epoch+1)+'.png'
                plt.savefig(plot_file_name)

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} time={:.2f}s L3_train_loss={:.4f} L4_train_loss={:.4f} L5_train_loss={:.4f} L3_D_loss={:.4f} L4_D_loss={:.4f} L5_D_loss={:.4f} L3_MAE_loss={:.4f} L4_MAE_loss={:.4f} L5_MAE_loss={:.4f} L3_MSE_loss={:.4f} L4_MSE_loss={:.4f} L5_MSE_loss={:.4f} L3_MAPE_loss={:.4f} L4_MAPE_loss={:.4f} L5_MAPE_loss={:.4f}'.format(
                                    epoch +1, self.epochs,
                                    elapsed_time,
                                    epoch_loss_L3, epoch_loss_L4, epoch_loss_L5,
                                    epoch_loss_D_L3, epoch_loss_D_L4, epoch_loss_D_L5,
                                    valid_MAE_loss_L3, valid_MAE_loss_L4, valid_MAE_loss_L5,
                                    valid_MSE_loss_L3, valid_MSE_loss_L4, valid_MSE_loss_L5,
                                    valid_MAPE_loss_L3, valid_MAPE_loss_L4, valid_MAPE_loss_L5))
        
        print('Best Epoch {} Best valid loss {:.4f}'.format(best_epoch+1, best_valid_loss))
        

    def sorted_aphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)
   
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
        # condition_list = [self.condition]
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
            FPT_list.append(FPT_condition_list)
            total_length.append(condition_total_length)

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
                        fpt = 0 #len(results)
                        return fpt
                if (results[i+j]==1):
                    count+=1
                    if(count==5):
                        fpt = i
                        return fpt
            count = 0   
        return fpt

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
        
        net_3.load_state_dict(torch.load(self.net3_RUL_model_save_path))
        net_3.to(self.device,dtype=torch.float64)
        net_3.requires_grad = False
        
        net_4.load_state_dict(torch.load(self.net4_RUL_model_save_path))
        net_4.to(self.device,dtype=torch.float64)
        net_4.requires_grad = False
        
        net_5.load_state_dict(torch.load(self.net5_RUL_model_save_path))
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

        save_image_path = './NSP_imageset/' + self.dataset + '_Step2_NSP_C'+str(self.condition) + 'B' + str(self.test_bearing) + '_dropout_ ' + str(self.dropout) + '_' + str(self.n_channels) +'.png'
        cv2.imwrite(save_image_path,v_img)

    def make_nsp(self):
        net_3 = UNet_L3(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)
        net_4 = UNet_L4(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)
        net_5 = UNet_L5(n_channels=2, n_classes=2, input_dimension= 2560, dropout=self.dropout)

        net_3.load_state_dict(torch.load(self.net3_RUL_model_save_path))
        net_3.to(self.device,dtype=torch.float64)
        net_3.requires_grad = False
        
        net_4.load_state_dict(torch.load(self.net4_RUL_model_save_path))
        net_4.to(self.device,dtype=torch.float64)
        net_4.requires_grad = False
        
        net_5.load_state_dict(torch.load(self.net5_RUL_model_save_path))
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
                    save_dir = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step2/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(condition)+ '_' + str(bearing_idx+1)+'/'
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

    def train_cnn_lstm(self, FPT_list):
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
                input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step2/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'
                # input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step2/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path = './MultiScale_NSP_2_Dropout/Step2/Bearing'+str(self.condition)+'_'+str(self.test_bearing) + '/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path = './MultiScale_NSP_2_Dropout/'+self.dataset + '_Step2/Bearing'+str(self.condition)+'_3/' + 'Bearing'+ str(self.condition)+ '_'
                # input_path='./MultiScale_NSP_2_Dropout/Bearing' + str(self.condition) + '_'
            else:
                input_path='./MultiScale_NSP_2/Bearing' + str(self.condition) + '_'
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor()
            ])
        else:
            input_path='./STFT/' + self.dataset + '/Bearing' 
            transform = cvtransforms.Compose([
            # you can add other transformations in this list
                cvtransforms.Resize((self.resize_height,self.resize_width), interpolation='NEAREST'),
                cvtransforms.ToTensor(),
            ])
        
        print(input_path)
        
        train_dataset = Dataload_Step2_FPT(input_path, transform, self.condition, self.num_bearings, self.train_bearing, FPT_list, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
        test_dataset = Dataload_Step2_FPT(input_path, transform, self.condition, self.num_bearings, [self.test_bearing], FPT_list, self.sequence_length)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        model = Classifier_NSP_LSTM(n_channels=self.n_channels, input_dimension=(self.resize_height,self.resize_width,3), sequence_len=self.sequence_length)
        model.to(device=self.device,dtype=torch.float64)

        optimizer_NSPLSTM = torch.optim.Adam(model.parameters(), lr = self.lr, betas=(0.5, 0.99), weight_decay=1e-5)
        
        cuda = True if torch.cuda.is_available() else False
        
        criterion_MSE = nn.MSELoss()
        criterion_MAE = nn.L1Loss()
        
        if cuda:
            criterion_MSE.cuda()
            criterion_MAE.cuda()
        
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
                
                image = image.to(device=self.device, dtype=torch.float64)
                label = label.to(device=self.device, dtype=torch.float64)
                    
                label = torch.reshape(label, (-1,1))
                
                predict = model(image)

                classification_loss_MAE = criterion_MAE(predict, label)
                classification_loss_MSE = torch.sqrt(criterion_MSE(predict, label))
        
                classification_loss_MAPE = torch.mean(torch.abs((label - predict) / label))

                total_loss = classification_loss_MAE * self.lambda_1 + classification_loss_MSE * self.lambda_2 + classification_loss_MAPE * self.lambda_3

                optimizer_NSPLSTM.zero_grad()
                total_loss.backward()
                optimizer_NSPLSTM.step()

                epoch_loss += total_loss.item()
                
            epoch_loss = epoch_loss / step

            model.requires_grad_(False)
            model.eval()

            valid_MAE_loss = 0
            valid_MSE_loss = 0
            valid_MAPE_loss = 0
            
            step = 0
            total_plot_predict=[]
            total_plot_label=[]
            with torch.no_grad():
                for image,label in test_loader:
                    step+=1
                    image = image.to(device=self.device, dtype=torch.float64)
                    label = label.to(device=self.device, dtype=torch.float64)
                    
                    output = model(image)
                    
                    label = torch.reshape(label, (-1,1))

                    MAE_loss = criterion_MAE(output, label)
                    MSE_loss = torch.sqrt(criterion_MSE(output, label))
                    MAPE_loss = torch.mean(torch.abs((label - output) / label))    
                    
                    valid_MAE_loss += MAE_loss.item()
                    valid_MSE_loss += MSE_loss.item()
                    valid_MAPE_loss += MAPE_loss.item()
                    
                    total_plot_predict = total_plot_predict + output.detach().cpu().numpy().tolist()
                    total_plot_label = total_plot_label + label.cpu().numpy().tolist()
                    
            valid_MAE_loss /= step
            valid_MSE_loss /= step
            valid_MAPE_loss /= step

            valid_loss = self.lambda_1 * valid_MAE_loss + self.lambda_2 * valid_MSE_loss + self.lambda_3 * valid_MAPE_loss

            writer.add_scalar('valid_loss/valid_loss', valid_loss, (epoch+1))
            writer.add_scalar('valid_loss/MAE_loss', valid_MAE_loss, (epoch+1))
            writer.add_scalar('valid_loss/MSE_loss', valid_MSE_loss, (epoch+1))
            writer.add_scalar('valid_loss/MAPE_loss', valid_MAPE_loss, (epoch+1))

            if valid_loss < best_valid_loss and epoch > 5:
                best_epoch = epoch
                best_valid_loss = valid_loss
                # torch.save(model.state_dict(), './save_model/cnnhs_model_C'+ str(condition) + 'B' + str(test_bearing) + '_' + str(n_channels) + '_dropout_' + str(dropout) +'.pth')
                total_plot_label = np.array(total_plot_label)
                total_plot_label = total_plot_label.reshape(-1)
                total_plot_predict = np.array(total_plot_predict)
                total_plot_predict = total_plot_predict.reshape(-1)

                plt.cla()
                plt.clf()
                x_axis = range(FPT_list[self.condition-1][self.test_bearing-1], FPT_list[self.condition-1][self.test_bearing-1] + len(total_plot_label)*self.sequence_length, self.sequence_length)
                plt.plot(x_axis,total_plot_label, '--', color='b', label='Actual RUL')
                plt.legend()
                plt.plot(x_axis,total_plot_predict, color='r', label='Predicted RUL')
                plt.legend()
                plot_file_name = './RUL_Plot/UnetNSPRUL_'+ '_C' + str(self.condition)+'_'+str(self.test_bearing)+ 'FPT_'+ str(FPT_list[self.condition-1][self.test_bearing-1]) + '_dropout_' + str(self.dropout) + '_' + str(self.n_channels) + '_' + str(self.sequence_length) + '_' + str(epoch+1)+'.png'
                plt.savefig(plot_file_name)

                            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} time={:.2f}s train_loss={:.4f} test_MAE_loss={:.4f} test_MSE_loss={:.4f} test_MAPA_loss={:.4f} total_valid_loss={:.4f}'.format(
                                    epoch +1, self.epochs,
                                    elapsed_time,
                                    epoch_loss, valid_MAE_loss, valid_MSE_loss, valid_MAPE_loss, valid_loss))

        print('Best Epoch {} Best valid loss {:.4f}'.format(best_epoch+1, best_valid_loss))

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
    parser.add_argument("--lambda_1", type=float, default=20.0, help="lambda1 for loss function")
    parser.add_argument("--lambda_2", type=float, default=20.0, help="lambda2 for loss function")
    parser.add_argument("--lambda_3", type=float, default=50.0, help="lambda3 for loss function")
    parser.add_argument("--lambda_domain", type=float, default=1.0, help="lambda3 for loss function")
    parser.add_argument("--mode", type=str, default='train_unet', help=" 'train_unet', 'make_nsp', 'train_cnn_lstm', 'get_fpt'")
    parser.add_argument("--resize_width", type=int, default=128)
    parser.add_argument("--resize_height", type=int, default=128)
    parser.add_argument("--dropout", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--dataset", type=str, default='femto', help=" 'femto', 'xjtu'")
    parser.add_argument("--FPT", type=int, default=1)

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

    train = TrainOps_step2(train_bearing=train, test_bearing=args.test_bearing, num_bearings=args.num_bearings, condition=args.condition, device=device, epochs=args.epoch, batch_size=args.batch_size, sequence_length=args.sequence_length,
                            lr=args.lr, discriminator_lr=args.discriminator_lr, 
                            lambda_1=args.lambda_1, lambda_2=args.lambda_2, lambda_3=args.lambda_3, lambda_domain=args.lambda_domain, n_channels=args.n_channels, dropout=args.dropout, resize_width=args.resize_width, resize_height=args.resize_height, dataset=args.dataset, FPT=args.FPT)

    if args.mode == 'train_unet':
        if args.FPT == 1:
            FPT_list = train.get_fpt()
        else:
            FPT_list = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

        print(FPT_list)
        train.train_GAN_LSTM(FPT_list)
    elif args.mode == 'intermediate':
        train.save_intermediate_imageset()
    elif args.mode == 'make_nsp':
        train.make_nsp()
    elif args.mode == 'train_LSTM':
        if args.FPT == 1:
            FPT_list = train.get_fpt()
        else:
            FPT_list = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        print(FPT_list)
        train.train_cnn_lstm(FPT_list)
    elif args.mode == 'step_LSTM':
        print('save intermediate imageset')
        train.save_intermediate_imageset()
        print('make nsp images')
        train.make_nsp()
        if args.FPT==1:
            FPT_list = train.get_fpt()
        else:
            FPT_list = [0,0,0,0,0,0,0]

        print(FPT_list)
        print('train CNN-LSTM')
        train.train_cnn_lstm(FPT_list)

