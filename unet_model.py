from matplotlib.pyplot import xcorr
import torch
import torch.nn.functional as F
import torch.nn as nn
from unet_parts import *


class Discriminator(nn.Module):
    def __init__(self, in_channels, input_dimension):
        super(Discriminator, self).__init__()

        self.input_dimension = input_dimension

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True)
        )

        self.flatten_size = int(self.input_dimension / 16 * 256)

        self.FCs = nn.Sequential(
            nn.Linear(self.flatten_size,1024),
            nn.LeakyReLU(True),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv1(x)           #[-1, 32, 1280]
        x = self.conv2(x)           #[-1, 64, 640]
        x = self.conv3(x)           #[-1, 128, 320]
        x = self.conv4(x)           #[-1, 256, 160]
        flatten = x.view(-1, self.flatten_size)

        output = self.FCs(flatten)
        return output

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels, input_dimension, n_domain):
        super(DomainDiscriminator, self).__init__()

        self.input_dimension = input_dimension
        self.n_domain = n_domain

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Dropout(0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Dropout(0.25)
        )

        self.flatten_size = int(self.input_dimension / 16 * 256)
        
        self.FC = nn.Sequential(
            nn.Linear(self.flatten_size,1024),
            nn.LeakyReLU(True)
        )
        self.adv_layer = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        self.domain_layer = nn.Sequential(
            nn.Linear(1024, n_domain),
            # nn.Softmax()
        )

    def forward(self,x):
        x = self.conv1(x)           #[-1, 32, 1280]
        x = self.conv2(x)           #[-1, 64, 640]
        x = self.conv3(x)           #[-1, 128, 320]
        x = self.conv4(x)           #[-1, 256, 160]
        flatten = x.view(-1, self.flatten_size)

        out = self.FC(flatten)
        
        output = self.adv_layer(out)
        domain_label = self.domain_layer(out)

        return output, domain_label

class UNet_L3(nn.Module):
    def __init__(self, n_channels, n_classes, input_dimension, bilinear=True, dropout=False):
        super(UNet_L3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_dimension = input_dimension
        n_output = 32

        self.inc = DoubleConv(self.n_channels, n_output)
        self.down1 = Down(n_output, n_output*2)
        self.down2 = Down(n_output*2, n_output*4)
        self.down3 = Down(n_output*4, n_output*8, dropout=dropout)
        

        self.up1 = Up(n_output*8, n_output*4, bilinear, dropout=dropout)
        self.up2 = Up(n_output*4, n_output*2, bilinear)
        self.up3 = Up(n_output*2, n_output, bilinear)
        self.outc = OutConv(n_output, self.n_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dimension*self.n_channels,512),
            nn.ReLU(),
            nn.Linear(512,self.n_classes)
        )

    def forward(self, x):   # x [-1 1 2560]
        x1 = self.inc(x)    # x1 [-1, 8, 2560]
        x2 = self.down1(x1) # x2 [-1, 16, 1280]
        x3 = self.down2(x2) # x3 [-1, 32, 640]
        x4 = self.down3(x3) # x4 [-1, 64, 320]

        x = self.up1(x4, x3)   # [-1, 48,640]
        x = self.up2(x, x2)    # [-1, 32, 1280]
        x = self.up3(x, x1)    # [-1, 20, 2560]
        
        
        logits = self.outc(x)
        return logits
        # logits = logits.view(-1,self.input_dimension*self.n_channels)
        # result = self.classifier(logits)
        # return result

    def extractor(self,x): # [-1, 2, 2560]
        x1 = self.inc(x)    # [-1, 8, 2560]
        x2 = self.down1(x1) # [-1, 16, 1280]
        x3 = self.down2(x2) # [-1, 32, 640]
        x4 = self.down3(x3) 

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet_L4(nn.Module):
    def __init__(self, n_channels, n_classes, input_dimension, bilinear=True, dropout=False):
        super(UNet_L4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_dimension = input_dimension
        n_output=32

        self.inc = DoubleConv(self.n_channels, n_output)
        self.down1 = Down(n_output, n_output*2)
        self.down2 = Down(n_output*2, n_output*4)
        self.down3 = Down(n_output*4, n_output*8, dropout=dropout)
        self.down4 = Down(n_output*8, n_output*16, dropout=dropout)

        self.up1 = Up(n_output*16, n_output*8, bilinear, dropout=dropout)
        self.up2 = Up(n_output*8, n_output*4, bilinear, dropout=dropout)
        self.up3 = Up(n_output*4, n_output*2, bilinear)
        self.up4 = Up(n_output*2, n_output, bilinear)
        self.outc = OutConv(n_output, self.n_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dimension*self.n_channels,512),
            nn.ReLU(),
            nn.Linear(512,self.n_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)    
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4) # batch,128,160
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        # logits = logits.view(-1,self.input_dimension*self.n_channels)
        # result = self.classifier(logits)
        # return result
    
    def extractor(self,x):
        x1 = self.inc(x)    
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_L5(nn.Module):
    def __init__(self, n_channels, n_classes, input_dimension, bilinear=True, dropout=False):
        super(UNet_L5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_dimension = input_dimension
        n_output=32

        self.inc = DoubleConv(self.n_channels, n_output)
        self.down1 = Down(n_output, n_output*2)
        self.down2 = Down(n_output*2, n_output*4)
        self.down3 = Down(n_output*4, n_output*8)
        self.down4 = Down(n_output*8, n_output*16, dropout=dropout)
        self.down5 = Down(n_output*16, n_output*32, dropout=dropout)

        self.up1 = Up(n_output*32, n_output*16, bilinear, dropout=dropout)
        self.up2 = Up(n_output*16, n_output*8, bilinear, dropout=dropout)
        self.up3 = Up(n_output*8, n_output*4, bilinear)
        self.up4 = Up(n_output*4, n_output*2, bilinear)
        self.up5 = Up(n_output*2, n_output, bilinear)
        self.outc = OutConv(n_output, self.n_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dimension*self.n_channels,512),
            nn.ReLU(),
            nn.Linear(512,self.n_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)    #1280,1
        x2 = self.down1(x1) # 640,1
        x3 = self.down2(x2) # 320, 1
        x4 = self.down3(x3) # 160, 1
        x5 = self.down4(x4) # 80,1
        x6 = self.down5(x5) # 40,1

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        logits = self.outc(x)

        return logits
        # logits = logits.view(-1,self.input_dimension*self.n_channels)
        # result = self.classifier(logits)
        # return result

    def extractor(self,x):
        x1 = self.inc(x)    #1280,1
        x2 = self.down1(x1) # 640,1
        x3 = self.down2(x2) # 320, 1
        x4 = self.down3(x3) # 160, 1
        x5 = self.down4(x4) # 80,1
        x6 = self.down5(x5) # 40,1

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        logits = self.outc(x)

        return logits

class Classifier_FCs(nn.Module):
    def __init__(self, n_channels, n_classes, input_dimension):
        super(Classifier_FCs, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_dimension = input_dimension

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dimension*self.n_channels,512),
            nn.ReLU(),
            nn.Linear(512,self.n_classes)
        )

    def forward(self, x):
        logits = x.view(-1,self.input_dimension*self.n_channels)

        result = self.classifier(logits)
        return result

class CNN_HS(nn.Module):
    def __init__(self, input_dimension, n_classes):
        super(CNN_HS,self).__init__()

        self.input_dimension = input_dimension
        self.n_classes = n_classes


        self.conv1 = nn.Sequential(
            nn.Conv2d(3,20, kernel_size=(10,10), stride=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(20,40, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(40,20, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU()
        )
        
        flatten_size = int(self.input_dimension[0]/32*self.input_dimension[1]/32*20)
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size,120),
            nn.ReLU(),
            nn.Linear(120,self.n_classes)
        )

    def forward(self, x):
        batch_size, C, H ,W = x.size()  #[-1, 3, 128, 128]
        # print('input:', x.size())
        x = self.conv1(x)               #[-1,20, 30, 30]
        # print('conv1:', x.size())
        x = self.conv2(x)               #[-1, 40, 6, 6]
        # print('conv2:', x.size())
        x = self.conv3(x)               #[-1, 20, 4, 4]
        # print('conv3:', x.size())
        
        batch_size, C, H, W = x.size()
        flatten = x.view(-1, C*H*W)
        # print('flatten', logits.size())
        logits = self.fc(flatten)


        return logits

class Classifier_LSTM(nn.Module):
    def __init__(self, n_channels, input_dimension, sequence_len):
        super(Classifier_LSTM, self).__init__()
        self.n_channels = n_channels
        self.input_dimension = input_dimension
        self.sequence_len = sequence_len
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 10, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.maxpool1 = nn.MaxPool1d(3,2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 1, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.maxpool2 = nn.MaxPool1d(3,2, padding=1)

        self.lstm_bn = nn.BatchNorm1d(int(self.input_dimension*2))
        self.lstm = nn.LSTM(int(self.input_dimension*2),100,3, dropout=0.5, batch_first=True, bidirectional=False)   
        
        self.classifier = nn.Sequential(
            # nn.Linear(100,50),
            # nn.LeakyReLU(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, sequence_len, C, D = x.size()
        # c_in = torch.reshape(x, (batch_size * sequence_len, C, D))
        # c_out = self.conv1(c_in)    
        # c_out = self.maxpool1(c_out)    
        # c_out = self.conv2(c_out)    
        # c_out = self.maxpool1(c_out)  
        # # print('maxpool2:',c_out.size())

        # r_in = c_out.view(batch_size, sequence_len, -1)
        # print('r_in:',r_in.size())
        r_in = x.view(batch_size, sequence_len, -1)

        # r_in = r_in.permute(0,2,1)
        # r_in_bn = self.lstm_bn(r_in)
        # r_in_bn = r_in_bn.permute(0,2,1)
        r_out, (h_n,h_c) = self.lstm(r_in)
        # print('r_out:',r_out.size())

        r_out2=r_out[:, -1, :]
        # print('r_out2:',r_out2.size())
        result = self.classifier(r_out2)
        
        # print('result:',result.size())
        # exit()
        return result

class Classifier_NSP_LSTM(nn.Module):
    def __init__(self, n_channels, input_dimension, sequence_len):
        super(Classifier_NSP_LSTM, self).__init__()
        self.n_channels = n_channels
        self.input_dimension = input_dimension
        self.sequence_len = sequence_len
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,20, kernel_size=(10,10), stride=(2,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(20,40, kernel_size=(5,5), stride=(2,2)),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(40,20, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        
        flatten_size = int(self.input_dimension[0]/32*self.input_dimension[1]/32*20)

        self.lstm = nn.LSTM(flatten_size,100,3, dropout=0.5, batch_first=True, bidirectional=False)   
        
        self.classifier = nn.Sequential(
            nn.Linear(100,50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, sequence_len, C, H, W = x.size()
        c_in = torch.reshape(x, (batch_size * sequence_len, C, H, W))
        c_out = self.conv1(c_in)
        c_out = self.conv2(c_out)
        c_out = self.conv3(c_out)
        bs, C, H, W = c_out.size()

        flatten = c_out.view(-1, H*W*C)
        r_in = torch.reshape(flatten, (batch_size, sequence_len, -1))
        # print('r_in', r_in.size())
        # r_in = r_in.permute(0,2,1)
        # r_in_bn = self.lstm_bn(r_in)
        # r_in_bn = r_in_bn.permute(0,2,1)
        r_out, (h_n,h_c) = self.lstm(r_in)
        # print('r_out:',r_out.size())
        # print('r_out', r_out.size())

        r_out2=r_out[:, -1, :]
        # print('r_out2:',r_out2.size())
        result = self.classifier(r_out2)
        
        # print('result:',result.size())
        # exit()
        return result

if __name__ == "__main__":
    unetunet = UNet_L5(n_channels=1, n_classes=1, bilinear=True)
    test_tensor = torch.randn(32,1,2560)
    check = unetunet(test_tensor)
    print(check.size())