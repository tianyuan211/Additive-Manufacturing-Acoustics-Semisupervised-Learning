import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from torch.nn import functional as F

class NeuralNet(object):

    def __init__(self, device, ngpu, dropout_rate,learning_rate=1e-3):

        self.device, self.ngpu = device, ngpu
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.encoder =  Encoder(dropout_rate,ngpu).to(self.device)
        self.decoder =  Decoder(dropout_rate,ngpu).to(self.device)
        self.discriminator = Discriminator(dropout_rate,ngpu).to(self.device)

        self.models = [self.encoder, self.decoder, self.discriminator]

        
        for idx_m, model in enumerate(self.models):
            if(self.device.type == 'cuda') and (self.models[idx_m].ngpu > 0):
                self.models[idx_m] = nn.DataParallel(self.models[idx_m], list(range(self.models[idx_m].ngpu)))

        self.num_params = 0
        for idx_m, model in enumerate(self.models):
            for p in model.parameters():
                self.num_params += p.numel()
            print(model)
        print("The number of parameters: %d" %(self.num_params))

        self.params = None
        for idx_m, model in enumerate(self.models):
            if(self.params is None):
                self.params = list(model.parameters())
            else:
                self.params = self.params + list(model.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate)
        
#%%



class Encoder(nn.Module): 
    def __init__(self,dropout_rate,ngpu):
        super(Encoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.ngpu = ngpu
        #input 500
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=3) 
        self.bn1 = nn.BatchNorm1d(16) 
        #output 2499
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9,stride=3)
        self.bn2 = nn.BatchNorm1d(32) 
        #output 832
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=9,stride=3)
        self.bn3 = nn.BatchNorm1d(64) 
        #output 276
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=9,stride=3)
        self.bn4 = nn.BatchNorm1d(128) 
        #output 90
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=9,stride=3)
        self.bn5 = nn.BatchNorm1d(256) 
        #output 90
                
        self.rnn1 = nn.Linear(17, 10)
        self.dropout = nn.Dropout(dropout_rate) 
        
    
    def forward(self, x):
        
       
        #Encoder
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        
        
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = torch.tanh(self.bn4(self.conv4(x)))
        x = self.dropout(x)
       
        x = torch.tanh(self.bn5(self.conv5(x)))
        x = self.dropout(x)
       
        x = self.rnn1(x)
        
        return x

#%%
    
class Decoder(nn.Module): 
    def __init__(self,dropout_rate,ngpu):
        super(Decoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.ngpu = ngpu
       
        self.rnn2 = nn.Linear(10, 17)
        
        self.bn5 = nn.BatchNorm1d(256)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=9,stride=3,output_padding=1)
       
        self.bn6 = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=9,stride=3,output_padding=2)
        
        self.bn7 = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=9,stride=3,output_padding=0)
        
        self.bn8 = nn.BatchNorm1d(32)
        self.deconv4 = nn.ConvTranspose1d(32, 16, kernel_size=9,stride=3,output_padding=2)
        
        self.bn9 = nn.BatchNorm1d(16)
        self.deconv5 = nn.ConvTranspose1d(16, 1, kernel_size=9,stride=3,output_padding=2)
        
        self.dropout = nn.Dropout(dropout_rate) 
                
    def forward(self, x):
        
        
        x = self.rnn2(x)

        x = torch.tanh(self.deconv1(self.bn5(x)))
        x = self.dropout(x)     

        x = torch.tanh(self.deconv2(self.bn6(x)))
        x = self.dropout(x)

        x = torch.tanh(self.deconv3(self.bn7(x)))
        x = self.dropout(x)

        x = torch.tanh(self.deconv4(self.bn8(x)))
        x = self.dropout(x)

        x = self.deconv5(x)
       
        return x

#%%

class Discriminator(nn.Module): 
    def __init__(self,dropout_rate,ngpu):
        super(Discriminator, self).__init__()
        self.dropout_rate = dropout_rate
        self.ngpu = ngpu
        
        self.dis_conv = nn.ModuleList([
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=3), 
        # printing(),
        nn.BatchNorm1d(16),
        nn.Tanh(),
        nn.MaxPool1d(2,2),
        
        nn.Conv1d(16, 32, kernel_size=9,stride=3),
        # printing(),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.MaxPool1d(2,2),
        
        nn.Conv1d(32, 64, kernel_size=9,stride=3),
        # printing(),
        nn.BatchNorm1d(64),
        nn.Tanh(),
        nn.MaxPool1d(2,2),
        
        nn.Conv1d(64, 128, kernel_size=9,stride=3),
        # printing(),
        nn.BatchNorm1d(128),
        nn.Tanh(),
        nn.MaxPool1d(2,2),
        
        
        ])
        
        self.dis_dense = nn.ModuleList([
            nn.Flatten(), #4352
           
            nn.Linear(256, 32),
            
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            
            nn.Linear(32, 1),
            
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        ])

    
    def forward(self, input):
        
        featurebank = []

        for idx_l, layer in enumerate(self.dis_conv):
            input = layer(input)
            
            if("torch.nn.modules.activation" in str(type(layer))):
                
                featurebank.append(input)
        convout = input
        

        for idx_l, layer in enumerate(self.dis_dense):
            input = layer(input)
            
            if("torch.nn.modules.activation" in str(type(layer))):
                featurebank.append(input)
        disc_score = input
        

        return disc_score, featurebank
        
#%%
class Flatten(nn.Module):
    def forward(self, input):
        input=input.view(input.size(0), -1)
        return input
    
#%%

class printing(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input 

#%%



