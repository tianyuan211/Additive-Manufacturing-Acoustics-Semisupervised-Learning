import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from torch.nn import functional as F

#%%
class Encoder(nn.Module): 
    def __init__(self,dropout_rate):
        super(Encoder, self).__init__()
        self.dropout_rate = dropout_rate
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
        
        # print(x.shape)
        #Encoder
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        # print(x.shape)
        
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        x = torch.tanh(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        x = torch.tanh(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        # print(x.shape)
            
        x = self.rnn1(x)
        # print(x.shape)
        
        return x

#%%

class Decoder(nn.Module): 
    def __init__(self,dropout_rate):
        super(Decoder, self).__init__()
        
       
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
        # print(x.shape)
        
        x = torch.tanh(self.deconv1(self.bn5(x)))
        x = self.dropout(x)     
        # print(x.shape)
        
        x = torch.tanh(self.deconv2(self.bn6(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        
        x = torch.tanh(self.deconv3(self.bn7(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        x = torch.tanh(self.deconv4(self.bn8(x)))
        x = self.dropout(x)
        # print(x.shape)
        
        # x = torch.tanh(self.deconv5(self.bn9(x)))
        x = self.deconv5(x)
       
        return x

#%%

class Lambda(nn.Module):
    
    #def __init__(self, hidden_size, latent_length):
    def __init__(self):
        super(Lambda, self).__init__()

    def forward(self, cell_output):
               
        self.latent_mean = cell_output
        self.latent_logvar = cell_output

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean    

#%%

class VRAE(BaseEstimator, nn.Module):
    
    def __init__(self, sequence_length, number_of_features, hidden_size=27, latent_length=27,
             batch_size=100, learning_rate=0.01,
             n_epochs=100, dropout_rate=0.2, optimizer='SGD', loss='MSELoss',
             cuda=False, print_every=2, clip=True, max_grad_norm=5, dload='.'):
    
        super(VRAE, self).__init__()
        
        
        self.dtype = torch.FloatTensor
        self.use_cuda = cuda
        
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
        
        
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
        
        
        self.encoder = Encoder(dropout_rate)
        self.lmbd = Lambda()
        self.decoder = Decoder(dropout_rate)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload
        
        if self.use_cuda:
            self.cuda()
        
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')
        
        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss()


    def forward(self, x):
    
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)
        
        return x_decoded, latent

    
    ##Calculate loss
    def _rec(self, x_decoded, x, loss_fn):
    
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)
        
        return kl_loss + recon_loss, recon_loss, kl_loss
    
    def compute_loss(self, X):
    
        x=Variable(X.type(self.dtype), requires_grad = True)
        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)
        
        return loss, recon_loss, kl_loss, x
        
   ## Training block
    def _train(self, train_loader):
    
        self.train()
        epoch_loss = 0
        kl = 0
        t = 0
        train_loss = []
        kl_training_loss = []
        
        for t, X in enumerate(train_loader):
            
            data = X
            data=data.squeeze(-1)
            X=data.unsqueeze(1)
        
            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            loss.backward()
        
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
                
                
            epoch_loss += loss.item()
            kl += kl_loss.item()
            self.optimizer.step()
        
            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))
        
            
            
        loss_train = epoch_loss / len(train_loader)
        train_loss.append(loss_train)
        loss_kl = kl / len(train_loader)
        kl_training_loss.append(loss_kl)
        print('Epoch loss: {:.4f}'.format(loss_train))
        print('Average loss: {:.4f}'.format(epoch_loss / t))
        
        
        return train_loss, kl_training_loss
    
    ## Call from the main block
    def fit(self, dataset, save = False):
    
        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=False)
        train_loss = []
        kl_training_loss = []
        
        for i in range(self.n_epochs):
            print('Epoch: %s' % i)
        
            loss,kl=self._train(train_loader)
            train_loss.append(loss)
            kl_training_loss.append(kl)
        self.is_fitted = True
        if save:
            self.save('model.pth')
        
        return train_loss, kl_training_loss
    
    
    ## Signal output from the bottle neck layer
    def _batch_transform(self, x):
             
        z=self.encoder(Variable(x.type(self.dtype), requires_grad = False))
        z=self.lmbd(z)
        z=z.squeeze(0).detach().cpu().numpy()
        
        return z
  
     ## Signal reconstruction after training
    def _batch_reconstruct(self, x):
    
        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)
        return x_decoded
    
    #Saving the model
    def save(self, file_name):
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

#%%
    def transform(self, dataset, save = False):
    
        self.eval()
        
        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=False) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []
        
                for t, x in enumerate(test_loader):
                    data = x
                    
                    data=data.squeeze(-1)
                    x=data.unsqueeze(1)
                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)
        
                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run
        
        raise RuntimeError('Model needs to be fit')
        
    def fit_transform(self, dataset, save = False):
    
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)
    
    def errortransform(self, dataset, device,save = False):
    
        self.eval()
        
        test_loader = DataLoader(dataset = dataset,
                                 batch_size = 1,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []
                x_dash = []
                for t, x in enumerate(test_loader):
                    data = x
                    data=data.squeeze(-1)
                    x=data.unsqueeze(1)
                    x_decoded = self._batch_reconstruct(x)
                    loss = self.loss_fn(x_decoded.to(device), x.to(device))
                    loss=loss.cpu().data.numpy()
                    x_decoded=x_decoded.cpu().data.numpy()
                    z_run.append(loss)
                    x_dash.append(x_decoded)
                    
                return z_run,x_dash
        
        raise RuntimeError('Model needs to be fit')
        
    def ploterrortransform(self, x, device,save = False):
    
        self.eval()
        if self.is_fitted:
            with torch.no_grad():
                losses = []
                predictions = []
                # for t, x in enumerate(test_loader):
                data = x
                
                data=data.squeeze(1)
                data=data.unsqueeze(0)
                x=data.unsqueeze(0)
                #print(x.shape)
                
                
                x_decoded = self._batch_reconstruct(x)
                loss = self.loss_fn(x_decoded.to(device), x.to(device))
                loss=loss.cpu().data.numpy()
                x_decoded=x_decoded.squeeze(0)
                x_decoded=x_decoded.squeeze(0)
                x_decoded=x_decoded.cpu().data.numpy()
                losses.append(loss)
                predictions.append(x_decoded)
                    
                return predictions,losses
        
        raise RuntimeError('Model needs to be fit')
        
