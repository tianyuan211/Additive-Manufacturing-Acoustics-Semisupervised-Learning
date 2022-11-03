import os, glob, inspect, time, math, torch

import numpy as np
import matplotlib.pyplot as plt
import loss_functions as lfs
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

print(PACK_PATH)


def training(neuralnet, dataset, epochs, batch_size):
    
    train_loader = DataLoader(dataset = dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  drop_last=False)

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))


    start_time = time.time()

    iteration = 0
    writer = SummaryWriter()

    list_enc, list_con, list_adv, list_tot = [], [], [], []
    Encoder_loss, Decoder_loss, Discriminator_loss,Total_Epochs = [], [], [], []
    
    for epoch in range(epochs):
        for t, X in enumerate(train_loader):
            
            
            x_tr_torch=X.squeeze(-1)
            x_tr_torch=x_tr_torch.unsqueeze(1)
        
            z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
            x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
            z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))
            dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
            
            dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))
            
            

            l_tot, l_enc, l_con, l_adv = \
                lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, \
                dis_x, dis_x_hat, features_real, features_fake)

            neuralnet.optimizer.zero_grad()
            l_tot.backward()
            neuralnet.optimizer.step()
            list_enc.append(l_enc)
            list_con.append(l_con)
            list_adv.append(l_adv)
            list_tot.append(l_tot)

            iteration += 1
            
        l_enc=l_enc.detach().cpu().numpy()
        l_con=l_con.detach().cpu().numpy()
        l_adv=l_adv.detach().cpu().numpy()
        Encoder_loss.append(l_enc)
        Decoder_loss.append(l_con)
        Discriminator_loss.append(l_adv)
        Total_Epochs.append(epoch)
        
        print("Epoch [%d / %d] (%d iteration)  Enc:%.3f, Con:%.3f, Adv:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, l_enc, l_con, l_adv, l_tot))
        
        for idx_m, model in enumerate(neuralnet.models):
            torch.save(model.state_dict(), PACK_PATH+"/model-%d" %(idx_m))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    return Encoder_loss, Decoder_loss, Discriminator_loss,Total_Epochs

def model_loading(neuralnet,param_paths):
    
    if(len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH+"/model-%d" %(idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(PACK_PATH+"/model-%d" %(idx_p)))
            neuralnet.models[idx_p].eval()
    return  neuralnet
    

def testing(neuralnet, dataset):


    param_paths = glob.glob(os.path.join(PACK_PATH))
    param_paths.sort()

    neuralnet=model_loading(neuralnet,param_paths)

    print("\nTest...")
    
    test_loader = DataLoader(dataset = dataset,
                                  batch_size = 1,
                                  shuffle = False,
                                  drop_last=True)
    scores_normal, scores = [], []
    for t, X in enumerate(test_loader):
            
            
            x_tr_torch=X.squeeze(-1)
            x_tr_torch=x_tr_torch.unsqueeze(1)
        
            
            enc=neuralnet.encoder.eval()
            dec=neuralnet.decoder.eval()
            disc=neuralnet.discriminator.eval()
            
            
            z_code = enc(x_tr_torch.to(neuralnet.device))
            x_hat = dec(z_code.to(neuralnet.device))
            z_code_hat = enc(x_hat.to(neuralnet.device))
            
            
            dis_x, features_real = disc(x_tr_torch.to(neuralnet.device))           
            dis_x_hat, features_fake = disc(x_hat.to(neuralnet.device))
            
            
            l_tot, l_enc, l_con, l_adv = \
                lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, \
                dis_x, dis_x_hat, features_real, features_fake)
                    
            score_anomaly = l_con.item() 
            l_con=l_con.detach().cpu().numpy()
            scores.append(l_con)
    normal_avg, normal_std = np.average(score_anomaly), np.std(score_anomaly)
    outbound = normal_avg + (normal_std * 3)
    
    return  scores,outbound   
   

def reconstruction(neuralnet, dataset):

    param_paths = glob.glob(os.path.join(PACK_PATH))
    param_paths.sort()
    neuralnet=model_loading(neuralnet,param_paths)

    print("\nReconstruction...")
    
    test_loader = DataLoader(dataset = dataset,
                                  batch_size = 1,
                                  shuffle = False,
                                  drop_last=False)
    scores_normal, scores = [], []
    for t, X in enumerate(test_loader):
            
            
            x_tr_torch=X.squeeze(-1)
            x_tr_torch=x_tr_torch.unsqueeze(1)
        
            
            enc=neuralnet.encoder.eval()
            dec=neuralnet.decoder.eval()
            disc=neuralnet.discriminator.eval()
            
            
            z_code = enc(x_tr_torch.to(neuralnet.device))
            x_hat = dec(z_code.to(neuralnet.device))
            
            z_code_hat = enc(x_hat.to(neuralnet.device))
            
            
            dis_x, features_real = disc(x_tr_torch.to(neuralnet.device))           
            dis_x_hat, features_fake = disc(x_hat.to(neuralnet.device))
            
            
            l_tot, l_enc, l_con, l_adv = \
                lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, \
                dis_x, dis_x_hat, features_real, features_fake)
                    
            
            score_anomaly = l_con.item() 
            loss=l_con.detach().cpu().numpy()
            raw=x_tr_torch.detach().cpu().numpy().squeeze()
            constructed=x_hat.detach().cpu().numpy().squeeze()
            
  
    return  loss, raw ,constructed 

