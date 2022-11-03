from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd
from plotly.graph_objs import *
import plotly
import torch
from prettytable import PrettyTable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_time_series(data, class_name, ax,colour,i, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 3 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path,color=colour, linewidth=3)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.450
  )
  ax.set_title(class_name)
  ax.set_ylim([-0.1, 0.1])
  ax.set_ylabel('Amplitude (V)')
  ax.set_xlabel('Window size (μs)')
  


def plot_prediction(data, model, title, ax):
  
  predictions,pred_losses = model.ploterrortransform(data,device)
  ax.plot(data,'black' , label='Original')
  ax.plot(predictions[0],'r', label='VAE prediction')
  ax.set_title(f'{title} (loss: {"{:.2f}".format(pred_losses[0])})')
  ax.legend()
  
  
def boxplotsupport(losses,classname):
    
    losses = np.asarray(losses)
    c=len(losses)
    filename = classname
    numbers = np.random.randn(c)
    df = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df=df.drop(['numbers'], axis=1)
    return df,losses
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

