
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from solver import training,testing,reconstruction


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
  
def plot_prediction(data, neuralnet, title, ax):
  
  pred_losses, data,predictions = reconstruction(neuralnet=neuralnet, dataset=data) 
  # predictions,pred_losses = model.ploterrortransform(data,device)
  ax.plot(data,'black' , label='Original')
  ax.plot(predictions,'r', label='GAN prediction')
  ax.set_title(f'{title} (loss: {"{:.2f}".format(pred_losses)})')
  ax.legend()
    
  

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