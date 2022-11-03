from neuralnet import NeuralNet
from utils import *
from Material_processing import *
from solver import training,testing,reconstruction


import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import time 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.cuda.empty_cache()
import neuralnet as nn
#%%

windowsize=5000
Material1 = "Inconel"
classname="Conduction mode"
class_1 = 'Balling'
class_2 = 'LoF pores' 
class_3 = 'Conduction mode'
class_4 = 'Keyhole pores'

#%%
# Whole dataset preparation

df=MaterialTsne(Material1, classname, windowsize)
df.target.value_counts()
class_names = [class_1,class_2,class_3,class_4]

#%%
# Display weightage of different classes
sns.set(font_scale = 1)
sns.set_style("whitegrid", {'axes.grid' : False})
graphname=str(Material1)+'_weightage'+'.png'
fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax = sns.countplot(df.target,palette=["#fbab17", "#0515bf", "#10a310", "#e9150d"])
ax.set_xticklabels(class_names);
ax.xaxis.label.set_size(10)
plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1,dpi=800)
plt.show()
plt.clf()
#%%
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})


#%%
# Whole dataset preparation
colour = ["#fbab17", "#0515bf", "#10a310", "#e9150d"]
graphname=str(Material1)+' moving average visualisation'+'.png'
classes = df.target.unique()
classes=np.sort(classes)
fig, axs = plt.subplots(
  nrows=2,
  ncols=2,
  sharey=False,
  figsize=(15, 7),
  dpi=800
)

for i, cls in enumerate(classes):
  ax = axs.flat[i]
  data = df[df.target == cls]     .drop(labels='target', axis=1)     .mean(axis=0)     .to_numpy()
  plot_time_series(data, class_names[i], ax,colour[i],i)
fig.tight_layout();
plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1)
plt.show()
plt.clf()


#%%

Material_1 = Normal_Regime(Material1, classname,windowsize) #selecting the normal class data
Material_1_database,Material_1_labels=dataprocessing (Material_1) #spliting the data into data and label
normaldataset, sequence_length, number_of_features = create_dataset(Material_1_database)

#%%
ngpu=1
lr=1e-3
epoch=300
batch=100
dropout_rate=0.2
#%%

if(not(torch.cuda.is_available())): ngpu = 0
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
neuralnet = nn.NeuralNet(device=device, ngpu=ngpu,dropout_rate=dropout_rate, learning_rate=lr,)
Encoder_loss, Decoder_loss, Discriminator_loss,Total_Epochs=training(neuralnet=neuralnet, dataset=normaldataset, epochs=epoch, batch_size=batch)

#%%

fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax.plot(Encoder_loss,color='red')
#ax.plot(validation_losses)
plt.ylabel('Encoder_loss')
plt.xlabel('Epochs')
plt.legend(['Encoder_loss'])
plt.title('Encoder loss')
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
plt.savefig('Encoder_loss.png',bbox_inches='tight',pad_inches=0.1,dpi=200)
plt.show();
plt.clf()


fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax.plot(Decoder_loss,color='blue')
#ax.plot(validation_losses)
plt.ylabel('Construction loss')
plt.xlabel('Epochs')
plt.legend(['Construction loss'])
plt.title('Construction loss')
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
plt.savefig('Decoder_loss.png',bbox_inches='tight',pad_inches=0.1,dpi=200)
plt.show();
plt.clf()


fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax.plot(Discriminator_loss,color='green')
#ax.plot(validation_losses)
plt.ylabel('Generator_loss')
plt.xlabel('Epochs')
plt.legend(['Generator_loss'])
plt.title('Generator loss')
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
plt.savefig('Generator_loss.png',bbox_inches='tight',pad_inches=0.1,dpi=200)
plt.show();
plt.clf() 



#%%
# Threshold Calculation
df=MaterialTemplate(Material1,windowsize)
losses_three,_=abnormaliy(df,class_3,neuralnet)
scores_normal = np.asarray(losses_three)
normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
Threshold = normal_avg + (normal_std * 2)
print('Threshold:',Threshold)
#%%

losses_three=class_normaliy(df,class_3,neuralnet,Threshold,color="#10a310")
losses_one=classabnormaliy(df,class_1,neuralnet,Threshold,color="#fbab17")
losses_two=classabnormaliy(df,class_2,neuralnet,Threshold,color="#0515bf")
losses_four=classabnormaliy(df,class_4,neuralnet,Threshold,color="#e9150d")

#%%

df3,one=boxplotsupport(losses_one,'Balling')
df4,two=boxplotsupport(losses_two,'LoF pores')
df5,three=boxplotsupport(losses_three,'Conduction mode')
df6,four=boxplotsupport(losses_four,'Keyhole pores')

categories=np.concatenate((df3,df4,df5,df6), axis=0)
classvalue=np.concatenate((one,two,three,four), axis=0)
categories = pd.DataFrame(categories)
classvalue = pd.DataFrame(classvalue)

boxplot_data = [classvalue, categories]
boxplot_data = pd.concat(boxplot_data,axis=1)

boxplot_data.columns = ['losses', 'classes']
plt.figure(figsize=(9,6), dpi=100)
sns.boxplot(x = 'classes', y = 'losses', data = boxplot_data,linewidth=0.5,palette=["#fbab17", "#0515bf", "#10a310", "#e9150d"])
plt.savefig('losses_boxplot.png',bbox_inches='tight',pad_inches=0.1,dpi=200)
plt.show()

#%%


Normal = Normal_Regime(Material1, classname,windowsize)
Normal,Normal_labels=dataprocessing (Normal)
Normal, sequence_length, number_of_features = create_dataset(Normal)

Anomaly = Abnormal_Regime(Material1, classname,windowsize)
Anomaly,Anomaly_labels=dataprocessing (Anomaly)
Anomaly, sequence_length, number_of_features = create_dataset(Anomaly)


fig, axs = plt.subplots(
  nrows=2,
  ncols=5,
  sharey=True,
  sharex=True,
  figsize=(18, 8)
)

for i, data in enumerate(Normal[:5]): 
    data = data.transpose(0, 1)      
    plot_prediction(data, neuralnet, title='Normal', ax=axs[0, i])

for i, data in enumerate(Anomaly[:5]):
    data = data.transpose(0, 1)
    plot_prediction(data, neuralnet, title='Anomaly', ax=axs[1, i])

fig.tight_layout();
plt.savefig('Normal and Anomaly.png',dpi=800)
plt.show()
plt.clf()

#%%
