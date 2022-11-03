from vrae import VRAE
from utils import *
from Material_processing import *
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch import optim, cuda
from tSNE import *
#%%


dload = './model_VAE_dir' #download directory
torch.cuda.empty_cache()
#%%

hidden_size = 27
hidden_layer_depth = 1
latent_length = 27
batch_size = 100
learning_rate = 0.001 #0.005
n_epochs = 300
dropout_rate = 0.1
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=5
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss

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

df=MaterialTsne(Material1, windowsize)
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
#Plot moving average visualisation

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

Material_1 = Normal_Regime(Material1, classname,windowsize) #Select the normal regimes
Material_1_database,Material_1_labels=dataprocessing (Material_1) #Data and labels split
Material_1_train, sequence_length, number_of_features = create_dataset(Material_1_database) #Tensor

#%% Model training


vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss,
            dload = dload)


      
train_loss, _=vrae.fit(Material_1_train)
PATH = 'vrae_VAE.pth'
vrae.save(PATH)

#%%

fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax.plot(train_loss,color='red')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['VAE training loss'])
plt.title('Training loss')
plt.savefig('Training loss.png',dpi=800)
plt.show();
plt.clf()


#%% Threshold and Reconstruction Error distribution 

dataset=MaterialTemplate(Material1,windowsize)

losses_three=Resconstruction_distribution(dataset,class_3,vrae)
scores_normal = np.asarray(losses_three)
normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
Threshold = normal_avg + (normal_std * 3)

print('Threshold:',Threshold)

losses_three=classabnormaliy(dataset,class_3,vrae,Threshold,color="#10a310")
losses_one=classabnormaliy(dataset,class_1,vrae,Threshold,color="#fbab17")
losses_two=classabnormaliy(dataset,class_2,vrae,Threshold,color="#0515bf")
losses_four=classabnormaliy(dataset,class_4,vrae,Threshold,color="#e9150d")


#%%
Normal = Normal_Regime(Material1, classname,windowsize)
Normal,Normal_labels=dataprocessing (Normal)
Normal, sequence_length, number_of_features = create_dataset(Normal)



Anomaly=MaterialTemplate(Material1,windowsize)
Anomaly = Anomaly.sample(frac=1.0)
class_name = str(Material1)+'_'+classname 
Anomaly = Anomaly[Anomaly.target != str(class_name)]
Anomaly, Anomaly_labels=dataprocessing (Anomaly)
Anomaly, sequence_length, number_of_features = create_dataset(Anomaly)


fig, axs = plt.subplots(
  nrows=2,
  ncols=5,
  sharey=True,
  sharex=True,
  figsize=(18, 8)
)

for i, data in enumerate(Normal[:5]): 
    plot_prediction(data, vrae, title='Normal', ax=axs[0, i])

for i, data in enumerate(Anomaly[:5]):
    plot_prediction(data, vrae, title='Anomaly', ax=axs[1, i])

fig.tight_layout();
plt.savefig('Normal and Anomaly.png',dpi=800)
plt.show()
plt.clf()

#%%

df3,one=boxplotsupport(losses_one,'Balling')
df4,two=boxplotsupport(losses_two,'LoF')
df5,three=boxplotsupport(losses_three,'Nopores')
df6,four=boxplotsupport(losses_four,'Keyhole')

categories=np.concatenate((df3,df4,df5,df6), axis=0)
classvalue=np.concatenate((one,two,three,four), axis=0)
categories = pd.DataFrame(categories)
classvalue = pd.DataFrame(classvalue)

boxplot_data = [classvalue, categories]
boxplot_data = pd.concat(boxplot_data,axis=1)

boxplot_data.columns = ['losses', 'classes']
plt.figure()
sns.boxplot(x = 'classes', y = 'losses', data = boxplot_data,linewidth=0.5,palette=["#fbab17", "#0515bf", "#10a310", "#e9150d"])
plt.savefig('losses_boxplot.png',bbox_inches='tight',pad_inches=0.1,dpi=200)
plt.show()
#%%

count_parameters(vrae)
#786736

# %%

Dataset_Tsne=MaterialTsne(Material1,windowsize)
Dataset_Tsne,Tsne_labels=dataprocessing (Dataset_Tsne)
Tsne_labels=Tsne_labels.to_numpy()
Dataset_Tsne, sequence_length, number_of_features = create_dataset(Dataset_Tsne)
Dataset_Tsne = vrae.transform(Dataset_Tsne)
Dataset_Tsne= Dataset_Tsne.reshape((Dataset_Tsne.shape[0], -1))



perp=40
graph_name= '3D_tsne'+'.png'
ax,fig=TSNEplot(Dataset_Tsne,Tsne_labels,graph_name,str('3D_tsne'),perp)
graph_name= 'tsne'+'.gif'

#%%
def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))

#%%

tsne_fit = np.load('tsne_3d.npy')
group = np.load('target.npy')

graph_name_2D='Tsne_Feature_2D' +'_'+str(perp)+'.png'
graph_title = "Feature space distribution"
plot_embeddings(tsne_fit, group,graph_name_2D,graph_title)