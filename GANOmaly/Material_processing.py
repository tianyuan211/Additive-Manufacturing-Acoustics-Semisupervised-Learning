import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from solver import training,testing,reconstruction


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Normal_Regime(Material_1, classname,windowsize):
    classfile_1 = str(Material_1)+'_classspace'+'_'+ str(windowsize)+'.npy'
    rawfile_1 = str(Material_1)+'_rawspace'+'_'+ str(windowsize)+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    df2 = pd.DataFrame(target_1) 
    df2.columns = ['Categorical']
    
    class_1 = 'Balling' 
    class_2 = 'LoF pores' 
    class_3 = 'Conduction mode' 
    class_4 = 'Keyhole pores' 
    
    df2=df2['Categorical'].replace(0,class_1)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,class_2)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,class_3)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,class_4)
    df2 = pd.DataFrame(df2) 
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target == str(Normal)]
    print(df_1.shape)
    
    return df_1

def Abnormal_Regime(Material_1, classname,windowsize):
    classfile_1 = str(Material_1)+'_classspace'+'_'+ str(windowsize)+'.npy'
    rawfile_1 = str(Material_1)+'_rawspace'+'_'+ str(windowsize)+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    df2 = pd.DataFrame(target_1) 
    df2.columns = ['Categorical']
    
    class_1 = 'Balling' 
    class_2 = 'LoF pores' 
    class_3 = 'Conduction mode' 
    class_4 = 'Keyhole pores' 
    
    df2=df2['Categorical'].replace(0,class_1)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,class_2)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,class_3)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,class_4)
    df2 = pd.DataFrame(df2) 
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target != str(Normal)]
    print(df_1.shape)
    
    return df_1


def MaterialTsne(Material_1, classname,windowsize):
    classfile_1 = str(Material_1)+'_classspace'+'_'+ str(windowsize)+'.npy'
    rawfile_1 = str(Material_1)+'_rawspace'+'_'+ str(windowsize)+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    
    
    df1 = pd.DataFrame(Features_1)  
    df1 = df1.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    df2 = pd.DataFrame(target_1) 
    df2.columns = ['Categorical']
    
    class_1 = 'Balling' 
    class_2 = 'LoF pores' 
    class_3 = 'Conduction mode' 
    class_4 = 'Keyhole pores'  
    
    df2=df2['Categorical'].replace(class_1,0)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(class_2,1)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(class_3,2)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(class_4,3)
    df2 = pd.DataFrame(df2) 
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    
    print(df_1.shape)
    
    return df_1


def MaterialTemplate(Material_1,windowsize):
    classfile_1 = str(Material_1)+'_classspace'+'_'+ str(windowsize)+'.npy'
    rawfile_1 = str(Material_1)+'_rawspace'+'_'+ str(windowsize)+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    df2 = pd.DataFrame(target_1) 
    df2.columns = ['Categorical']
    
    class_1 = 'Balling' 
    class_2 = 'LoF pores' 
    class_3 = 'Conduction mode' 
    class_4 = 'Keyhole pores' 
    
    df2=df2['Categorical'].replace(0,class_1)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,class_2)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,class_3)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,class_4)
    df2 = pd.DataFrame(df2) 
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    
    print(df_1.shape)
    
    return df_1

def dataprocessing (df):
    database = df
    labels=database.iloc[:,-1]
    database = database.drop(labels='target', axis=1)
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    return database,labels

def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

  
def classabnormaliy(df,class_name,neuralnet,Threshold,color):
    
    df = df[df.target == str(class_name)].drop(labels='target', axis=1)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    
    print(df.shape)
    dataset, seq_len, n_features = create_dataset(df)
    losses,_=testing(neuralnet=neuralnet, dataset=dataset)
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    sns.distplot(losses, bins=50,rug_kws={"color": "w"}, kde=True,color=color);
    graphname=str(class_name)+'_anomaly'+'.png'
    plt.title('loss distribution for '+str(class_name))
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1,dpi=200)
    plt.show()
    plt.clf()
    
    correct = sum(l > Threshold for l in losses)
    print(f'Correct {str(class_name)} predictions: {correct}/{len(dataset)}')
    return losses

def class_normaliy(df,class_name,neuralnet,Threshold,color):
    
    df = df[df.target == str(class_name)].drop(labels='target', axis=1)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    
    print(df.shape)
    dataset, seq_len, n_features = create_dataset(df)
    losses,_=testing(neuralnet=neuralnet, dataset=dataset)
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    sns.distplot(losses, bins=50,rug_kws={"color": "w"}, kde=True,color=color);
    graphname=str(class_name)+'_normal'+'.png'
    plt.title('loss distribution for '+str(class_name))
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1,dpi=200)
    plt.show()
    plt.clf()
    correct = sum(l < Threshold for l in losses)
    print(f'Correct {str(class_name)} predictions: {correct}/{len(dataset)}')
    return losses
        

def abnormaliy(df,class_name,neuralnet):
    
    df = df[df.target == str(class_name)].drop(labels='target', axis=1)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    print(df.shape)
    dataset, seq_len, n_features = create_dataset(df)
    losses,Threshold=testing(neuralnet=neuralnet, dataset=dataset)
    return losses,Threshold


def boxplotsupport(losses,classname):
    
    losses = np.asarray(losses)
    c=len(losses)
    filename = classname
    numbers = np.random.randn(c)
    df = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df=df.drop(['numbers'], axis=1)
    return df,losses