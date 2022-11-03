import numpy as np
#import gzip, cPickle
#from tsne import bh_sne
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.random.seed(1974)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
#import plotly.express as px
import matplotlib


def TSNEplot(z_run,test_labels,graph_name,graph_title,perplexity):
    
    output = z_run
    #array of latent space, features fed rowise
    
    target = test_labels
    #groundtruth variable
    
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ',perplexity)
    

    group=target
    group = np.ravel(group)
    
    
    RS=np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
    np.save('tsne_3d.npy',tsne_fit)
    tsne_fit = np.load('tsne_3d.npy')
    np.save('target.npy',group)
    group = np.load('target.npy')
    
    
    df2 = pd.DataFrame(group) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'Balling')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'LoF')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Nopores')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'Keyhole')
    group = pd.DataFrame(df2)
    group = group.to_numpy()
    group = np.ravel(group)
    
    
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    x3=tsne_fit[:, 2]
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=group))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    uniq=["Balling","LoF","Nopores","Keyhole"]
    
    z = range(1,len(uniq))
    hot = plt.get_cmap('hsv')
    
    
    
    fig = plt.figure(figsize=(12,9), dpi=100)
    
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=115)#115
    
    
    #ax.legend(markerscale=1)
    # Plot each species
    marker= ["*",">","X","o"]
    color = [ "#fbab17","#0515bf","#10a310","#e9150d"]
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    #ax.legend(markerscale=15)
    ax.set_ylim(min(x2), max(x2))
    ax.set_zlim(min(x3), max(x3))
    ax.set_xlim(min(x1), max(x1))
    
    for i in range(len(uniq)):
        
        indx = (df['label']) == uniq[i]
        
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        #plt.scatter(x1[indx],x2[indx],x3[indx] ,color=scalarMap.to_rgba(i),label=uniq[i],marker=marker[i])
        ax.plot(a, b, c ,color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=8)
        #plt.scatter(a, b, c ,color=color[i],label=uniq[i],marker=marker[i])
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=25)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=25)
    ax.set_zlabel('Dimension 3',labelpad=20,fontsize=25)
    plt.title(graph_title,fontsize = 30)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    #plt.zticks(fontsize = 25)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name, bbox_inches='tight',dpi=100)
    plt.show()
    
    return ax,fig

def plot_embeddings(tsne_fit, targets,graph_name_2D,graph_title, xlim=None, ylim=None):
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    
    group=targets
    
    df2 = pd.DataFrame(group) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'Balling')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'LoF')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Nopores')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'Keyhole')
    group = pd.DataFrame(df2)
    group = group.to_numpy()
    group = np.ravel(group)
    
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    uniq=["Balling","LoF","Nopores","Keyhole"]
    
    z = range(1,len(uniq))
       
    
    fig = plt.figure(figsize=(12,9), dpi=100)
    
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    
    marker= ["*",">","X","o"]
    color = [ "#fbab17","#0515bf","#10a310","#e9150d"]
    
    
    for i in range(len(uniq)):
        indx = (df['label']) == uniq[i]
        # print(indx)
        a=x1[indx]
        b=x2[indx]
        
        plt.plot(a, b, color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=8)
        
        a=x1[indx]
        b=x2[indx]
        
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=25)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=25)
    plt.title(graph_title,fontsize = 30)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    #plt.zticks(fontsize = 25)
    # plt.legend(loc='upper left',frameon=False)
    plt.legend(uniq,bbox_to_anchor=(1.32, 1.05))
    plt.savefig(graph_name_2D, bbox_inches='tight',dpi=100)
    plt.show()

