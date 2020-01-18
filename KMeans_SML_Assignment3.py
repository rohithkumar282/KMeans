#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import decimal
import pandas as pd
import math
import csv
import random
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('CSE575-HW03-Data.csv',header= None)
data = np.concatenate((dataset[0], dataset[1]))
data= data.reshape((256,1))
X1 = np.concatenate((data[:128],data[128:]),axis=1)
scaler = StandardScaler()
#z-score normalisation
X = scaler.fit_transform(X1)

def findclosestcentroids(X, centroids):
    m=X.shape[0]
    idx=np.zeros(m)
    for i in range (m):
        dist=np.sum(np.square(X[i,:]-centroids),axis=1)
        idx[i]=np.argmin(dist)
    return idx

def computecentroids(X,idx,K):
    m,n=X.shape
    centroids=np.zeros((K,n))
    for i in range (K):
        x=X[idx==i]
        if x.shape[0]>0:
            avg=np.mean(x,axis=0)
            centroids[i,:]=avg
            
    return centroids

def plotdatapts(X,idx,K):
    color=cm.rainbow(np.linspace(0,1,K))
    plt.scatter(X[:,0],X[:,1],c=color[idx.astype(int),:])
    
def plotprogresskmeans(X,previous,idx,K,i):
    plotdatapts(X,idx,K)
    plt.title('iteration number {}'.format(i+1))
    
def Kmeans(X,initial_centroids,max_iters,plot_progress):
    m,n=X.shape
    K=initial_centroids.shape[0]
    centroids=initial_centroids
    previous_centroids=np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
    idx=np.zeros(m)
    for i in range (max_iters):
        previous_centroids[i,:]=centroids
        idx=findclosestcentroids(X,centroids)
        if plot_progress:
            plt.figure()
            plotprogresskmeans(X,previous_centroids, idx, K, i)
            plt.show()
        centroids = computecentroids(X, idx, K)
    return centroids, idx

for K in range(2,10):
    print(' K = ', K)
    max_iters=10
    initial_centroids = X[np.random.choice(X.shape[0],K,replace = False)]
    centroids, idx = Kmeans(X, initial_centroids, max_iters, True)


# In[ ]:




