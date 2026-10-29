#!/usr/bin/env python
# coding: utf-8

# In[1]:


#doen't work on older versions of cellpose. v0.7.2 doesn' run on gpu
import numpy as np
import time, os, sys
import scipy.io
#import os.path
from os import path

#import mxnet as mx
#import matplotlib.pyplot as plt
import glob
import sys

from skimage import io

from cellpose import models

# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto')


filelist = glob.glob('processed/MAX*/aligned/alignedn2vhyb01.tif')
filelist1 = glob.glob('processed/MAX*/aligned/alignedfixedn2vgeneseq01.tif')
folderlist=glob.glob('processed/MAX*/aligned/')
#len(filelist)


# In[2]:



for i in range(len(filelist)): 
    if i%10==0:
        print(i)
    if not path.exists(folderlist[i]+'cellmask.mat'):
        imgs=[]
        currfile=filelist[i]
        currfile1=filelist1[i]
        currfolder=folderlist[i]
     #   print(currfile)
        im=io.imread(currfile)
        im1=im[0:3,:,:].sum(axis = 0)
        im_1=io.imread(currfile1)
        im1_1=im_1[0:3,:,:].sum(axis = 0)
        im1=im1+im1_1
        im2=im[4,:,:]
    #    plt.figure()
    #    plt.imshow(im1)
        im3=np.concatenate((im1[:,:,None],im2[:,:,None]),axis=2)
        imgs.append(im3)
        # define CHANNELS to run segementation on
        # grayscale=0, R=1, G=2, B=3
        # channels = [cytoplasm, nucleus]
        # if NUCLEUS channel does not exist, set the second channel to 0
        channels = [1,2]


        masks, flows, styles, diams = model.eval(imgs, diameter=40, channels=channels)

        maski = masks[0]
        scipy.io.savemat(folderlist[i]+'cellmask.mat',dict(maski=maski))


# In[3]:




