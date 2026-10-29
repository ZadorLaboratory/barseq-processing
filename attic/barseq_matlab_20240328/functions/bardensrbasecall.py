#!/usr/bin/env python
# coding: utf-8

# In[1]:



#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import bardensr
import tensorflow as tf
import time
import numpy.random as npr
import matplotlib.pyplot as plt
import bardensr.plotting
import tqdm.notebook
import h5py
import scipy as sp
import scipy.io
import time
import imageio
import pandas as pd

import glob


#get_ipython().run_line_magic('matplotlib', 'inline')
basefn=''


# control slice indices and target fdr threshold
controlidx=range(25,35) #control slice numbers
fdrthresh=0.05 #set target fdr threshold
trim=160

# gene names
matfile = scipy.io.loadmat(basefn+'codebook.mat')
gene_names = np.array([str(a[0]) for a in (matfile['codebook'][:, 0])])

# codebook
codebook=scipy.io.loadmat(basefn+'codebookforbardensr.mat')['codebookbin1'].astype(float)
codebook=codebook.transpose([2,1,0])
R,C,J=codebook.shape

# flattened version, flattening over all 28 frames
codeflat=np.reshape(codebook,(R*C,J))

# get list of codes which are for error correction (ERror Correction Codes :) )
ercc_codes=np.array([x.startswith('unused') for x in gene_names])


def grab_xiaoyin_data(fn,trim):
#use this for basecalling the actual image,but also allows trimming. The trimming should be fairly small (usually starting at 5%) 
    D=[]
    for i in range(1,R+1):
        with imageio.get_reader(fn+'%02d.tif'%i) as f:
            for c in range(C):
                D.append(f.get_data(index=0)[c][None])
    D1=np.array(D)
    ydim=np.shape(D1)[2]
    xdim=np.shape(D1)[3]
    return D1[:,:,trim:-trim,trim:-trim]  # <-- 28 x 1 x 3200 x 3200


# In[5]:


def grab_xiaoyin_data_center(fn):
#use this for calculating threshold from the center of the images
    cropfactor2=0.4 #this should be <0.5
    D=[]
    for i in range(1,R+1):
        with imageio.get_reader(fn+'%02d.tif'%i) as f:
            for c in range(C):
                D.append(f.get_data(index=0)[c][None])
    D1=np.array(D)
    ydim=np.shape(D1)[2]
    xdim=np.shape(D1)[3]
    return D1[:,:,round(ydim*cropfactor2):round(ydim*(1-cropfactor2)),round(xdim*cropfactor2):round(xdim*(1-cropfactor2))]  # <-- 28 x 1 x 3200 x 3200


# In[6]:


folderlist=glob.glob('processed/MAX*/aligned/')
if len(folderlist)>=36:
    folderlistcontrol=[folderlist[i] for i in controlidx]
else:
    folderlistcontrol=folderlist


# # get normalizing constants (using all fovs) and perform normalization
# 

# In[7]:


maxs=np.array([grab_xiaoyin_data_center(fn+'alignedfixedn2vgeneseq').max(axis=(1,2,3)) for fn in folderlistcontrol])
maxmax=np.median(maxs,axis=0) #max of each channel/cycle across fovs. Used to normalize all images. 


# # using normazing constants, compute evidence tensors, and get a threshold based on fdr estimated from unused barcodes

# In[8]:


noisefloor=.01

ercc_maxes=[]

evidence_tensors=[]
for fn in folderlistcontrol:
    et=bardensr.spot_calling.estimate_density_singleshot(
        grab_xiaoyin_data_center(fn+'alignedfixedn2vgeneseq')/maxmax[:,None,None,None],
        codeflat,
        noisefloor
    )  # <-- NOTE: we could save this et because we'll use it later
       # but computing et isn't actually much slower than loading it from disk... :)
    ercc_maxes.append(et[:,:,:,ercc_codes].max(axis=(0,1,2)))
ercc_maxes=np.array(ercc_maxes)#This is the max values of all unused barcodes in all fovs. 
thresh=np.median(np.median(ercc_maxes,axis=1)) # this is the initial thresh level for optimization


# In[9]:


thresh


# In[10]:


noisefloor=0.05
ercc_c_all=[]
total_c_all=[]
for fn in folderlistcontrol:
    et=bardensr.spot_calling.estimate_density_singleshot(
        grab_xiaoyin_data_center(fn+'alignedfixedn2vgeneseq')/maxmax[:,None,None,None],
        codeflat,
        noisefloor
    )

    #calculate fdr rate from unused barcodes
    for thresh1 in np.linspace(thresh-0.1,thresh+0.1,10):
        spots=bardensr.spot_calling.find_peaks(et,thresh1,use_tqdm_notebook=False)
        spots.to_csv(fn+'spots.csv',index=False)
        print(f"in {fn} found {len(spots)} spots")
        #calculate fdr rate from unused barcodes
        ercc_c=0;
        for ercc_idx in ercc_codes.nonzero()[0]:
            ercc_c=ercc_c+(spots.j==ercc_idx).to_numpy().sum()
        ercc_c_all.append(ercc_c)
        total_c_all.append(len(spots)-ercc_c)


# In[11]:



#calculate fdr        
ercc_c_all1=np.reshape(ercc_c_all,[len(folderlistcontrol),10])
total_c_all1=np.reshape(total_c_all,[len(folderlistcontrol),10])+1
fdr=ercc_c_all1/ercc_codes.sum()*(~ercc_codes).sum()/(total_c_all1)
fdrmean=ercc_c_all1.mean(axis=0)/ercc_codes.sum()*(~ercc_codes).sum()/(total_c_all1.mean(axis=0))

thresh_refined=np.linspace(thresh-0.1,thresh+0.1,10)[(fdrmean<fdrthresh).nonzero()[0][0]]#this is the new threshold optimized by targeted fdr value
thresh_refined
with open('thresh_refined.txt','w') as f:
    f.write(str(thresh_refined))



# # spot call each fov, using the thresholds we decided on, and the normalization we decided on

# In[14]:



for fn in folderlist:
    et=bardensr.spot_calling.estimate_density_singleshot(
        grab_xiaoyin_data(fn+'alignedfixedn2vgeneseq',trim)/maxmax[:,None,None,None],
        codeflat,
        noisefloor
    )

    spots=bardensr.spot_calling.find_peaks(et,thresh_refined,use_tqdm_notebook=False)
    spots.loc[:,'m1']=spots.loc[:,'m1']+trim
    spots.loc[:,'m2']=spots.loc[:,'m2']+trim
    spots.to_csv(fn+'bardensrresult.csv',index=False)
    print(f"in {fn} found {len(spots)} spots")
    

