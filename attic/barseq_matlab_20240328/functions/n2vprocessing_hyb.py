#!/usr/bin/env python
# coding: utf-8

# In[1]:


from n2v.models import N2VConfig, N2V
import numpy as np
import os

import imageio
import glob
import tifffile as tf



# In[4]:


model_name = 'n2v_hyb_20230323'
# the base directory in which our model will live
basedir = 'c:\\barseq_envs\\n2vmodels'
# We are now creating our network model.
#model=N2V(config,model_name,basedir=basedir)
model1= N2V(config=None, name=model_name+'GFP', basedir=basedir)
model2= N2V(config=None, name=model_name+'YFP', basedir=basedir)
model3= N2V(config=None, name=model_name+'TxRed', basedir=basedir)
model4= N2V(config=None, name=model_name+'Cy5', basedir=basedir)

basefn=''

pos_folders=glob.glob('processed\\MAX*');
hybcycle=len(glob.glob(pos_folders[0]+'\\hyb*.tif'))


def grab_xiaoyin_data(fn):
    D=[]
    
    for i in range(1,1+hybcycle): # 4 hyb cycles
        D1=[]
        with imageio.get_reader(fn+'hyb%02d.tif'%i) as f:
            for c in range(6): #including all channel
                D1.append(f.get_data(index=c)[None])
            D.append(np.transpose(np.array(D1),(1,2,3,0)))

                    
    return D #np.transpose(D,(0,2,3,1))  # <-- Rounds x 2048 x 2048 x Channels



# In[5]:


folderlist=glob.glob('processed/MAX*/')
for folder in folderlist:
    imgs=grab_xiaoyin_data(folder)
    
    for i in range(len(imgs)): #cycles
        pred_img=[]
        pred_img.append(model1.predict(imgs[i][0,...,0],axes='YX'))
        pred_img.append(model2.predict(imgs[i][0,...,1],axes='YX'))
        pred_img.append(model3.predict(imgs[i][0,...,2],axes='YX'))
        pred_img.append(model4.predict(imgs[i][0,...,3],axes='YX'))
        pred_img.append(imgs[i][0,...,4]) #append non-seq channels at the end without prediction
        pred_img.append(imgs[i][0,...,5]) #append non-seq channels at the end without prediction
        tf.imwrite(folder+'n2vhyb%02d.tif'%(i+1),(pred_img[0]-pred_img[0].min()).astype('uint16'),append = False)
        for n in range(1, len(pred_img)):
            tf.imwrite(folder+'n2vhyb%02d.tif'%(i+1),(pred_img[n]-pred_img[n].min()).astype('uint16'),append = True)

