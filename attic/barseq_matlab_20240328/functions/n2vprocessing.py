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


model_name = 'n2v_singleseq'
# the base directory in which our model will live
basedir = 'c:\\barseq_envs\\n2vmodels'
# We are now creating our network model.
#model=N2V(config,model_name,basedir=basedir)
model1= N2V(config=None, name=model_name+'G', basedir=basedir)
model2= N2V(config=None, name=model_name+'T', basedir=basedir)
model3= N2V(config=None, name=model_name+'A', basedir=basedir)
model4= N2V(config=None, name=model_name+'C', basedir=basedir)

basefn=''


pos_folders=glob.glob('processed\\MAX*');
genecycle=len(glob.glob(pos_folders[0]+'\\geneseq*.tif'))

#genecycle=7




def grab_xiaoyin_data(fn):
    D=[]
    
    for i in range(1,1+genecycle): # 7 cycles
        D1=[]
        with imageio.get_reader(fn+'geneseq%02d.tif'%i) as f:
            for c in range(4): #including DIC channel
                D1.append(f.get_data(index=c)[None])
            try: # try to append DIC channel, but it's fine if DIC is missing
                D1.append(f.get_data(index=4)[None])
            except:
                pass
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
        try:
            pred_img.append(imgs[i][0,...,4]) #append non-seq channels at the end without prediction
        except:
            pass
        #pred_img=model.predict(imgs[i][0,...],axes='YXC')
        #print(pred_img.shape)
        #for n in range(pred_img.shape[2]):
        #    tf.imwrite(folderlist[0]+'n2vBCseq%02d.tif'%i,pred_img[...,n]-pred_img[...,n].min(),append = True)
        tf.imwrite(folder+'n2vgeneseq%02d.tif'%(i+1),(pred_img[0]-pred_img[0].min()).astype('uint16'),append = False)
        for n in range(1, len(pred_img)):
            tf.imwrite(folder+'n2vgeneseq%02d.tif'%(i+1),(pred_img[n]-pred_img[n].min()).astype('uint16'),append = True)

