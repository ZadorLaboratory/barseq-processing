from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread, imwrite
from csbdeep.io import save_tiff_imagej_compatible
import glob
import os
import argparse

def call_n2v(pth,basedir,fname):
    if fname=='hyb':
        model_name0 = 'n2v_hyb_20230323GFP'
        model_name1 = 'n2v_hyb_20230323YFP'
        model_name2 = 'n2v_hyb_20230323TxRed'
        model_name3 = 'n2v_hyb_20230323Cy5'
    else:
        model_name0 = 'n2v_singleseqG'
        model_name1 = 'n2v_singleseqT'
        model_name2 = 'n2v_singleseqA'
        model_name3 = 'n2v_singleseqC'
    run_n2v(pth,fname,model_name0,model_name1,model_name2,model_name3,basedir)   


def run_n2v(pth,fname,model_name0,model_name1,model_name2,model_name3,basedir):
    model0 = N2V(config=None, name=model_name0, basedir=basedir)
    model1 = N2V(config=None, name=model_name1, basedir=basedir)
    model2 = N2V(config=None, name=model_name2, basedir=basedir)
    model3 = N2V(config=None, name=model_name3, basedir=basedir)
    folders,_,_,_=get_folders(pth)
    for folder in folders:
        filenames=sorted(glob.glob(os.path.join(pth,'processed',folder,fname+"*.tif")))
        #print(filenames,filenames[0].split('/')[-1])
        for filename in filenames:
            #print(f"Reading {folder} {filename.split('/')[-1]}")
            I=imread(filename)
            Ipred=I
            Ipred[0,:,:] = model0.predict(I[0,:,:], axes='YX')
            Ipred[1,:,:] = model1.predict(I[1,:,:], axes='YX')
            Ipred[2,:,:] = model2.predict(I[2,:,:], axes='YX')
            Ipred[3,:,:] = model3.predict(I[3,:,:], axes='YX')
            imwrite(os.path.join(pth,'processed',folder,'n2v'+filename.split('/')[-1]),uint16m(Ipred),photometric='minisblack')
        # I HAVE NOT SUBTRACTED MINIMUM PER CHANNEL FROM EACH CHANNEL BEFORE WRITING
    
        
    
def uint16m(x):
    """
    Preprocessing function:
    MATLAB equivalent of uint16-does clipping between data bounds and then converts
    """
    y=np.uint16(np.clip(np.round(x),0,65535))
    return y

def get_folders(pth):
    """
    Universal function:
    Scans the processed directory to get total tiles (folders),names, positions and x and y grid points 
    """
    dr=sorted(glob.glob(os.path.join(pth,'processed','MAX*')))
    folder=[]
    xp=[]
    yp=[]
    pos=[]
    for i,j in enumerate(dr):
        folder.append(str(j).split('/')[-1])
        temp=str(j).split('/')[-1].split('_')
        pos.append(temp[1])
        xp.append(int(temp[2])+1)
        yp.append(int(temp[3])+1)
    return folder,pos,xp,yp


def main(): 
    parser=argparse.ArgumentParser()
    parser.add_argument("pth",help="path to your experiment folder",type=str)
    parser.add_argument("fname",help="Modality name: geneseq  bcseq or hyb",type=str)
    parser.add_argument("--basedir",help="path to your trained models",type=str,default='/home/nrg/n2vmodels/')
    
    args=parser.parse_args()
    call_n2v(args.pth,args.basedir,args.fname)
    print('n2v finished')

if __name__ == "__main__":
    main()