import os
import torch
import numpy as np
from cellpose import models, io
from cellpose.io import imread
import glob
import argparse

# model_name pth dia_est in_name out_name



def get_folders_local(pth):
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
    parser.add_argument("--model-name",help="Cellpose model name",type=str,default='cyto3')
    parser.add_argument("--diameter",help="approximate diameter of the cell",type=int, default=40)
    # parser.add_argument("pth",help="path to your experiment folder",type=str,default='cell_inp2.tif')
    parser.add_argument("--outname",help="output file name",type=str,default='cell_mask_cyto3.tif')

    args=parser.parse_args()
    use_gpu = torch.cuda.is_available()
    
    in_name='cell_inp2.tif'
    out_name='cell_mask_cyto3_redo2.tif'
    io.logger_setup()
    model = models.Cellpose(model_type=args.model_name,gpu=use_gpu)
    [folders,pos,xp,yp]=get_folders_local(args.pth)
    channels = [[0,1]]
    for folder in folders:
        imgs = io.imread(os.path.join(args.pth,'processed',folder,'aligned',in_name))
        masks, flows, styles, diams = model.eval(imgs, diameter=args.diameter, channels=channels)
        io.imsave(os.path.join(args.pth,'processed',folder,'aligned',args.outname),masks)
    print('Cellpose finished')

if __name__ == "__main__":
    main()