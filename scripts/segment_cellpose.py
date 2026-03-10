#!/usr/bin/env python
#
# Use Cellpose to segemnt cells. 
#
# current inputs. 
#         hyb:  5 channels. 
#             hyb.  channel 3 (all-genes)
#             hyb.  channel 5. DAPI
#         geneseq:
#             sum(all_channels) from either geneseq01 or all geneseq* 
#
import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
from joblib import load, dump

#import torch
import numpy as np

#from cellpose import models, io
#from cellpose.io import imread

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
 
#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def segment_cellpose( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    run cellpose
      
    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'segment_cellpose'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    cellpose_input_image = prepare_cellpose_input(infiles, outfiles )
    logging.debug(f'got cellpose input image shape={cellpose_input_image.shape}')

    logging.info(f'writing to {outfile}')
    write_image(outfile, cellpose_input_image)
    logging.debug(f'done writing {outfile}')    


def prepare_cellpose_input(infiles, outfiles):
    '''
                           nuc_ch=5,
                           num_chyb=5,
                           num_cgene=4,
                           other_channels = list(range(0,num_chyb))
    '''
    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')    

    (dirpath, base, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    suboutdir = os.path.join(outdir, subdir)
    os.makedirs(suboutdir, exist_ok=True)
    outfile = os.path.join( outdir, subdir, f'{base}.cp_inp.tif' )
    logging.debug(f'preparing cellpose input to be written to {outfile}')

    hyb_image = read_image(infiles[0], channels=[ 0, 1 , 2 , 3 , 4 ])
    cp_input_image = np.zeros( [2, hyb_image.shape[1], hyb_image.shape[2]] )
    gene_composite = np.zeros( [ hyb_image.shape[1],hyb_image.shape[2] ] )
    for infile in infiles[1:]:
        gene_image = read_image(infile, channels=[0, 1, 2, 3] )
        gene_composite = gene_composite + np.sum( gene_image, axis=0 )
    nuclear_image = hyb_image[4]
    cyto_image = np.sum( hyb_image[0:3], axis=0 ) + gene_composite 
    cp_input_image[0,:,:]=uint16m(cyto_image)
    cp_input_image[1,:,:]=uint16m(nuclear_image)
    logging.debug(f'made cellpose input image. shape={cp_input_image.shape}')
    logging.debug(f'writing intermediate cellpose input to {outfile} ...')
    write_image(outfile, cp_input_image)
    logging.debug(f'returning intermediate image...')
    return cp_input_image


#
# CODE FROM NOTEBOOK
#
def top_level():
    prepare_cellpose_input(pth,
                           nuc_ch,
                           num_chyb,
                           num_cgene,
                           tilesize)
    run_diff_env_scripts("cellpose_segmentation", 
                         config_file, 
                         pth=pth, 
                         diameter=diameter, 
                         outname=outname, 
                         model_name=model_name)
    import_cellpose_all_tiles(pth,dilation_radius,outname)

# model_name pth dia_est in_name out_name

def prepare_cellpose_input_ndg(pth,
                           nuc_ch=5,
                           num_chyb=5,
                           num_cgene=4,
                           tilesize=3200):
    """
    Cell segmentation function:
    1. creates an input image stack for cellpose input per tile with nuclear  
    information from DAPI (hyb) and gene from rest of the hyb and all geneseqs
    """
    [folders,pos,x,y]=get_folders(pth)
    other_ch=list(range(0,num_chyb))
    del other_ch[nuc_ch-1]
    cell_inp=np.zeros([2, tilesize, tilesize])
    for folder in folders:
        Ih=tfl.imread(os.path.join(pth,'processed',folder,'aligned','alignedn2vhyb01.tif'), key=range(0,num_chyb,1))
        Ig=np.zeros([Ih.shape[1],Ih.shape[2]])
        gene_cycles=sorted(glob.glob(os.path.join(pth,'processed',folder,'aligned','alignedfixedn2vgeneseq*.tif')))
        for gene_cycle in gene_cycles:
            Ig=Ig+np.sum(tfl.imread(gene_cycle, key=range(0,num_cgene,1)), axis=0)
        Inuc=Ih[nuc_ch-1,:,:]
        Icyto=np.sum(Ih[other_ch,:,:],axis=0)+Ig
        cell_inp[0,:,:]=uint16m(Icyto)
        cell_inp[1,:,:]=uint16m(Inuc)
        tfl.imwrite(os.path.join(pth,'processed',folder,'aligned','cell_inp2.tif'),uint16m(cell_inp),photometric='minisblack')

import subprocess,yaml
def run_diff_env_scripts(name,config_name, **params):
    cfg=yaml.safe_load(open(config_name))
    spec=cfg["tasks"][name]
    argv=[item.format(**params) for item in spec["argv"]]
    env=spec["env"].split(":",1)[1]
    cmd=["conda", "run", "-n", env, "python", *argv]
    subprocess.run(cmd, check=True)

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

def cellpose_runner_main():
    parser=argparse.ArgumentParser()
    parser.add_argument("pth",help="path to your experiment folder",type=str)
    parser.add_argument("--model-name",help="Cellpose model name",type=str,default='cyto3')
    parser.add_argument("--diameter",help="approximate diameter of the cell",type=int, default=40)
    parser.add_argument("--outname",help="output file name",type=str,default='cell_mask_cyto3.tif')

    args=parser.parse_args()
    use_gpu=torch.cuda.is_available()
    
    in_name='cell_inp2.tif'
    out_name='cell_mask_cyto3_redo2.tif'
    io.logger_setup()
    model=models.Cellpose(model_type=args.model_name,gpu=use_gpu)
    [folders,pos,xp,yp]=get_folders_local(args.pth)
    channels=[[0,1]]
    for folder in folders:
        imgs=io.imread(os.path.join(args.pth,'processed',folder,'aligned',in_name))
        masks,flows,styles,diams=model.eval(imgs,diameter=args.diameter,channels=channels)
        io.imsave(os.path.join(args.pth,'processed',folder,'aligned',args.outname),masks)
    print('Cellpose finished')
    


if __name__ == '__main__':
    FORMAT='%(asctime)s (UTC) [ %(levelname)s ] %(filename)s:%(lineno)d %(name)s.%(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-d', '--debug', 
                        action="store_true", 
                        dest='debug', 
                        help='debug logging')

    parser.add_argument('-v', '--verbose', 
                        action="store_true", 
                        dest='verbose', 
                        help='verbose logging')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')
    
    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')

    parser.add_argument('-t','--template', 
                    metavar='template',
                    default=None,
                    required=False, 
                    type=str, 
                    help='label for this stage config')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='outfile. ')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        loglevel = 'debug'
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   
        loglevel = 'info'
    
    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    logging.debug(f'Running with config={args.config}:\n{cdict}')
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    segment_cellpose( infiles=args.infiles, 
                      outfiles=args.outfiles,
                      stage=args.stage,  
                      cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

