#!/usr/bin/env python
#
# take in all position-specific .tform_original.joblib -> global tforms_original.joblib
# new dict indexed by. position string (i.e. 'Pos1')
# { 
# 
# }
#
import argparse
import logging
import os
import re
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np

from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops_table
from joblib import dump, load
from natsort import natsorted as nsort

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def merge_stitch_ashlar_pd( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-stitch'

    # We know arity is single, so we can grab the single outfile
    # We also know this is an experiment-wide output, so it doesn't need tile info. 
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    outfile = os.path.join( outdir, 'tforms_original.joblib' )
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
       
    # get params
    transform_rescale_factor= cp.getfloat(stage, 'transform_rescale_factor' )
    logging.debug(f'transform_rescale_factor={transform_rescale_factor}')
    intensity_scaling=cp.getint(stage, 'intensity_scaling' )
    tilesize=cp.getint(stage, 'tilesize' )
    display_additional_rescale=cp.getfloat(stage, 'display_additional_rescale' )

    #
    infile_names = [ os.path.split(ifn)[1] for ifn in infiles ]
    logging.debug(f'infile_names = {infile_names}') 
    
    Texp={}
    for i, infile in enumerate(infiles):
        tilename = os.path.split(infile)[1]
        logging.info(f'handling {infile} tilename={tilename}')
        Tfull = load(infile)
        Texp[f'Pos{i+1}']=Tfull
        Tfull={}

    logging.debug(f'Aggregated {len(infiles)} Ashlar positions...')
    logging.info(f'Writing full aggregated output to {outfile}')
    dump(Texp, outfile)

    # Also do rescaling on all data and write to tforms_rescaled
    sx=[]
    sy=[]   
    for position_id in nsort( list(Texp.keys())):
        logging.debug(f'rescaling position {position_id}')
        for tilename in nsort( list( Texp[position_id])):
            logging.debug(f'rescaling tilename {tilename}')
            Texp[position_id][tilename]['ref_pos'] = [ Texp[position_id][tilename]['position'][0] * transform_rescale_factor,
                                                       Texp[position_id][tilename]['position'][1] * transform_rescale_factor
                                                     ]
    trf_string = str(transform_rescale_factor).replace('.','p')
    outfile = os.path.join( outdir, f'tforms_rescaled{trf_string}.joblib' )
    logging.info(f'Writing full aggregated and rescaled output to {outfile}')    
    dump(Texp, outfile)








# NOTEBOOK CODE
def produce_tform_and_stitched_images(pth,sx,sy,intensity_scaling=3,tilesize=3200,transform_rescale_factor=0.5,display_additional_rescale=0.2):
    """
    Stitching function:
    1. For all tiles, save the rescaled (downscaled) transformation per tile, apply transformation on RGB of each tile, stitch all the tramsformed RGB into one big image and store it in check_registration folder
    2. Calls function to create a merged transform file for global position estimation
    """ 
    [folders,pos,x,y]=get_folders(pth)
    unique_pos=nsort(np.unique(pos))
    check_reg_folder=os.path.join(pth,'processed','checkregistration')
    os.makedirs(check_reg_folder,exist_ok=True)
    folder_names=np.array(folders)
    filenames=sorted(glob.glob(os.path.join(pth,'processed',folders[0],'RGB',"*"+".tif")))
    T=load(os.path.join(pth,'processed','tforms_rescaled'+str(transform_rescale_factor).replace('.','p')+'.joblib'))
    
    for f, filename in enumerate(filenames):
        filename=filename.split('/')[-1]
        I=np.zeros([int(np.max(sy)+tilesize*transform_rescale_factor),int(np.max(sx)+tilesize*transform_rescale_factor),3])
        #ax.imshow(I,cmap='jet',vmin=0,vmax=250)
        for n_pos in unique_pos:
            pos_id=np.array([i for i,name in enumerate(pos) if name==n_pos])
            for ids in pos_id:
                tilename=folder_names[ids]+'.tif'
                tform=skimage.transform.SimilarityTransform(scale=transform_rescale_factor, translation=[T[n_pos][tilename]["ref_pos"][0],T[n_pos][tilename]["ref_pos"][1]]) 
                dump(tform,os.path.join(pth,'processed',folder_names[ids],'global_tform_'+str(transform_rescale_factor).replace('.','p')+'.joblib'))
                tile=tfl.imread(os.path.join(pth,'processed',folder_names[ids],'RGB',filename))
                # if is_grayscale:
                #     It=skimage.transform.warp(np.squeeze(tile[stitching_channel-1,:,:]),tform.inverse,preserve_range=True,output_shape=(I.shape[0],I.shape[1]))
                # else:
                It=skimage.transform.warp(np.squeeze(tile),tform.inverse,preserve_range=True,output_shape=(I.shape[0],I.shape[1]))
                I=np.maximum(I,It)
   
        Irgb=np.uint8(np.clip(I*intensity_scaling,0,255))
        Irgb_rescaled = np.uint8(np.clip(rescale(Irgb, display_additional_rescale, channel_axis=-1, preserve_range=True, anti_aliasing=True),0,255))
        tfl.imwrite(os.path.join(check_reg_folder,n_pos+'_'+filename),Irgb_rescaled,photometric='rgb')
    merge_transforms(pth,str(transform_rescale_factor).replace('.','p'))

def merge_transforms(pth, name):
    """
    Stitching function:
    Merge all downscaled transforms per tile into a final transformation file
    """ 
    [folders,_,_,_]=get_folders(pth)
    T={}
    for folder in folders:
        T[folder]=load(os.path.join(pth,'processed',folder,'global_tform_'+name+'.joblib'))
    dump(T,os.path.join(pth,'processed','tforms_final.joblib'))


def merge_ashlar_results(pth,transform_rescale_factor=0.5,num_c=4):
    """
    Stitching function:
    1. ASHLAR based stitching results are encoded in a global dictionary 
    2. For each position (slice)-position of tiles and their names is saved as a sub-dictionary
    3. One final dictionary with positions as keys is stored as tforms_original file
    4. Calls function to rescale transformation
    """ 
    [folders,pos,_,_]=get_folders(pth)
    unique_pos=nsort(np.unique(pos))
    folder_names=np.array(folders)
    Texp={}
    Tfull={}
    for n_pos in unique_pos:
        T={}
        df=pd.read_csv(os.path.join(pth,'MAX_'+n_pos+'.positions.tsv'), sep='\t')
        pos_id=np.array([i for i,name in enumerate(pos) if name==n_pos])
        for ids in pos_id:
            tilename=folder_names[ids]+'.tif'
            T['position']=[df.iloc[ids,2],df.iloc[ids,1]]
            T['grid']=[0,0]
            Tfull[tilename]=T
            T={}
        Texp[n_pos]=Tfull
        Tfull={}
    dump(Texp,os.path.join(pth,'processed','tforms_original.joblib'))
    sx,sy=rescale_transformation(pth,folders,unique_pos,pos,transform_rescale_factor,num_c)
    return sx,sy

def merge_transforms(pth,name):
    """
    Stitching function:
    Merge all downscaled transforms per tile into a final transformation file
    """ 
    [folders,_,_,_]=get_folders(pth)
    T={}
    for folder in folders:
        T[folder]=load(os.path.join(pth,'processed',folder,'global_tform_'+name+'.joblib'))
    dump(T,os.path.join(pth,'processed','tforms_final.joblib'))

def rescale_transformation(pth,folders,unique_pos,pos,rescale_factor=0.5,num_c=4):
    """
    Stitching function:
    1. Reads the original transformation dictionary
    2. Downscales the coordinates as per rescale_factor and write in as new key-value pairs per tile in the original dictionary
    3. Writes the modified dictionary
    
    """ 

    folder_names=np.array(folders)
    T=load(os.path.join(pth,'processed','tforms_original.joblib'))
    sx=[]
    sy=[]
    for n_pos in unique_pos:
        pos_id=np.array([i for i,name in enumerate(pos) if name==n_pos])
        for ids in pos_id:
            tilename=folder_names[ids]+'.tif'
            T[n_pos][tilename]["ref_pos"]=[T[n_pos][tilename]["position"][0]*rescale_factor,T[n_pos][tilename]["position"][1]*rescale_factor]
            sx.append(T[n_pos][tilename]["ref_pos"][0])
            sy.append(T[n_pos][tilename]["ref_pos"][1])
        
    #pprint.pprint(T)
    dump(T,os.path.join(pth,'processed','tforms_rescaled'+str(rescale_factor).replace('.','p')+'.joblib'))
    return sx,sy






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
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='File[s] to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='Output file[s]. ') 
       
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

    merge_stitch_ashlar_pd( infiles=args.infiles, 
                            outfiles=args.outfiles, 
                             cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')




