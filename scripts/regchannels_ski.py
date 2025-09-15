#!/usr/bin/env python
#
#

import argparse
import logging
import os
import sys

import datetime as dt

import numpy as np
#import tifffile as tf
import skimage as ski

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def regchannels_ski( infiles, outfiles, stage=None, cp=None):
    '''
    
    def channel_alignment(I_filtered,fname,config_pth,pth,name,num_c,is_affine,writefile):
        I_filtered=I_filtered.copy()
        Ishifted=np.zeros_like(I_filtered)
        I_rem=I_filtered[num_c:,:,:]
        I_filtered=I_filtered[0:num_c,:,:]
        
        chshift=scipy.io.loadmat(os.path.join(config_pth,fname))['chshift20x']
        
        for i in range(chshift.shape[0]):
            if is_affine:
                tform=chshift[i] # refine this later on-ng
            else:
                tform=skimage.transform.SimilarityTransform(translation=-chshift[i,:]) # remember this takes -shifts-ng
                
            It=skimage.transform.warp(np.squeeze(I_filtered[i,:,:]),tform,preserve_range=True,output_shape=(I_filtered.shape[1],I_filtered.shape[2]))
            Ishifted[i,:,:]=np.expand_dims(It,0)
        Ishifted[num_c:,:,:]=I_rem    
        Ishifted=uint16m(Ishifted)
        if writefile:
            tfl.imwrite(os.path.join(pth,name),Ishifted,photometric='minisblack')
        return Ishifted

    Ibcksub_shifted=channel_alignment(Ibcksub,chshift_filename,config_pth,pth,'bck_sub'+filename,num_c,is_affine,0) 
        # last argument is to not to write intermediate file-ng CHANNEL ALIGN-NG
    
    '''
    
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'regchannels'
    
    logging.info(f'stage={stage}')
        
    microscope_profile = cp.get('experiment','microscope_profile')
    chshift_file = cp.get(microscope_profile,'channel_shift')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    chshift_path = os.path.join(resource_dir, chshift_file)
    is_affine = cp.getboolean(stage,'is_affine')
    
    logging.debug(f'chshift_path = {chshift_path} is_affine={is_affine}')

    chshift = load_df(chshift_path, as_array=True)
    n_channels = len(chshift)
    logging.debug(f'loaded channel shift. len={len(chshift)} ')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')
        
        (dirpath, base, ext) = split_path(os.path.abspath(infile))
                
        I = read_image(infile)
        I=I.copy()
        Ishifted=np.zeros_like(I)
        I_rem=I[n_channels:,:,:]
        I=I[0:n_channels,:,:]
        for i in range(chshift.shape[0]):
            if is_affine:
                # refine this later on-ng
                tform=chshift[i] 
            else:
                # remember this takes -shifts-ng  
                tform=ski.transform.SimilarityTransform(translation=-chshift[i,:])              
            It=ski.transform.warp(np.squeeze(I[i,:,:]), 
                                      tform, 
                                      preserve_range=True, 
                                      output_shape=(I.shape[1],I.shape[2]))
            Ishifted[i,:,:]=np.expand_dims(It,0)
        Ishifted[n_channels:,:,:]=I_rem    
        Ishifted=uint16m(Ishifted)
        
        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, Ishifted)
        logging.debug(f'done writing {outfile}')


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
         
    (outdir, file) = os.path.split(args.outfiles[0])
    logging.debug(f'ensuring outdir {outdir}')
    os.makedirs(outdir, exist_ok=True)
        
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    regchannels_ski( infiles=args.infiles, 
                     outfiles=args.outfiles,
                     stage=args.stage,  
                     cp=cp )
    logging.info(f'done processing output to {outdir}') 
