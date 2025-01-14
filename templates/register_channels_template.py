#!/usr/bin/env python
#
#  PER-FILE -> PARALLEL OUTDIR
#  Script to register channels within a single image. 
#
#  channel-registered images in <outdir>/<base>.chreg.tif

import argparse
import logging
import os
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import tifffile as tf


def register_channels_native( infiles, outdir, image_type='geneseq', cp=None):
    '''
    Within each infile, register all channels to reference_idx channel. 
    image_type = [ geneseq | bcseq | hyb ]
  
    '''
    if cp is None:
        cp = get_default_config()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
    
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_types = cp.get('barseq','image_types').split(',')
    image_channels = cp.get(image_type, 'channels').split(',')
    channel_shift_file = cp.get('register_channels','channel_shift')
    channel_shift_path = f'{resource_dir}/{channel_shift_file}' 
    channel_shift = pd.read_csv(channel_shift_path, sep='\t', index_col=0, keep_default_na=False, comment="#")
    
    logging.debug(f'image_types={image_types} channels={image_channels}')        
    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]}')

    for filename in infiles:
        (dirpath, base, ext) = split_path(os.path.abspath(filename))
        logging.debug(f'handling {filename}')
        imgarray = imageio.imread(filename)
        
        pred_image = []
        for i, img in enumerate(imgarray):
            try:
                logging.debug(f'{base}.{ext}[{i}] shape={img.shape} dtype={img.dtype}')
                pimg = models[i].predict(img, axes='YX')
                logging.debug(f'got model output: {base}.{ext}[{i}] shape={pimg.shape} dtype={pimg.dtype}')
                pimg = pimg.astype(output_dtype)
                if do_min_subtraction:
                    pimg = pimg - pimg.min()  
                logging.debug(f'new dtype={pimg.dtype}')
                pred_image.append(pimg)
            except:
                logging.warning(f'ran out of models, appending channel [{i}] unchanged.')
                pred_image.append(img)
               
        logging.debug(f'done predicting {base}.{ext} {len(pred_image)} channels. ')
        outfile = f'{outdir}/{base}.denoised.{ext}'
        newimage = np.dstack(pred_image)
        # produces e.g. shape = ( 3200,3200,5)
        newimage = np.rollaxis(newimage, -1)
        # produces e.g. shape = ( 5, 3200, 3200)        
        tf.imwrite( outfile, newimage)
        logging.debug(f'done writing {outfile} ')



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
    
    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None, 
                    type=str, 
                    help='outdir. output base dir if not given.')
    
    parser.add_argument('infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 
       
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
      
    outdir = os.path.abspath('./')
    if args.outdir is not None:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    register_channels_native( infiles=args.infiles, 
                              outdir=outdir, 
                              cp=cp )
    
    logging.info(f'done processing output to {outdir}')

