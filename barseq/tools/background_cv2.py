#!/usr/bin/env python
import argparse
import logging
import os
import sys

import datetime as dt

import cv2
import numpy as np

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def background_cv2( infiles, outfiles, stage=None, cp=None):
    '''
    Subtracts background from select_channels of input images. 
    Remainder channels are added back unchanged. 

    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'background-geneseq'
            
    image_type = cp.get(stage,'image_type')
    radius = int(cp.get('cv2','radius'))
    output_dtype = cp.get( stage,'output_dtype')
    channel_names =  get_config_list(cp, image_type, 'channels')
    select_channels = get_config_list(cp, stage, 'channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    num_c = len(select_channels)

    logging.debug(f'output_dtype={output_dtype} radius = {radius} num_channels={num_c} select_channels = {select_channels}')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')        
        
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        I = read_image( infile)        
        I=I.copy()
        I_filtered=np.zeros_like(I)
        I_rem=I[num_c:,:,:]
        I=I[0:num_c,:,:]
        k=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (radius,radius))
        for i in range(len(I)):
            bck=cv2.morphologyEx(I[i,:,:], cv2.MORPH_OPEN, kernel = k)
            I_filtered[i,:,:] = I[i,:,:] - np.expand_dims(bck,0)
        
        I_filtered[num_c:,:,:]=I_rem    
        I_filtered=uint16m(I_filtered)

        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image( outfile, I_filtered, photometric = 'minisblack' )
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
        
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    background_cv2( infiles=args.infiles, 
                    outfiles=args.outfiles,
                    stage=args.stage,  
                    cp=cp )
    
    logging.info(f'done processing output to {outdir}') 
