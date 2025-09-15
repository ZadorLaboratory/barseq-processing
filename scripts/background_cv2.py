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
    image_type = [ geneseq | bcseq | hyb ]

    is_affine=0
    num_initial_c=5 # initial channels in first round of sequencing to read
    num_later_c=4 # subsequent cycle channels to read
    num_c=4 # channels to perform bleedthrough-correction and channel alignment, 
    needs more elegant solution--ng
    subsample_rate=4
    do_coarse=0
    resize_factor=2
    block_size=128
    chshift_filename='chshift20x-20220218.mat'
    chprofile_filename='chprofile20x-50-30-20-40-20220218.mat'
    radius=31
    local=1

    def back_sub_opencv_open(I,radius,pth,name,num_c,writefile):
        I=I.copy()
        I_filtered=np.zeros_like(I)
        I_rem=I[num_c:,:,:]
        I=I[0:num_c,:,:]
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
        for i in range(len(I)):
            bck=cv2.morphologyEx(I[i,:,:], cv2.MORPH_OPEN, kernel= k)
            I_filtered[i,:,:]=I[i,:,:]-np.expand_dims(bck,0)
        
        I_filtered[num_c:,:,:]=I_rem    
        I_filtered=uint16m(I_filtered)
        if writefile:
            tfl.imwrite(os.path.join(pth,name),I_filtered,photometric='minisblack')
        return I_filtered
        
    I=tfl.imread(os.path.join(pth,'n2vgeneseq02.tif'),key=range(0,4,1))
    Ifilt=back_sub_opencv_open(I,31,pth,'bcksb.tif',4,0)

    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'background'
            
    image_types = cp.get('barseq','image_types').split(',')
    radius = int(cp.get('cv2','radius'))
    output_dtype = cp.get( stage,'output_dtype')
    num_channels = 4

    #logging.debug(f'image_types={image_types} channels={image_channels}')
    logging.debug(f'output_dtype={output_dtype} radius = {radius} num_channels={num_channels}')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')        
        
        (dirpath, base, ext) = split_path(os.path.abspath(infile))

        I = read_image( infile)        
        I=I.copy()
        I_filtered=np.zeros_like(I)
        I_rem=I[num_channels:,:,:]
        I=I[0:num_channels,:,:]
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
        for i in range(len(I)):
            bck=cv2.morphologyEx(I[i,:,:], cv2.MORPH_OPEN, kernel = k)
            I_filtered[i,:,:]=I[i,:,:]-np.expand_dims(bck,0)
        
        I_filtered[num_channels:,:,:]=I_rem    
        I_filtered=uint16m(I_filtered)

        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image( outfile, I_filtered )
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

    background_cv2( infiles=args.infiles, 
                    outfiles=args.outfiles,
                    stage=args.stage,  
                    cp=cp )
    
    logging.info(f'done processing output to {outdir}') 
