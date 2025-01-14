#!/usr/bin/env python
#
#  PER-FILE -> PARALLEL OUTDIR
#
# Script template to do denoising. 
# Input:     set of image files   <base>.tif
#            each image has N channels. 
# Output:   denoised images in <outdir>/<base>.denoised.tif
#
#

import argparse
import logging
import os
import sys

import datetime as dt
from configparser import ConfigParser

import imageio
import tifffile as tf

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.core import *
from barseq.utils import *

def denoise_tool( infiles, outdir, image_type='barcode', cp=None):
    '''
    image_type = [ geneseq | bcseq | hyb ] ...
    All input and output should be from-to the same directory. 
    Caller handles  directory/experiment level organization.
    
    '''
    if cp is None:
        cp = get_default_config()
    
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))


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

    denoise_n2v( infiles=args.infiles, 
                 outdir=outdir, 
                 cp=cp )
    
    logging.info(f'done processing output to {outdir}')