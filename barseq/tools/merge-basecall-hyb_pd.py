#!/usr/bin/env python
#
# merges and bardensr data for all tiles in a position.
#
import argparse
import joblib
import logging
import os
import re
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np

from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops_table

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def merge_basecall_hyb_pd( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-basecall-hyb'

    # We know arity is single, so we can grab the single outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
           
    #
    tile_dict = {}
    for infile in infiles:
        (subdir, base, current_label, current_ext) = parse_rpath(infile)
        logging.info(f'handling {infile} base={base}')
        gho = joblib.load(infile)
        tile_dict[base] = gho

    logging.debug(f'Merged {len(infiles)} genehyb. ')
    joblib.dump(tile_dict, outfile)
    logging.info(f'Done.')

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

    merge_basecall_hyb_pd( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')