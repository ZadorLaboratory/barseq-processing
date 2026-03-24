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

def merge_basecall_geneseq_pd( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-basecall-geneseq'

    # We know arity is single, so we can grab the single outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
       
    # get params
    
    #
    lroi_x=[] 
    lroi_y=[]
    gene_id=[]

    for infile in infiles:
        logging.info(f'handling {infile}...')
        try:
            T=pd.read_csv(infile, header=0)
            lroi_x.append(T.m2)
            lroi_y.append(T.m1)
            gene_id.append(T.j)
        except:
            lroi_x.append([])
            lroi_y.append([])
            gene_id.append([])
            logging.debug(f'No geneseq rolonies found for tile {infile}\n')

    logging.debug(f'Aggregated {len(infiles)} bardensrresults. Lengths: lroi_x={len(lroi_x)} lroi_y={len(lroi_y)} gene_id={len(gene_id)}')
    joblib.dump({"lroi_x":lroi_y,"lroi_y":lroi_x,"gene_id":gene_id}, outfile)
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

    merge_basecall_geneseq_pd( infiles=args.infiles, 
                       outfiles=args.outfiles, 
                       cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')