#!/usr/bin/env python
#
# Aggregate geneseq data and create expression matrix. 
# 
import argparse
import joblib
import logging
import math
import os
import re
import pprint
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from barseq.utils import *
from barseq.imageutils import *

def aggregate_py(infiles, outfiles, stage=None, cp=None):
    #   cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #   inputs:  
    #         rolonies-bcseq.joblib
    #
    #   outputs:
    #         matrix_bcseq.tsv
    #         matrix_bcseq.joblib
    #
    #   We  move to joblibs being secondary outputs...
    #

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-bcseq'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know infiles are single, so grab first
    infile = infiles[0]

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    # resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))

    rolonies=joblib.load(infile)
    df = make_dict_dataframe(rolonies)    
    #slice_ids = natsort( list( df['slice_id'].unique()) )
    # for slice_id in slice_ids:
    #
    # cellids are already globally unique, even across slices
    # so we don't need to subdivide.
    #
    # sdf = df[  df['slice_id'] == slice_id].reset_index() 
    sdf = df.drop(['pos40x_x','pos40x_y', 'slice_id'], axis=1 )
    # gbdf = sdf.groupby( [ 'cell_id','sequence'], observed=True).size().reset_index(name='count')
    expmat = sdf.groupby( [ 'cell_id','sequence'], observed=True).size().reset_index(name='count')
    #expmat = gbdf.pivot(index='cell_id', columns='sequence',values='count') 
    #expmat.fillna(0.0, inplace=True) 
    #expmat = expmat.astype(int)
    logging.info('Writing out to {outfile}...')
    expmat.to_csv(outfile, sep='\t')

    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(outdir, f'{base}.joblib')
    matrix_out = {}
    matrix_out['expmat'] =  expmat.to_numpy()
    # matrix_out['genes'] =  list(expmat.columns)
    matrix_out['cell_id'] = list(expmat.index)

    logging.info(f'Writing data to joblib: {outfile} ')
    joblib.dump( matrix_out, outfile,)
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

    aggregate_py( infiles=args.infiles, 
                          outfiles=args.outfiles,
                          stage=args.stage,  
                          cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')
