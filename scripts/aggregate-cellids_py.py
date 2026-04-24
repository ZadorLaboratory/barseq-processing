#!/usr/bin/env python
#
# Aggregate and add cell_ids
# used for hyb
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

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *


def aggregate_cellids_py(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #. There may be more inputs that required, so only select relevant ones...
    # E.g.
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/all_segmentation.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/genehyb.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_original.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_rescaled0p5.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/geneseq/basecalls.joblib
    # 

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-cellids'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    #   'basecalls.joblib'.  'all_segmentation.joblib'   'genehyb.joblib'
    #
    # return order will be alphabetical
    #
    input_map = { 'gene_rol' : 'basecalls.joblib',
                  'hyb_rol' :  'genehyb.joblib',
                  'seg' : 'all_segmentation.joblib'
                  }

    (gene_rol_file, hyb_rol_file, seg_file) = select_input_files(infiles, input_map)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)

    T={}
    tilename_list = nsort( list(seg.keys()) )
    for i, tilename in enumerate( tilename_list) :
        logging.debug(f'handling {tilename}') 
        t={}
        mask=seg[tilename]['dilated_labels']
        coord_xg=gene_rol[tilename]['lroi_x']
        coord_yg=gene_rol[tilename]['lroi_y']
        coord_xh=hyb_rol[tilename]['lroi_x'][0][0]
        coord_yh=hyb_rol[tilename]['lroi_y'][0][0]
        t['cellid']= assign_rolony_to_cell(mask, coord_xg, coord_yg)
        t['cellidhyb']= assign_rolony_to_cell(mask, coord_xh, coord_yh)
        T[tilename]=t
    joblib.dump(T,os.path.join(outfile))
    logging.info(f'Done.')

def assign_rolony_to_cell(mask, coord_x, coord_y):
    """
    Global transformation function:
    1. Calls get_cellid function if there are rolonies detected in this tile or else assigns empty cell id to this tile
    """
    #logging.debug(f'handling coord_x = {coord_x}, coord_y={coord_y}')
    if len(coord_x):
        cell_id=get_cellid(mask, coord_x, coord_y)
    else:
        cell_id=[] # earlier this was [] and was causing error later
    return cell_id

def get_cellid(mask, coord_x, coord_y):
    """
    Global transformation function:
    1. For any detected rolony-assigns it to a cell
    2. Returns the cell ids for all rolonies in this tile
    """
    #logging.debug(f'handling coord_x = {coord_x}, coord_y={coord_y}')
    coord_xl=[int(np.round(x)) for x in coord_x]
    coord_yl=[int(np.round(x)) for x in coord_y]
    cell_id=mask[coord_xl,coord_yl]
    return cell_id

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

    aggregate_cellids_py( infiles=args.infiles, 
                          outfiles=args.outfiles,
                          stage=args.stage,  
                          cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')
