#!/usr/bin/env python
#
# Apply transforms
# 
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


def aggregate_transform_np(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #             'tforms_final.joblib'
    #
    #. There may be more inputs that required, so only select relevant ones...
    # E.g.
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/all_segmentation.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/genehyb.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_original.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_rescaled0p5.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/geneseq/basecalls.joblib
    # 
    # lroi10x.joblib is main flag output. 

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-transform'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile
    # primary outfile is lroi10x.joblib
    #  
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
                  'seg' : 'all_segmentation.joblib',
                  'tforms' : 'tforms_final.joblib',
                  }

    (gene_rol_file, hyb_rol_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)
    tform_final =joblib.load(tforms_file)

    tilename_list = nsort( list(seg.keys() ))
    T={}
    for i, tilename in enumerate(tilename_list):
        logging.debug(f'handling {tilename}') 
        t={}
        tform=tform_final[tilename]        
        [x,y]=apply_transform(tform, gene_rol[tilename]['lroi_y'], gene_rol[tilename]['lroi_x'])
        t['lroi10x_x']=x
        t['lroi10x_y']=y
        [x,y]=apply_transform(tform, hyb_rol[tilename]['lroi_y'][0][0],hyb_rol[tilename]['lroi_x'][0][0]) 
        t['lroi10xhyb_x']=x
        t['lroi10xhyb_y']=y
        [x,y]=apply_transform(tform, seg[tilename]['cent_y'],seg[tilename]['cent_x']) 
        t['cellpos10x_x']=x
        t['cellpos10x_y']=y
        T[tilename]=t
    
    logging.info(f'Writing output to {outfile}')
    joblib.dump(T, outfile)
    logging.info(f'Done.')
    

def apply_transform(tform, coord_x, coord_y):
    """
    Global transformation function:
    1. Transforms the local coordinates of rolonies and cells to global downsized coordinates per tile
    """
    if len(coord_x):
        if not (isinstance(coord_x,list) or isinstance(coord_x,np.ndarray)):
            coord_x=coord_x.to_list()
            coord_y=coord_y.to_list()
        q=np.zeros([len(coord_x),2])
        q[:,0]=np.reshape(coord_x,(1,-1))
        q[:,1]=np.reshape(coord_y,(1,-1))
        v=tform(q)
        x=v[:,0]
        y=v[:,1]
    else:
        x=[]
        y=[]
    return x,y



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

    aggregate_transform_np( infiles=args.infiles, 
                            outfiles=args.outfiles,
                            stage=args.stage,  
                            cp=cp )
    logging.info(f'done processing output to {args.outfiles[0]}')
