#!/usr/bin/env python
#
# take in all position-specific .tform_original.joblib -> global tforms_original.joblib
# new dict indexed by. position string (i.e. 'Pos1')
# { 
# 
# }
#
import argparse
import logging
import os
import re
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np

import skimage
import joblib
from natsort import natsorted as nsort

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def merge_stitch_ashlar_pd( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-stitch'

    # We know arity is single, so we can grab the single outfile
    # We also know this is an experiment-wide output, so it doesn't need tile info. 
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    outfile = os.path.join( outdir, 'tforms_original.joblib' )
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
       
    # get params
    transform_rescale_factor= cp.getfloat(stage, 'transform_rescale_factor' )
    trf_string = str(transform_rescale_factor).replace('.','p')
    logging.debug(f'transform_rescale_factor={transform_rescale_factor} trf_string={trf_string}')
    intensity_scaling=cp.getint(stage, 'intensity_scaling' )
    tilesize=cp.getint(stage, 'tilesize' )
    display_additional_rescale=cp.getfloat(stage, 'display_additional_rescale' )

    # infiles are MAX_Pos1_000_000.tform_original.joblib from /stitch
    # one file per <position>
    infile_names = [ os.path.split(ifn)[1] for ifn in infiles ]
    logging.debug(f'infile_names = {infile_names}') 
    
    # dict for full (multi-position) experiment?
    #   tforms_original.joblib 
    #   tforms_rescaled0p5.joblib
    # outputs to /merge/hyb/
    Texp={}
    for i, infile in enumerate(infiles):
        tilename = os.path.split(infile)[1]
        #tilename = os.path.splitext(tilename)[0]
        logging.info(f'handling {infile} tilename={tilename}')
        Tfull = joblib.load(infile)
        Texp[f'Pos{i+1}']=Tfull
        Tfull={}
    logging.debug(f'Aggregated {len(infiles)} Ashlar positions...')
    logging.info(f'Writing full aggregated output to {outfile}')
    joblib.dump(Texp, outfile)

    # Also do rescaling on all data and write to tforms_rescaled
    # tforms_rescaled.joblib
    sx=[]
    sy=[]   
    for position_id in nsort( list(Texp.keys())):
        logging.debug(f'rescaling position {position_id}')
        for tilename in nsort( list( Texp[position_id])):
            logging.debug(f'rescaling tilename {tilename}')
            Texp[position_id][tilename]['ref_pos'] = [ Texp[position_id][tilename]['position'][0] * transform_rescale_factor,
                                                       Texp[position_id][tilename]['position'][1] * transform_rescale_factor
                                                     ]
    
    outfile = os.path.join( outdir, f'tforms_rescaled{trf_string}.joblib' )
    logging.info(f'Writing full aggregated and rescaled output to {outfile}')    
    joblib.dump(Texp, outfile)

    # Possibly deal with RGB files later
    
    # Make global_tform_rescaled for each tile. 
    # Use original + rescaled data structure. 
    
    Tfinal = {}

    for pos_id in nsort( list( Texp.keys() ) ):
        tilenames = nsort( list( Texp[pos_id].keys() ) )
        for tilename in tilenames:
            ( base, ext ) = os.path.splitext(tilename) 
            tform=skimage.transform.SimilarityTransform( scale=transform_rescale_factor, 
                                                    translation= [Texp[pos_id][tilename]["ref_pos"][0], 
                                                                  Texp[pos_id][tilename]["ref_pos"][1]]
                                                    ) 
            outfile = os.path.join( outdir, f'{base}.global_tform_{trf_string}.joblib' )
            logging.info(f'Writing global_tform to {outfile}')    
            joblib.dump(tform, outfile )
            Tfinal[tilename] = tform
    
        outfile = os.path.join( outdir, f'{base}.global_tform_{trf_string}.joblib' )
        logging.info(f'Writing global_tform to {outfile}')
        # Switch to marking by position eventually.    
        # joblib.dump(Tfinal, os.path.join(outdir, f'{pos_id}.tforms_final.joblib'))
        joblib.dump(Tfinal, os.path.join(outdir, f'tforms_final.joblib'))
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

    merge_stitch_ashlar_pd( infiles=args.infiles, 
                            outfiles=args.outfiles, 
                             cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')




