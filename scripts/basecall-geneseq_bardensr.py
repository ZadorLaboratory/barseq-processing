#!/usr/bin/env python
#
# Do basecalling on batches of images.
# Intrinsically consumes multiple cycles, to output file is single for multiple
# inputs. So --outfile is only arg. 
#

import argparse
import joblib
import json
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

import numpy as np

import bardensr
import bardensr.plotting

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def basecall_bardensr( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    load codebook, 
    run bardensr, 
    output evidence tensor dataframe to <outdir>/<mode>/<prefix>.brdnsr.tsv   
    arity is single. 
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-geneseq'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    
    intensity_thresh = None
    (subdir, base, current_label, current_ext) = parse_rpath(outfile)
    param_file = os.path.join(subdir, f'bardensrparams.json')
    if os.path.exists(param_file):
        with open(param_file, 'r' ) as f:
            data = json.load(f)
            intensity_thresh = float( data['intensity_thresh_refined'] )
            noisefloor_final = float( data['noisefloor_final'])
            trim = int( data['trim'])
            cropf = float( data['cropf'])
            median_max_list = data['median_max']
            median_max_list = [ np.float64(x) for x in median_max_list ]
            median_max = np.array(median_max_list)
            logging.debug(f'type(median_max) = {type(median_max)} type(median_max[0])= {type(median_max[0])} ')
            logging.info(f'Successfully loaded intensity_thresh = {intensity_thresh} median_max = {median_max}')
    else:
        logging.warning(f'param_file={param_file} does not exist. Exitting.')
        sys.exit(1)
    logging.debug(f'noisefloor_final={noisefloor_final} trim={trim} cropf={cropf}')

    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = get_config_list(cp, stage, 'codebook_bases')
    cbfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cbfile}')
    codebook_df = load_codebook_file(cbfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook_df} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles)
    logging.info(f'Detected tilesets of {n_cycles} cycles.')
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook_df, 
                                                                        codebook_bases, 
                                                                        n_cycles=n_cycles)
    logging.info(f'R={R} C={C} J={J} codeflat.shape={codeflat.shape} len(genes)={len(genes)} pos_unused_codes={pos_unused_codes}')

    img_stack_trimmed = bd_read_images(infiles, R, C, trim=trim )
    logging.debug(f'img_stack_trimmed.shape = {img_stack_trimmed.shape} img_stack_trimmed.sum() = {img_stack_trimmed.sum()}')
    img_norm = img_stack_trimmed / median_max[ :, None, None, None]
    
    logging.debug(f'img_norm.shape={img_norm.shape} img_norm.sum()={img_norm.sum()} codeflat={codeflat} noisefloor_final={noisefloor_final}')
    et = bardensr.spot_calling.estimate_density_singleshot( img_norm, 
                                                            codeflat, 
                                                            noisefloor_final)
    logging.debug(f'estimated_density sum = {et.sum()} intensity_thresh={intensity_thresh}')
    
    spots = bardensr.spot_calling.find_peaks( et, intensity_thresh, use_tqdm_notebook=False)
    spots.loc[:,'m1'] = spots.loc[:,'m1'] + trim
    spots.loc[:,'m2'] = spots.loc[:,'m2'] + trim

    (odirpath, obase, olabel, oext) = split_path(outfile)
    logging.info(f'In {obase} found {len(spots)} spots.')            
    spots.to_csv(outfile, index=False)   
    logging.debug(f'wrote spots to outfile={outfile}')

    of = os.path.join(odirpath, f'{obase}.codeflat.joblib')
    joblib.dump(codeflat, of)
 

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
                    help='stage to use as template')
    
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

    basecall_bardensr( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

