#!/usr/bin/env python
#
# Calculate required bardensr per-experiment image processing thresholds/parameters. 
# 
# 
import argparse
import itertools
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

import matplotlib.pylab as plt
import numpy as np

import bardensr
import bardensr.plotting

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def calc_params_bardensr( infiles, outfiles, stage=None, cp=None):
    '''
    take in all tileset files.
    alternate? take in all experiment files??
    calculate: 
    thresh_refined 
    noisefloor_final
    median_max
    
    fdrthresh=0.05,
    trim=160,
    cropf=0.4,
    noisefloor_ini=0.01,
    
    noisefloor_final=0.05    
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'calc-params'

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
    
    noisefloor_ini = cp.getfloat( stage, 'noisefloor_ini')
    noisefloor_final = cp.getfloat(stage, 'noisefloor_final')
    fdrthresh = cp.getfloat( stage, 'fdrthresh')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    logging.debug(f'noisefloor_ini={noisefloor_ini} trim={trim} cropf={cropf}')


    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = cp.get(stage, 'codebook_bases').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_codebook_file(cfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles)
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook, codebook_bases, n_cycles=n_cycles)

    # OUTPUT DICT
    param_outputs = {}

    # CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    logging.debug(f'calculating max_per_RC...')
    max_per_RC=[ bd_read_image_single(infile, R, C, cropf=cropf).max(axis=(1,2,3)) for infile in infiles ]
    
    # Expected to be 28 values. channels * cycles. 
    # first max(), then median of those max() per cycle. 
    s = pprint.pformat(max_per_RC, indent=4)
    logging.debug(f'max per RC = {s}')
    median_max=np.median(max_per_RC, axis=0)
    s = pprint.pformat(median_max, indent=4)
    logging.debug(f'median_max = {s}')

    # ESTABLISHING BASE THRESHOLD AT THE MEDIAN OF MAXIMUM ERROR READOUT
    err_max=[]
    evidence_tensors=[]
    for file in infiles:
        logging.debug(f'spot_calling.estimate_density_singleshot. file={file} R={R} C={C} trim={trim} noisefloor_ini = {noisefloor_ini}')
        trimmed = bd_read_image_single(file, R, C, trim=trim)
        img_norm = trimmed / median_max[:, None, None, None]
        et = bardensr.spot_calling.estimate_density_singleshot( img_norm , codeflat, noisefloor_ini )
        err_max.append( et[ :, :, :, pos_unused_codes].max(axis=(0,1,2)))
    err_max = np.array( err_max )
    thresh = np.median( np.median( err_max, axis=1))
    logging.info(f'intensity_thresh_ini={thresh}')

    # FIND OPTIMUM THRESHOLD WITH LOWEST FDR 
    err_c_all=[]
    total_c_all=[]
    for file in infiles:
        dirpath, base, label, ext = split_path( os.path.abspath(file))
        dirpath, subdir, label, ext = split_path( os.path.abspath(dirpath))
        logging.debug(f'handling image base={base}')
        cropped = bd_read_image_single(file, R, C, cropf=cropf)
        img_norm = cropped / median_max[:, None, None, None]
        et=bardensr.spot_calling.estimate_density_singleshot( img_norm , codeflat, noisefloor_final)
        for thresh1 in np.linspace( thresh-0.1, thresh+0.1, 10):
            spots = bardensr.spot_calling.find_peaks(et, thresh1, use_tqdm_notebook=False)
            #suboutdir = os.path.join( outfile_dir, 'bdparams', subdir)
            #os.makedirs(suboutdir, exist_ok=True)
            #logging.debug(f"found {len(spots)} spots in {file}")
            #outsub = os.path.join(suboutdir, f'{base}.{thresh1}.spots.csv')
            #logging.debug(f'writing spots to {outsub}')
            #spots.to_csv(outsub, index=False)
            
            err_c=0
            for err_idx in pos_unused_codes[0]:
                err_c=err_c + (spots.j == err_idx).to_numpy().sum()
            err_c_all.append( err_c )
            total_c_all.append(len(spots) - err_c)      

    # CALCULATE FALSE DISCOVERY RATE, GIVEN N_SPOTS FOUND AT INTENSITY THRESHOLD         
    err_c_all1 = np.reshape(err_c_all, [ len(infiles), 10 ])
    total_c_all1 = np.reshape(total_c_all, [ len(infiles), 10]) + 1
    fdr = err_c_all1 / len(pos_unused_codes[0]) * (len(genes)-len(pos_unused_codes[0])) / (total_c_all1)
    fdrmean = err_c_all1.mean(axis=0) / len(pos_unused_codes[0]) * (len(genes) - len(pos_unused_codes[0])) / (total_c_all1.mean(axis=0))

    thresh_refined=np.linspace( thresh-0.1, thresh+0.1, 10)[(fdrmean < fdrthresh).nonzero()[0][0]]

    #this is the new threshold optimized by targeted fdr value
    logging.info(f'intensity_thresh_refined = {thresh_refined}')
    
    param_outputs['intensity_thresh_refined'] = thresh_refined
    param_outputs['noisefloor_ini'] = noisefloor_ini
    param_outputs['noisefloor_final'] = noisefloor_final
    logging.info(f"threshold {thresh_refined} with noise floor {noisefloor_final}")
    logging.info(f"param_outputs= {param_outputs} {len(infiles)} input files. ")
    
    with open(outfile, 'w' ) as f:
        json.dump(param_outputs, f)
    logging.info(f'wrote params to {outfile}')
    return param_outputs

    
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
                    default='basecall-geneseq', 
                    type=str, 
                    help='stage we care calculating for. input learn')

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

    
    param_outputs = calc_params_bardensr(infiles=args.infiles, 
                                            outfiles=args.outfiles,
                                            stage=args.stage,  
                                            cp=cp   )
    print(param_outputs)
    
    logging.info(f'done processing output to {args.outfiles}')
 
 