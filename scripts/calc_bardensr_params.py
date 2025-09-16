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

def calc_bardensr_parameters(indir,
                             outdir, 
                             outfile,
                             stage='basecall-geneseq',
                             cp=None,
                             ):
    '''
    Calculate bardenser experiment-specific image parameters. 

    @arg  indir      Barseq working directory, with stage subdirs. 
    @arg  outfile    parameter file output. [ bardensr-geneseq.params.txt ] ?
    @arg  stage      pipeline stage we are calculating for [basecall-geneseq]
    @arg  cp         experiment configuration file. 
    
    @return
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
    
    outfile = os.path.abspath(outfile)
    outfile_dir, fname = os.path.split(outfile)
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir, exist_ok=True)
        logging.debug(f'made outfile_dir={outfile_dir}')    
    
    project_id = cp.get('project','project_id')
    fdrthresh=cp.getfloat(stage, 'fdrthresh')
    trim=cp.getint(stage, 'trim')
    cropf=cp.getfloat(stage, 'cropf')
    noisefloor_ini=cp.getfloat(stage, 'noisefloor_ini')
    noisefloor_final=cp.getfloat(stage, 'noisefloor_final')
    logging.debug(f'fdrthresh={fdrthresh} trim={trim} cropf={cropf} noisefloor_ini={noisefloor_ini}  noisefloor_final={noisefloor_final} ')    
    
    logging.info(f'Processing experiment {project_id} indir={indir} outdir={outdir} to {outfile}')
    
    bse = BarseqExperiment(indir, outdir, cp)
    logging.debug(f'got BarseqExperiment metadata: {bse}')

    stagedir = cp.get(stage, 'stagedir')
    instage = cp.get(stage, 'instage')
    logging.debug(f'instage={instage}')
    instagedir = cp.get(instage, 'stagedir')
    logging.debug(f'instagedir={instagedir}')
    modes = cp.get( stage, 'modes').split(',') 
    prefix = os.path.join(outdir, stagedir)
    logging.debug(f'prefix={prefix}')
    infiles = []
    for mode in modes: 
        file_list = bse.get_filelist(mode=mode, stage=instage)
        for rpath in file_list:
            infiles.append( os.path.join(outdir, instagedir, rpath)   )
    logging.debug(f'got set of {len(infiles)} tiles, E.g. {infiles[0]}. Choosing sample...')

    # Load codebook to get R,C,J
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    codebook_file = cp.get(f'basecall-{mode}' , 'codebook_file')
    codebook_bases = cp.get(f'basecall-{mode}' , 'codebook_bases').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_codebook_file(cfile)   
        
    logging.info(f'making codebook object...') 
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook, codebook_bases)
    logging.debug(f'R={R} C={C} J={J}')

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
        dirpath, base, ext = split_path( os.path.abspath(file))
        dirpath, subdir, ext = split_path( os.path.abspath(dirpath))
        logging.debug(f'handling image base={base}')
        cropped = bd_read_image_single(file, R, C, cropf=cropf)
        img_norm = cropped / median_max[:, None, None, None]
        et=bardensr.spot_calling.estimate_density_singleshot( img_norm , codeflat, noisefloor_final)
        for thresh1 in np.linspace( thresh-0.1, thresh+0.1, 10):
            spots = bardensr.spot_calling.find_peaks(et, thresh1, use_tqdm_notebook=False)
            suboutdir = os.path.join( outfile_dir, 'bdparams', subdir)
            os.makedirs(suboutdir, exist_ok=True)
            logging.debug(f"found {len(spots)} spots in {file}")
            outsub = os.path.join(suboutdir, f'{base}.{thresh1}.spots.csv')
            logging.debug(f'writing spots to {outsub}')
            spots.to_csv(outsub, index=False)
            
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

    parser.add_argument('-I','--indir', 
                    metavar='indir',
                    default=None, 
                    type=str, 
                    help='Overall data input dir (with cycle subdirs) ')

    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    required = True,
                    default=None, 
                    type=str, 
                    help='BARseq working directory (with stage subdirs)')   
    
    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    default='./bardensr_params.txt', 
                    type=str, 
                    help='file to store calculation results')

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

    indir = os.path.abspath( os.path.expanduser( args.indir))
    outdir = os.path.abspath( os.path.expanduser( args.outdir))
    outfile = os.path.abspath( os.path.expanduser( args.outfile))
    
    logging.info(f'indir={indir}\noutdir={outdir}\noutfile={outfile}\nstage={args.stage}')
    param_outputs = calc_bardensr_parameters(indir=indir,
                                             outdir=outdir,  
                                             outfile=outfile,
                                             stage=args.stage,  
                                             cp=cp)
    print(param_outputs)
    
    logging.info(f'done processing output to {args.outfile}')
 
 