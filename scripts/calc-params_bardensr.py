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
    
    noisefloor_ini = cp.getfloat( stage, 'noisefloor_ini')
    #noisefloor_final = cp.getfloat(stage, 'noisefloor_final')
    #intensity_thresh = cp.getfloat(stage, 'intensity_thresh')
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

    # CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    logging.debug(f'calculating max_per_RC...')
    max_per_RC=[ bd_read_image_single(infile, R, C, cropf=cropf).max(axis=(1,2,3)) for infile in infiles ]
    #max_per_RC=bd_read_images(infiles, R, C, cropf=cropf).max(axis=(1,2,3)) 
    #logging.debug(f'max_per_RC len={len(max_per_RC)} item len={len(max_per_RC[0])}')

    # Expected to be 28 values. channels * cycles. 
    # first max(), then median of those max() per cycle. 
    s = pprint.pformat(max_per_RC, indent=4)
    logging.debug(f'max per RC = {s}')
    median_max=np.median(max_per_RC, axis=0)
    logging.debug(f'median_max=\n{median_max}')
    #s = pprint.pformat(median_max, indent=4)
    #logging.debug(f'median_max = {s}')

    img_norm = bd_read_images(infiles, R, C, trim=trim ) / median_max[ :, None, None, None]
    logging.debug(f'img_norm shape={img_norm.shape}\ncodeflat={codeflat}\nnoisefloor_final={noisefloor_final}')
    et = bardensr.spot_calling.estimate_density_singleshot( img_norm, codeflat, noisefloor_final)
    logging.debug(f'estimated_density et = {et}')
    spots = bardensr.spot_calling.find_peaks( et, intensity_thresh, use_tqdm_notebook=False)
    spots.loc[:,'m1'] = spots.loc[:,'m1'] + trim
    spots.loc[:,'m2'] = spots.loc[:,'m2'] + trim            
    spots.to_csv(outfile, index=False)   
    logging.debug(f'wrote spots to outfile={outfile}')






def calc_bardensr_parameters_old(indir,
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
        dirpath, base, label, ext = split_path( os.path.abspath(file))
        dirpath, subdir, label, ext = split_path( os.path.abspath(dirpath))
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

# NOTEBOOK CODE
# R is number of cycles. 
# R = 7 trim = 160 C = 4
def image_reader_trimmed(infile, trim, R, C):
    """
    Helper function for basecalling function for geneseq:
    Returns a border trimmed in xy image stack (with cycles and channels intact)
    """
    I=[]
    for i in range(1,R+1):
        for j in range(C):
            I.append(np.expand_dims(tfl.imread(infile,key=j),axis=0))
    I=np.array(I)
    return(I[:,:,trim:-trim,trim:-trim])

def bardensr_call(pth,config_pth,num_channels=4,codebook_name='codebookM1all.mat',fdrthresh=0.05,trim=160,cropf=0.4,noisefloor_ini=0.01,noisefloor_final=0.05):
    """
    Basecalling function for geneseq:
    1. Calls the function that creates binarized codebook as per the number of geneseq cycles in the experiment
    2. Calculates normalization factors, intensity thresholds for spot detection based on allowed fdr
    3. Calls Bardensr spot calling and peak detection to identify and decode rolonies per tile
    4. Saves basecalling results per tile
    """ 
    make_codebook_bin(pth,num_channels,codebook_name)

    controlidx=np.array(range(25,35))
    [folders,_,_,_]=get_folders(pth)
    
    codebook=scipy.io.loadmat(os.path.join(config_pth,codebook_name))['codebook']
    genes=np.array([str(x[0][0]) for x in codebook],dtype=str)
    cb=load(os.path.join(pth,'processed','codebookforbardensr.joblib'))
    cb=np.transpose(cb,axes=(1,2,0))
    R,C,J=cb.shape
    codeflat=np.reshape(cb,(-1,J))
    pos_unused_codes=np.where(np.char.startswith(genes,'unused'))
    err_codes=genes[pos_unused_codes]

    # SELECTING 10 FOLDERS AS CONTROL
    if len(folders)>=36:
        control_folders=[folders[k] for k in controlidx]
    else:
        control_folders=folders

    # NORMALIZATION-PREPROCESSING--CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    max_per_RC=[image_reader_cropped(os.path.join(pth,'processed',i,'aligned'),R,C,cropf).max(axis=(1,2,3)) for i in control_folders]
    median_max=np.median(max_per_RC,axis=0)

    # ESTABLISHING BASE THRESHOLD AT THE MEDIAN OF MAXIMUM ERROR READOUT 

    err_max=[]
    evidence_tensors=[]
    for folder in control_folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_trimmed(os.path.join(pth,'processed',folder,'aligned'),trim,R,C)/median_max[:,None,None,None],codeflat,noisefloor_ini)
        err_max.append(et[:,:,:,pos_unused_codes].max(axis=(0,1,2)))
    err_max=np.array(err_max)
    thresh=np.median(np.median(err_max,axis=1))
    print(thresh)

    # FINDING OPTIMUM TRHESHOLD WITH LOWEST FDR ON CONTROL FOLDERS

    err_c_all=[]
    total_c_all=[]
    for folder in control_folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_cropped(os.path.join(pth,'processed',folder,'aligned'),R,C,cropf)/median_max[:,None,None,None],codeflat,noisefloor_final)
        for thresh1 in np.linspace(thresh-0.1,thresh+0.1,10):
            spots=bardensr.spot_calling.find_peaks(et,thresh1,use_tqdm_notebook=False)
            spots.to_csv(os.path.join(pth,'processed',folder,'aligned','spots.csv'),index=False)
            print(f"in {folder} found {len(spots)} spots")
            err_c=0
            for err_idx in pos_unused_codes[0]:
                err_c=err_c+(spots.j==err_idx).to_numpy().sum()
            err_c_all.append(err_c)
            total_c_all.append(len(spots)-err_c)      
    #calculate fdr        
    err_c_all1=np.reshape(err_c_all,[len(control_folders),10])
    total_c_all1=np.reshape(total_c_all,[len(control_folders),10])+1
    fdr=err_c_all1/len(pos_unused_codes[0])*(len(genes)-len(pos_unused_codes[0]))/(total_c_all1)
    fdrmean=err_c_all1.mean(axis=0)/len(pos_unused_codes[0])*(len(genes)-len(pos_unused_codes[0]))/(total_c_all1.mean(axis=0))
    thresh_refined=np.linspace(thresh-0.1,thresh+0.1,10)[(fdrmean<fdrthresh).nonzero()[0][0]]#this is the new threshold optimized by targeted fdr value
    print(thresh_refined)  
    with open(os.path.join(pth,'processed','thresh_refined.txt'),'w') as f:
        f.write(str(thresh_refined))
    with open(os.path.join(pth,'processed','noise_floors.txt'),'w') as f:
        f.write(str(noisefloor_final))
    print(f"threshold {thresh_refined} with noise floor {noisefloor_final}")
   
   
    
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
 
 