#!/usr/bin/env python
#
# Do basecalling on batches of images.
#
#
#
#

import argparse
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

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *


def basecall_bardensr( infiles, outdir, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    load codebook, 
    run bardensr, 
    output evidence tensor dataframe to <outdir>/<mode>/<prefix>.brdnsr.tsv   
    
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-geneseq'

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    
    noisefloor_final = cp.getfloat(stage, 'noisefloor_final')
    

    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = cp.get(stage, 'codebook_bases').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_codebook_file(cfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles)
    (codeflat, R, C, J, pos_unused_codes) = make_codebook_object(codebook, codebook_bases, n_cycles=n_cycles)
    
       
    # bd_read_image(infile, R, C, cropf=cropf)
    # need median_max
    # need thresh_refined
    # need noisefloor_final
    et= bardensr.spot_calling.estimate_density_singleshot( bd_read_image(infile, R, C, trim=trim ) / median_max[:, None, None, None], codeflat, noisefloor_final)
    spots=bardensr.spot_calling.find_peaks(et, thresh_refined, use_tqdm_notebook=False)
    spots.loc[:,'m1']=spots.loc[:,'m1']+trim
    spots.loc[:,'m2']=spots.loc[:,'m2']+trim
    
    
    outfile = f'{outdir}/{image_type}/{base}.spots.tsv'
    spots.to_csv(outfile,index=False)   
    logging.debug(f'wrote spots to outfile={outfile}') 
    









def bardensr_call(pth,
                  config_pth,
                  num_channels=4,
                  codebook_name='codebookM1all.mat',
                  fdrthresh=0.05,
                  trim=160,
                  cropf=0.4,
                  noisefloor_ini=0.01,
                  noisefloor_final=0.05):
    
    make_codebook_bin(pth, num_channels, codebook_name)

    controlidx=range(25,35)
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
        control_folders=folders[controlidx]
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

    # MAIN BASE-CALLING ON ALL FOLDERS-TRIMMED IMAGES WITH FINALIZED THRESHOLD
    # # spot call each fov, using the thresholds we decided on, and the normalization we decided on
    for folder in folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_trimmed(os.path.join(pth,'processed',folder,'aligned'),trim,R,C)/median_max[:,None,None,None],codeflat, noisefloor_final)
        spots=bardensr.spot_calling.find_peaks(et,thresh_refined,use_tqdm_notebook=False)
        spots.loc[:,'m1']=spots.loc[:,'m1']+trim
        spots.loc[:,'m2']=spots.loc[:,'m2']+trim
        spots.to_csv(os.path.join(pth,'processed',folder,'aligned','bardensrresult.csv'),index=False)
        print(f"in {folder} found {len(spots)} spots")





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
    
    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None, 
                    type=str, 
                    help='outdir. output base dir if not given.')

    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')
   
    parser.add_argument('infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 
       
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
      
    outdir = os.path.abspath('./')
    if args.outdir is not None:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    basecall_bardensr( infiles=args.infiles, 
                       outdir=outdir,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {outdir}')

