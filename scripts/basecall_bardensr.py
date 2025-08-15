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


def bd_read_image(infile, R, C, trim=None, cropf=None ):
    I = []
    for i in range(1,R+1):
        for j in range(C):
            I.append( np.expand_dims( read_image(infile, channel=j), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[:,:,round(ny*cropf):round(ny*(1-cropf)),round(nx*cropf):round(nx*(1-cropf))]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:,:,trim:-trim,trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    return I


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

    fdrthresh=cp.getfloat(stage, 'fdrthresh')
    trim=cp.getint(stage, 'trim')
    cropf=cp.getfloat(stage, 'cropf')
    noisefloor_ini=cp.getfloat(stage, 'noisefloor_ini')
    noisefloor_final=cp.getfloat(stage, 'noisefloor_final')
    logging.debug(f'fdrthresh={fdrthresh} trim={trim} cropf={cropf} noisefloor_ini={noisefloor_ini}  noisefloor_final={noisefloor_final} ')
    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    

    # load codebook from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = cp.get(stage, 'codebook_bases').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_df(cfile)
    num_channels = len(codebook_bases)
    genes = np.reshape( np.array( codebook['gene'], dtype='<U8'), (np.size(codebook,0),-1) ) 
    logging.debug(f'loaded codebook:\n{codebook} codebook_bases={codebook_bases}')    
    
    # make codebook array to match actual number of cycles
    # it is possible that there are fewer cycles than codebook sequence lengths?
    n_cycles = len(infiles)
    codebook_char = np.zeros((len(codebook),n_cycles),dtype=str)
    logging.debug(f'made empty array shape={codebook_char.shape} filling... ')
    
    codebook_seq = codebook['sequence']
    for i in range(len(codebook)):
        for j in range(n_cycles):
            #bcodebook[i,j]=codebook[i][1][0][j]        
            codebook_char[i,j] = codebook_seq.iloc[i][j]
    logging.debug(f'made sequence array {codebook_char}. making binary array.')
        
    codebook_bin=np.ones(np.shape(codebook_char), dtype=np.double)    
    bmax = math.pow(2, len(codebook_bases) - 1)
    rmap = {}
    for bchar in codebook_bases:
        rmap[bchar] = bmax
        bmax = bmax / 2
    logging.debug(f'made binary mappings for chars: {rmap}')
    
    codebook_bin=np.reshape( np.array([ rmap[x] for y in codebook_char for x in y]), np.shape(codebook_char))
    logging.debug(f'binary codebook = {codebook_bin}')
    #codebook_bin=np.reshape( np.array([float( x.replace('G','8').replace('T','4').replace('A','2').replace('C','1')) for y in codebook_char for x in y]), np.shape(codebook_char))
    codebook_bin=np.matmul(np.uint8(codebook_bin), 2**np.transpose(np.array((np.arange(4 * n_cycles -4, -1, -4)))))
    codebook_bin=np.array([bin(i)[2:].zfill(n_cycles * num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook_char, 0), n_cycles * num_channels))
    logging.debug(f'reshaped codebook_bin={codebook_bin}')

    co=[[genes[i],codebook_bin[j,:]] for i in range(np.size(genes,0))]
    co=[codebook,co]  
    codebook_bin=np.reshape(codebook_bin,(np.size(codebook_bin,0),-1,num_channels))
    logging.debug(f'final codebook_bin={codebook_bin}')
    
    cb = np.transpose(codebook_bin, axes=(1,2,0))
    R,C,J=cb.shape
    codeflat=np.reshape(cb,(-1,J))
    
    # CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    logging.debug(f'calculating max_per_RC...')
    max_per_RC=[ bd_read_image(infile, R, C, cropf=cropf).max(axis=(1,2,3)) for infile in infiles ]
    
    s = pprint.pformat(max_per_RC, indent=4)
    logging.debug(f'max per RC = {s}')
    median_max=np.median(max_per_RC,axis=0)
    s = pprint.pformat(median_max, indent=4)
    logging.debug(f'median_max = {s}')
    
    # ESTABLISHING BASE THRESHOLD AT THE MEDIAN OF MAXIMUM ERROR READOUT
   
   
    outfile = f'{outdir}/{image_type}/{base}.spots.tsv'
    logging.debug(f'outfile={outfile}') 
    








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
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_trimmed(os.path.join(pth,'processed',folder,'aligned'),trim,R,C)/median_max[:,None,None,None],codeflat,noisefloor_final)
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

