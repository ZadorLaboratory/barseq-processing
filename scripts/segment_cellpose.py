#!/usr/bin/env python
#
# Do basecalling on batches of images.
# used for hyb

import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
from joblib import load, dump

import matplotlib.pylab as plt
import numpy as np

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *


def basecall_ski( infiles, outdir, stage=None, cp=None):
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
        stage = 'basecall-hyb'

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

    all_genes_ch=cp.getint(stage, 'all_genes_ch')    
    thresh_str = cp.get( stage,'thresh')    
    prominence_str = cp.get( stage,'prominence')
    logging.debug(f'params. thresh_str={thresh_str} prominence_str={prominence_str} evaluating... ')
    prominence = eval( prominence_str ) 
    thresh = eval( thresh_str )
    logging.debug(f'all_genes_ch={all_genes_ch} thresh={thresh} prominence={prominence}')
       
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
    max_per_RC=[ bd_read_image(infile, R, C, cropf=cropf).max(axis=(1,2,3)) for infile in infiles ]
    
    # Expected to be 28 values. channels * cycles. 
    # first max(), then median of those max() per cycle. 
    #
    s = pprint.pformat(max_per_RC, indent=4)
    logging.debug(f'max per RC = {s}')
    median_max=np.median(max_per_RC, axis=0)
    #s = pprint.pformat(median_max, indent=4)
    #logging.debug(f'median_max = {s}')
    for infile in infiles:
        (dirpath, base, ext) = split_path(os.path.abspath(infile))
        (prefix, subdir) = os.path.split(dirpath)
        suboutdir = os.path.join(outdir, subdir)
        os.makedirs(suboutdir, exist_ok=True)
        outfile = os.path.join( outdir, subdir, f'{base}.spots.csv' )

        img_norm = bd_read_image(infile, R, C, trim=trim ) / median_max[:, None, None, None]
        et = bardensr.spot_calling.estimate_density_singleshot( img_norm, codeflat, noisefloor_final)
        spots = bardensr.spot_calling.find_peaks( et, intensity_thresh, use_tqdm_notebook=False)
        spots.loc[:,'m1'] = spots.loc[:,'m1'] + trim
        spots.loc[:,'m2'] = spots.loc[:,'m2'] + trim            
        spots.to_csv(outfile, index=False)   
        logging.debug(f'wrote spots to outfile={outfile}') 


#
# CODE FROM NOTEBOOK
#


def basecall_hyb_all_tiles(pth,
                       cyclename,
                       config_pth,
                       codebook_hyb_name,
                       thresh,prominence,num_c,all_genes_ch,no_deconv):
    pass

def basecall_hyb_all_tiles(pth,cyclename,config_pth,codebook_hyb_name,thresh,prominence,num_c=4,all_genes_ch=2,no_deconv=1):
    [folders,_,_,_]=get_folders(pth)
    codebook_hyb_path=os.path.join(config_pth,codebook_hyb_name)
    codebook_hyb=scipy.io.loadmat(codebook_hyb_path)['codebookhyb']
    lroi_x_all=[]
    lroi_y_all=[]
    id_t_all=[]
    sig_t_all=[]
    for folder in folders:
        hybseq=sorted(glob.glob(os.path.join(pth,'processed',folder,'aligned',cyclename+"*"+"tif")))
        m=len(hybseq)

        if m==1:
            hyb_raw=tfl.imread(os.path.join(hybseq[0]),key=range(0,num_c,1))
            if no_deconv:
                hyb_2=hyb_raw
                hyb_2[all_genes_ch,:,:]=0
                [lroi_x_ind,lroi_y_ind,id_t_ind,sig_t_ind]=basecall_hyb_one_image(os.path.join(pth,'processed',folder,'aligned'),num_c,all_genes_ch,hyb_2,thresh,prominence)
            else:
                print('Deconvolution to be introduced')
        else:
            print('Multiple hybseq basecall to be developed')
        lroi_x_all.append(lroi_x_ind)
        lroi_y_all.append(lroi_y_ind)
        id_t_all.append(id_t_ind)
        sig_t_all.append(sig_t_ind)
        print(f'HYB BASECALL COMPLETE FOR FOLDER {hybseq[0].split("/")[-3]}')
    dump({"lroi_x":lroi_x_all,"lroi_y":lroi_y_all,"gene_id":id_t_all,"signal":sig_t_all},os.path.join(pth,'processed','genehyb'+'.joblib'))



def basecall_hyb_one_image(pthw,num_c,all_genes_ch,hyb_2,thresh,prominence):
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    mask=np.zeros_like(hyb_2)
    for n in range(num_c):
        if n==all_genes_ch:
            mask[n,:,:]=0
            continue
        else:
            a=hyb_2[n,:,:]
            a_mask=a>thresh[n]
            a_masked=a*a_mask
            a_max= extrema.h_maxima(a_masked,prominence[n])
            label_peaks = label(a_max)
            m=regionprops(label_peaks,a_masked)
            mask[n,:,:]=uint16m(binary_dilation(a_max))
            [lroi_x,lroi_y,id_t,sig_t]=quantify_peaks(lroi_x,lroi_y,id_t,sig_t,m,hyb_2)

    tfl.imwrite(os.path.join(pthw,'mask_hyb.tif'),mask,photometric='minisblack')
    return(lroi_x,lroi_y,id_t,sig_t)

def quantify_peaks(lroi_x,lroi_y,id_t,sig_t,m,hyb_2):
    sig1=[]
    lroi1_x=[]
    lroi1_y=[]
    id1=[]
    for i,peaks in enumerate(m):
        lroi1_x.append(peaks.centroid[0])
        lroi1_y.append(peaks.centroid[1])
        sig1.append(peaks.intensity_max)
        id1.append(np.argmax(hyb_2[:,peaks.coords[0][0],peaks.coords[0][1]])+1) # added 1 here to max channel to match codebook 1,2 and 4
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id1)
    sig_t.append(sig1)
    return(lroi_x,lroi_y,id_t,sig_t)




    

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

