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


def basecall_hyb_ski( infiles, outfiles, stage=None, cp=None):
    '''
    take in arbitrary list of files, parallel output to outfiles. 
    
    
      
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-hyb'

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
    (dirpath, base, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')

    # Stage-specific tool params
    all_genes_ch=cp.getint(stage, 'all_genes_ch')    
    thresh_str = cp.get( stage,'thresh')    
    relaxed = cp.getboolean( stage, 'relaxed')
    no_deconv = cp.getboolean( stage, 'no_deconv')
    filter_overlap = cp.getint( stage, 'filter_overlap')
    num_c = cp.getint( stage, 'num_c')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    
    # Parameters that need evaluation
    prominence_str = cp.get( stage,'prominence')
    logging.debug(f'params. thresh_str={thresh_str} prominence_str={prominence_str} evaluating... ')
    prominence = eval( prominence_str ) 
    thresh = eval( thresh_str )
    logging.debug(f'all_genes_ch={all_genes_ch} thresh={thresh} prominence={prominence}')
       
    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_channels = cp.get(stage, 'codebook_channels').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    
    # Basecall loop.
    for infile in infiles:
        if len(infiles) == 1:
            #raw = read_image(infile, key=range(0,num_c,1))
            raw = read_image(infile)
            if no_deconv:
                raw_2 = raw
                raw_2[all_genes_ch,:,:] = 0
                [lroi_x_ind, lroi_y_ind, id_t_ind, sig_t_ind]=basecall_ski_single(infile, 
                                                                               num_c=num_c, 
                                                                               all_genes_ch=all_genes_ch, 
                                                                               raw_2=raw_2, 
                                                                               thresh=thresh, 
                                                                               prominence=prominence)
                logging.debug(f'got result: lroi_x_ind={lroi_x_ind}, lroi_y_ind={lroi_y_ind}, id_t_ind={id_t_ind}, sig_t_ind={sig_t_ind} ')
            else:
                logging.error('Deconvolution to be introduced')
        else:
            logging.warning('Multiple basecall TBD')

    logging.debug(f'dumping results to {outfile}')
    dump({"lroi_x":lroi_x_ind,
          "lroi_y":lroi_y_ind,
          "gene_id":id_t_ind,
          "signal":sig_t_ind},
          outfile)

   



def basecall_ski_single(infile, 
                        num_c,
                        all_genes_ch,
                        raw_2,
                        thresh,
                        prominence                        
                        ):
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    mask=np.zeros_like(raw_2)
    for n in range(num_c):
        if n == all_genes_ch:
            mask[n,:,:]=0
            continue
        else:
            a = raw_2[n,:,:]
            a_mask = a > thresh[n]
            a_masked = a * a_mask
            a_max= extrema.h_maxima( a_masked, prominence[n])
            label_peaks = label(a_max)
            m = regionprops(label_peaks, a_masked)
            mask[n,:,:] = uint16m(binary_dilation(a_max))
            [lroi_x,lroi_y,id_t,sig_t] = quantify_peaks(lroi_x, lroi_y, id_t, sig_t, m, raw_2)

    #tfl.imwrite(os.path.join(pthw,'mask_hyb.tif'), mask, photometric='minisblack')
    return(lroi_x, lroi_y, id_t, sig_t)


def quantify_peaks(lroi_x, lroi_y, id_t, sig_t, m, raw_2 ):
    '''
    Basecalling function:
    1. Based on the regionprops results per tile, this function creates hyb basecalling output and decodes the gene
    2. Returns the basecall output to the calling function
    ''' 
    
    sig1=[]
    lroi1_x=[]
    lroi1_y=[]
    id1=[]
    for i,peaks in enumerate(m):
        lroi1_x.append( peaks.centroid[0])
        lroi1_y.append( peaks.centroid[1])
        sig1.append( peaks.intensity_max)
        id1.append( np.argmax(raw_2[:,peaks.coords[0][0],peaks.coords[0][1]])+1) # added 1 here to max channel to match codebook 1,2 and 4
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id1)
    sig_t.append(sig1)
    return( lroi_x, lroi_y, id_t, sig_t)



#
# CODE FROM NOTEBOOK
#

def basecall_hyb_all_tiles_orig(pth,cyclename,config_pth,codebook_hyb_name,thresh,prominence,num_c=4,all_genes_ch=2,no_deconv=1):
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



def basecall_hyb_one_image_orig(pthw,num_c,all_genes_ch,hyb_2,thresh,prominence):
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

def quantify_peaks_orig(lroi_x,lroi_y,id_t,sig_t,m,hyb_2):
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

    basecall_hyb_ski( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

