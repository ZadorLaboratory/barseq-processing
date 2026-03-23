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


def basecall_ski( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    load codebook, 
      
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
    (dirpath, base, label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')


# NOTEBOOK CODE

def basecall_barcodes_rolony(pth,relaxed=0,thresh=[30,30,30,30],prominence=[30,30,30,30],num_cycles=15,num_c=4):
    [folders,pos,x,y]=get_folders(pth)
    lroi_x_all=[]
    lroi_y_all=[]
    id_t_all=[]
    sig_t_all=[]
    score_t_all=[]
    seq_t_all=[]
    for folder in folders:
        print(f'BARCODE BASECALL FOLDER {folder}')
        pthw=os.path.join(pth,'processed',folder,'aligned')
        I=[]
        for i in range(num_cycles):
            I.append(tfl.imread(os.path.join(pthw,'alignedregn2vbcseq'+str("%0.2d"%(i+1))+'.tif'),key=range(0,4,1)))
        [lroi_x,lroi_y,id_t,sig_t,score_t,seq_t]=basecall_bc_one_image(pthw,num_c,I,thresh,prominence)
        lroi_x_all.append(lroi_x[0])
        lroi_y_all.append(lroi_y[0])
        id_t_all.append(id_t[0])
        sig_t_all.append(sig_t[0])
        score_t_all.append(score_t[0])
        seq_t_all.append(seq_t[0])
    score_t_all=np.array(score_t_all)
    score_t_all[np.isnan(score_t_all)]=0.5
    score_t_all=score_t_all.tolist()
    dump({"lroi_x_all":lroi_x_all,"lroi_y_all":lroi_y_all,"id_t_all":id_t_all,"sig_t_all":sig_t_all,"score_t_all":score_t_all,"seq_t_all":seq_t_all},os.path.join(pth,'processed','bc.joblib'))


def basecall_bc_one_image(pthw,num_c,I,thresh,prominence):
    """
    Basecalling function:
    1. Basecalls bc for one tile
    2. Writes segmented rolony image per tile
    3. Returns basecall results to the calling function
    """ 
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    score_t=[]
    mask=np.zeros_like(I[0])
    for n in range(num_c):
        a=I[0][n,:,:]
        a_mask=a>thresh[n]
        a_masked=a*a_mask
        a_max= extrema.h_maxima(a_masked,prominence[n])
        
        
        mask[n,:,:]=uint16m(binary_dilation(a_max))
        # OK, SO I WILL REMOVE DILATION FROM HERE AND DILATE AFTER OVERLAP REMOVAL IN NEXT LINE
    comb_mask=clear_overlapping_rolonies(np.max(mask,axis=0))
    tfl.imwrite(os.path.join(pthw,'mask_bc.tif'),mask,photometric='minisblack')
    tfl.imwrite(os.path.join(pthw,'comb_mask_bc.tif'),comb_mask,photometric='minisblack')
    label_peaks = label(comb_mask)
    m=regionprops(label_peaks,comb_mask)
    
    [lroi_x,lroi_y,id_t,sig_t,score_t,seq]=quantify_peaks_bc(lroi_x,lroi_y,id_t,sig_t,score_t,m,I)   

    dump({"lroi_x":lroi_x,"lroi_y":lroi_y,"id_t":id_t,"sig_t":sig_t,"score_t":score_t,"seq":seq},os.path.join(pthw,'basecalls-bc.joblib'))

    return(lroi_x,lroi_y,id_t,sig_t,score_t,seq)


def clear_overlapping_rolonies(a):
    c=np.array([[0,1,1],[0,0,1],[0,0,1]])
    a=a & skimage.util.invert(binary_dilation(a,c))
    return a


def quantify_peaks_bc(lroi_x,lroi_y,id_t,sig_t,score_t,m,I):
    """
    Basecalling function:
    1. Based on the regionprops results per tile, this function creates bc basecalling output and decodes the gene
    2. Returns the basecall output to the calling function
    """ 
    sig2=[]
    lroi1_x=[]
    lroi1_y=[]
    id2=[]
    score2=[]
    gene_map=np.array(list("GTAC")) 
    
    for i,peaks in enumerate(m):
        lroi1_x.append(peaks.centroid[0])
        lroi1_y.append(peaks.centroid[1])
        score1=[]
        sig1=[]
        id1=[]
        #score=0
        for j in range(len(I)):
            intensity=I[j][:,peaks.coords[0][0],peaks.coords[0][1]]
            #score[np.isnan(score)]=0.5
            sig1.append(np.max(intensity))
            id1.append(np.argmax(intensity))
            score1.append(np.max(intensity)/np.sqrt(np.sum(np.square(intensity))))
        sig2.append(sig1)
        id2.append(id1)
        score2.append(score1)
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id2)
    sig_t.append(sig2)
    score_t.append(score2)
    seq=gene_map[id_t]   
    return(lroi_x,lroi_y,id_t,sig_t,score_t,seq)

def basecall_bc_soma_all(pth,num_ch=4,mname='dil_cell_mask_cyto3.tif',fname='alignedregn2vbcseq'):
    bc_label_all=[]
    bc_sig_all_channels_all=[]
    bc_sig_all=[]
    bc_id_all=[]
    bc_score_all=[]
    bc_seq_all=[]
    [folders,pos,x,y]=get_folders(pth)

    for folder in folders:
        [bc_label,bc_sig_all_channels,bc_sig,bc_id,bc_score,bc_seq]=basecall_bc_soma_one_image(pth,folder,num_ch,mname,fname)
        bc_sig_all_channels_all.append(bc_sig_all_channels)
        bc_sig_all.append(bc_sig)
        bc_id_all.append(bc_id)
        bc_score_all.append(bc_score)
        bc_seq_all.append(bc_seq)
        bc_label_all.append(bc_label)
        print(f'Basecalled soma-bc folder {folder}')

    dump({"bc_label":bc_label_all,"bc_sig_all_channels":bc_sig_all_channels_all,"bc_sig":bc_sig_all," bc_id": bc_id_all,"bc_score":bc_score_all,"bc_seq":bc_seq_all},os.path.join(pth,'processed','all_bccells_intensity.joblib'))


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

    basecall_ski( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

