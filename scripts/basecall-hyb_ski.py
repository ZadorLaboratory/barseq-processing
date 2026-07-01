#!/usr/bin/env python
#
# Do basecalling on batches of images.
# used for hyb

import argparse
import logging
import math
import os
import re
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
import joblib

import matplotlib.pylab as plt
import numpy as np

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def basecall_hyb_ski( infiles, outfiles, stage=None, cp=None):
    '''
    Handle single tile. For current hyb, should be single infile. 
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

    # We know input is single, so we can grab the infile 
    infile = infiles[0]

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    position_regex = cp.get(stage, 'position_regex')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, file_label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')

    # Stage-specific tool params
    all_genes_ch = cp.getint(stage, 'all_genes_ch')    
    thresh_str = cp.get( stage,'thresh')    
    relaxed = cp.getboolean( stage, 'relaxed')
    no_deconv = cp.getboolean( stage, 'no_deconv')
    filter_overlap = cp.getint( stage, 'filter_overlap')
    num_c = cp.getint( stage, 'num_c')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    
    # Parameters that need evaluation
    prominence_str = cp.get( stage, 'prominence')
    logging.debug(f'params. thresh_str={thresh_str} prominence_str={prominence_str} evaluating... ')
    prominence = eval( prominence_str ) 
    thresh = eval( thresh_str )
    logging.debug(f'all_genes_ch={all_genes_ch} thresh={thresh} prominence={prominence}')

    (dirpath, base ) = os.path.split(infile)
    m = re.search(position_regex, base)
    if m is not None:
        pos_id = m.group(1)
    else:
        logging.error(f'Unable to extract position index from file base: {base}')
        sys.exit(2)
    logging.info(f'handling pos_id={pos_id}')

    #hyb_raw=tfl.imread(os.path.join(hybseq[0]), key=range(0,num_c,1))
    readchannels = list(range(0,num_c))
    hyb_2=read_image(infile, channels=readchannels)
    # zero-ing all-genes channel 3 (index 2)
    hyb_2[all_genes_ch,:,:] = 0  
    logging.debug(f'basecalling {infile} hyb_2.shape = {hyb_2.shape} all_genes_ch = {all_genes_ch} thresh={thresh} prominence={prominence}')
    
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    mask = np.zeros_like(hyb_2)
    for n in range(num_c):
        if n == all_genes_ch:
            mask[n, :, :] = 0
            continue
        else:
            a = hyb_2[n,:,:].copy()
            a_mask = a > thresh[n]
            a_masked = a * a_mask
            a_max = extrema.h_maxima(a_masked, prominence[n])
            label_peaks = label(a_max)
            m = regionprops(label_peaks, a_masked)
            # m_t = regionprops_table(label_peaks, a_masked)
            mask[n,:,:] = uint16m(binary_dilation(a_max))
            [lroi_x, lroi_y, id_t, sig_t] = quantify_peaks( lroi_x, lroi_y, id_t, sig_t, m, hyb_2)

    # Write mask_hyb image. 
    (dirpath, base ) = os.path.split(infile)
    (base,ext) = os.path.splitext(base)
    ext = ext[1:]
    of = os.path.join( outdir, f'{base}.mask_hyb.{ext}' )
    logging.debug(f'Writing mask to {of}')
    write_image(of, mask)

    gene_map = np.zeros(hyb_2.shape[1:], dtype=np.uint8)
    for ch_idx in range(len(lroi_x)):
        for k in range(len(lroi_x[ch_idx])):
            r = int(round(lroi_x[ch_idx][k]))
            c = int(round(lroi_y[ch_idx][k]))
            gene_map[r, c] = id_t[ch_idx][k] + 1  # +1 so background stays 0

    # Write out gene map. 
    of = os.path.join( outdir, f'{base}.basecall_map_hyb.{ext}' )
    logging.debug(f'Writing basecall map to {of}')
    write_image(of, gene_map)
    #return(lroi_x, lroi_y, id_t, sig_t)

    logging.debug(f'got result: lroi_x={len(lroi_x)}, lroi_y={len(lroi_y)}, id_t={len(id_t)}, sig_t={len(sig_t)} ')

    data_dict = {"lroi_x":  lroi_x, 
                 "lroi_y": lroi_y, 
                 "gene_id": id_t, 
                 "signal": sig_t}

    # Write out joblib
    logging.info(f'Writing results to {outfile}')
    joblib.dump(data_dict, outfile)


def quantify_peaks(lroi_x, lroi_y, id_t, sig_t, m, hyb_2):
    """
    Basecalling function:
    1. Based on the regionprops results per tile, this function creates hyb basecalling output and 
       decodes the gene
    2. Returns the basecall output to the calling function
    """ 
    sig1=[]
    lroi1_x=[]
    lroi1_y=[]
    id1=[]
    ch_to_gene={0:0, 1:1, 3:2}  # skip all_genes_ch=2
    for i, peaks in enumerate(m):
        lroi1_x.append(peaks.centroid[0])
        lroi1_y.append(peaks.centroid[1])
        sig1.append(peaks.intensity_max)
        peaks_s = print_regionprop(peaks)
        #logging.debug(f'hyb_2 = {hyb_2}')
        logging.debug(f'hyb_2.shape = {hyb_2.shape}')
        logging.debug(f'\n [{i}] {peaks_s}')
        id1.append( ch_to_gene[ np.argmax( hyb_2[:, peaks.coords[0][0], peaks.coords[0][1]]) ])
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id1)
    sig_t.append(sig1)
    return(lroi_x, lroi_y, id_t, sig_t)

def print_regionprop(prop):
    s = ''
    s += f'\n  coords[0][0] = {prop.coords[0][0]}'
    s += f'\n  coords[0][1] = {prop.coords[0][1]}'
    s += f'\n  orientation={prop.orientation}'
    return s

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

