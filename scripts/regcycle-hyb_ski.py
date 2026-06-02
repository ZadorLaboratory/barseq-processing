#!/usr/bin/env python
#
#
import argparse
import logging
import os
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np
import skimage
from skimage.exposure import match_histograms
from skimage.registration import phase_cross_correlation as pcc

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def regcycle_hyb_ski(infiles, outfiles, template=None, stage=None, cp=None ):
    '''
    @arg infiles    tiles across 1 or more cycles
    @arg outdir     TOP-LEVEL out directory
    @arg template   optional file to use as template against infiles, 
                    otherwise register to first. 
    @arg cp         ConfigParser object
    @arg stage      stage label in cp
    
    infiles will be hyb/        will be a single file. 
            use only reg_channel for registration. 
    template will be geneseq01/ single file. 

    '''
    if cp is None:
        cp = get_default_config()
    
    if stage is None:
        stage = 'regcycle-hyb'

    image_type = cp.get(stage, 'image_type')
    channel_names =  get_config_list(cp, image_type, 'channels')
    reg_channels = get_config_list(cp, stage, 'reg_channels')
    reg_indexes = channel_names_index_map(reg_channels, channel_names)

    select_channels = get_config_list(cp, stage, 'select_channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    
    template_image_type = cp.get( stage, 'template_image_type')
    template_channel_names = get_config_list(cp, template_image_type , 'channels')
    template_select_channels = get_config_list(cp, stage, 'template_select_channels')
    template_select_indexes = channel_names_index_map(  template_select_channels, 
                                                        template_channel_names)

    logging.info(f' stage={stage} template={template}')
    logging.debug(f'select_channels={select_channels} select_indexes={select_indexes}') 
    logging.debug(f'template_select_channels={template_select_channels} template_select_indexes={template_select_indexes}')

    upsample_factor=cp.getint(stage, 'upsample_factor')
    logging.debug(f'upsample_factor={upsample_factor}')

    # We know output is singleton
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
    logging.info(f'Handling {infiles} -> {outfile}')
    (dirpath, base, label, ext) = split_path(os.path.abspath(infiles[0]))

    # We know input is singleton
    infile = infiles[0]

    # Determine template(fixed) file name
    if template is None:
        template_file = infiles[0]
    else:
        template_file = template
    logging.debug(f'fixed_file = {template_file}')

    # template_image=tfl.imread(template,key=range(0,num_c,1))
    # Make template_sum_norm of template (fixed) file. i.e. geneseq01
    logging.debug(f'Reading template file: {template_file} all channels. ')
    template_image = read_image( template_file, template_select_indexes)

    template_sum=np.double(np.sum( template_image, axis=0))
    template_sum_norm=np.divide(template_sum, np.max(template_sum, axis=None))
    fixed=template_sum_norm.copy()
    logging.debug(f'fixed (template) file shape={fixed.shape}')

    logging.debug(f'Reading moving file: {infile} ')
    hyb_orig = read_image( infile , select_indexes )  
    
    moving_hyb=hyb_orig[reg_indexes,:,:]
    moving_input = moving_hyb.copy()
    moving=np.squeeze(moving_input,axis=0)
    moving_norm=np.divide(moving,np.max(moving,axis=None))
    moving=moving_norm

    moving=np.uint8(np.clip(moving*255,0,255))
    fixed=np.uint8(np.clip(fixed*255,0,255))

    hmatched_moving = match_histograms(moving, fixed)
    shift_values,_,_ = pcc(fixed, hmatched_moving, upsample_factor=upsample_factor)
    tform_hyb=skimage.transform.SimilarityTransform( translation=(-shift_values[1], -shift_values[0]))

    Ih_aligned=np.zeros_like(hyb_orig)
    for i in range(hyb_orig.shape[0]):
        Ih_aligned[i,:,:] = skimage.transform.warp( np.squeeze(hyb_orig[i,:,:]), 
                                                    tform_hyb, 
                                                    preserve_range=True, 
                                                    output_shape=(  template_sum_norm.shape[0], 
                                                                    template_sum_norm.shape[1])
                                                  )
    logging.debug(f'done processing {base}.{ext} ')
    logging.info(f'writing to {outfile}')
    write_image(outfile, Ih_aligned)
    logging.debug(f'done writing {outfile}')



# Notebook code
def align_hyb_to_gene( moving_hyb, template_sum_norm, Ibcksub_shifted_btcorr ):
    """
    Preprocessing function:
    Given two images from one tile--hyb and gene--aligns hyb to geneseq
    Returns aligned image and transformation matrix
    """
    moving=moving_hyb.copy()
    moving=np.squeeze(moving,axis=0)
    moving_norm=np.divide(moving, np.max(moving,axis=None))
    moving=moving_norm
    
    fixed=template_sum_norm.copy()
    moving=np.uint8(np.clip(moving*255,0,255))
    fixed=np.uint8(np.clip(fixed*255,0,255))

    hmatched_moving = match_histograms(moving, fixed)
    
    shift_values,_,_ = pcc(fixed, hmatched_moving, upsample_factor=100)
    
    tform_hyb=skimage.transform.SimilarityTransform(translation=(-shift_values[1],-shift_values[0]))

    Ih_aligned=np.zeros_like(Ibcksub_shifted_btcorr)
    for i in range(Ibcksub_shifted_btcorr.shape[0]):
        Ih_aligned[i,:,:]=skimage.transform.warp(np.squeeze(Ibcksub_shifted_btcorr[i,:,:]),tform_hyb,preserve_range=True,output_shape=(template_sum_norm.shape[0],template_sum_norm.shape[1]))
    
    return Ih_aligned,tform_hyb

    
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

    (outdir, file) = os.path.split(args.outfiles[0])
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")
    regcycle_hyb_ski( infiles=args.infiles,  
                  outfiles=args.outfiles,
                  template=args.template, 
                  stage=args.stage, 
                  cp=cp )
    
    logging.info(f'done processing output to {outdir}')