#!/usr/bin/env python
#
# Use Cellpose to segemnt cells. 
#
# current inputs. 
#         hyb:  5 channels. 
#             hyb.  channel 3 (all-genes)
#             hyb.  channel 5. DAPI
#         geneseq:
#             sum(all_channels) from either geneseq01 or all geneseq* 
#
import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
from joblib import load, dump

import torch
import numpy as np

from cellpose import models, io
from cellpose.io import imread

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
 
from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def segment_cellpose( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    by convention hyb then geneseq. 
    create imagestack, 
    run cellpose

    Input, e.g.
        
        (['hyb01/MAX_Pos1_000_000.tif',
        'geneseq01/MAX_Pos1_000_000.tif',
        'geneseq02/MAX_Pos1_000_000.tif',
        'geneseq03/MAX_Pos1_000_000.tif',
        'geneseq04/MAX_Pos1_000_000.tif',
        'geneseq05/MAX_Pos1_000_000.tif',
        'geneseq06/MAX_Pos1_000_000.tif',
        'geneseq07/MAX_Pos1_000_000.tif'],
        --> 
        ['hyb/MAX_Pos1_000_000.cp_mask_cyto3.tif'])


    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'segment'

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


    model_name = cp.get(stage, 'model_name')
    cell_diameter = cp.getint(stage, 'cell_diameter')
    use_gpu = torch.cuda.is_available()
    logging.info(f'running with model_name={model_name} cell_diam={cell_diameter} use_gpu={use_gpu}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, infile_label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    
    logging.info('Preparing input image stack...')
    cellpose_input_stack = prepare_cellpose_input(infiles, outfiles, stage=stage, cp=cp )
    logging.debug(f'got cellpose input image shape={cellpose_input_stack.shape}')

    logging.info('Running cellpose')
    model = models.Cellpose( model_type = model_name,
                             gpu = use_gpu)
    channels = [[0,1]]
    logging.info('running cellpose...')
    masks, flows, styles, diams = model.eval( cellpose_input_stack, 
                                              diameter=cell_diameter, 
                                              channels=channels )
    logging.debug(f'got masks. shape={masks.shape}')

    logging.info(f'writing to {outfile}')
    write_image(outfile, masks)
    logging.debug(f'done writing {outfile}')    


def prepare_cellpose_input(infiles, outfiles, stage=None, cp=None):
    '''
        cyto = hyb[0,1,2,3] + geneseq all channel all cycle composite, 
        nuclear = hyb[4].

        nuc_ch=5,
        num_chyb=5,
        num_cgene=4,
        other_channels = list(range(0,num_chyb))
    '''

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'segment'        

    num_cycles_hyb = cp.getint( stage, 'num_cycles_hyb') 
    num_cycles_geneseq = cp.getint(stage, 'num_cycles_geneseq')

    # For segmentation, we have the most complex input
    # multiple modes/cycles but would like to abstract channel names...
    instage = cp.get(stage, 'instage')
    instage_mode = get_config_list(cp, stage, 'instage_mode')

    mode = get_config_list(cp, stage, 'modes')

    # We know arity is single, so we can grab the outfile to build cellpose_input 
    # intermediate filename.  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')    

    (odirpath, obase, olabel, oext) = split_path(os.path.abspath(outfile))
    (prefix, subdir) = os.path.split(odirpath)
    logging.debug(f'outdirpath={odirpath} obase={obase} olabel={olabel} subdir={subdir}')
    outfile = os.path.join( outdir, f'{obase}.cellpose_input.tif' )
    logging.debug(f'preparing cellpose input to be written to {outfile}')

    channel_names_hyb =  get_config_list(cp, 'hyb', 'channels')
    channel_names_geneseq = get_config_list(cp, 'geneseq', 'channels')

    select_channels_hyb = get_config_list(cp, stage, 'select_channels_hyb')
    select_indexes_hyb = channel_names_index_map(select_channels_hyb, channel_names_hyb)

    select_channels_geneseq =  get_config_list(cp, stage, 'select_channels_geneseq')
    select_indexes_geneseq = channel_names_index_map(select_channels_geneseq, channel_names_geneseq)

    allgenes_channel_hyb = get_config_list(cp, stage, 'allgenes_channel_hyb')
    allgenes_indexes_hyb = channel_names_index_map(allgenes_channel_hyb, channel_names_hyb)

    nuclear_channel_hyb = get_config_list(cp, stage, 'nuclear_channel_hyb')
    nuclear_indexes_hyb = channel_names_index_map(nuclear_channel_hyb, channel_names_hyb)

    cyto_channel_geneseq = get_config_list(cp, stage, 'cyto_channel_geneseq')
    cyto_indexes_geneseq = channel_names_index_map(cyto_channel_geneseq, channel_names_geneseq )

    cyto_channel_hyb = get_config_list(cp, stage, 'cyto_channel_hyb')
    cyto_indexes_hyb = channel_names_index_map(cyto_channel_hyb, channel_names_hyb )

    # Not needed now, but preparing for multi-cycle hyb...
    hyb_files = infiles[0:num_cycles_hyb]
    geneseq_files = infiles[num_cycles_hyb:num_cycles_geneseq + num_cycles_hyb]
    logging.debug(f'hyb_files = {hyb_files} geneseq_files = {geneseq_files}')

    hyb_file = hyb_files[0]
    hyb_image = read_image(hyb_file)

    cp_input_image = np.zeros( [2, hyb_image.shape[1], hyb_image.shape[2]] )
    geneseq_composite = np.zeros( [ hyb_image.shape[1], hyb_image.shape[2] ] )

    for geneseq_file in geneseq_files:
        geneseq_image = read_image(geneseq_file, channels=cyto_indexes_geneseq  )
        geneseq_composite = geneseq_composite + np.sum( geneseq_image, axis=0 )
    
    nuclear_image = hyb_image[ nuclear_indexes_hyb ]
    nuclear_image = np.squeeze(nuclear_image)
    
    cyto_image = np.sum( hyb_image[cyto_indexes_hyb ], axis=0 ) + geneseq_composite 
    
    cp_input_image[0,:,:]=uint16m(cyto_image)
    cp_input_image[1,:,:]=uint16m(nuclear_image)
    cp_input_image = uint16m(cp_input_image)

    logging.debug(f'made cellpose input image. shape={cp_input_image.shape}')
    logging.debug(f'writing intermediate cellpose input to {outfile} ...')
    write_image(outfile, cp_input_image)
    logging.debug(f'returning intermediate image...')
    return cp_input_image


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

    segment_cellpose( infiles=args.infiles, 
                      outfiles=args.outfiles,
                      stage=args.stage,  
                      cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

