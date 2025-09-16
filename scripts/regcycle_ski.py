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
import skimage as ski
from skimage.util import view_as_blocks
from scipy.signal import convolve2d , correlate2d, fftconvolve

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def regcycle_ski(infiles, outfiles, template=None, stage=None, cp=None ):
    '''
    
    @arg infiles    tiles across cycles
    @arg outdir     TOP-LEVEL out directory
    @arg template   optional file to use as template against infiles, 
                    otherwise register to first. 
    @arg cp         ConfigParser object
    @arg stage      stage label in cp
    
    '''
    if cp is None:
        cp = get_default_config()
    
    if stage is None:
        stage = 'regcycle'
    
    subsample_rate = int(cp.get(stage,'subsample_rate'))
    resize_factor = int(cp.get(stage,'resize_factor'))
    block_size = int(cp.get(stage,'block_size')) 
    do_coarse = get_boolean( cp.get(stage, 'do_coarse') )
    num_channels = int(cp.get(stage,'num_channels'))
    logging.info(f' stage={stage} template={template}')
    logging.debug(f'num_channels={num_channels} do_coarse={do_coarse} block_size={block_size}')
    logging.debug(f'resize_factor={resize_factor} subsample_rate={subsample_rate}')

    if template is None:
        fixed_file = infiles[0]
    else:
        fixed_file = template
    logging.debug(f'fixed_file = {fixed_file}')

    fixed = read_image( fixed_file )
    fixed_sum = np.double(np.sum(fixed,axis=0))
    fixed_sum = np.divide(fixed_sum,np.max(fixed_sum, axis=None))
    sz=fixed_sum.shape
    b_x=np.floor(sz[0]/block_size)
    b_y=np.floor(sz[1]/block_size)

    for i, infile in enumerate( infiles ):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')
        (dirpath, base, ext) = split_path(os.path.abspath(infile))
        

        moving = read_image( infile )
        total_channels = len(moving )
        logging.debug(f'loaded image w/ {total_channels} channels. processing {num_channels} channels.')
        
        moving_sum=np.double(np.sum(moving, axis=0))
        moving_sum=np.divide(moving_sum, np.max(moving_sum, axis=None))

        if b_x*block_size!=sz[0]:
            fixed_sum=fixed_sum[0:b_x*block_size-1,:]
            moving_sum=moving_sum[0:b_x*block_size-1,:]
    
        if b_y*block_size!=sz[1]:
            fixed_sum=fixed_sum[:,0:b_y*block_size-1]
            moving_sum=moving_sum[:,0:b_y*block_size-1]
            
        moving_rescaled=np.uint8(ski.transform.rescale(moving_sum, resize_factor)*255) 
        
        # check if this uint8 needs to be changed as per matlab standard-ng
        fixed_rescaled=np.uint8(ski.transform.rescale(fixed_sum, resize_factor)*255)
        moving_split=view_as_blocks(moving_rescaled, block_shape=( block_size*resize_factor, 
                                                                   block_size*resize_factor))
        fixed_split=view_as_blocks(fixed_rescaled, block_shape=( block_size*resize_factor, 
                                                                 block_size*resize_factor))
        fixed_split_lin=np.reshape(fixed_split,(-1, fixed_split.shape[2], fixed_split.shape[3]))       
        fixed_split_sum=[np.sum(j,axis=None) for i,j in enumerate(fixed_split_lin)]
        idx=np.argsort(fixed_split_sum)[::-1]
        fixed_split_sum=np.reshape(fixed_split_sum,(fixed_split.shape[0],fixed_split.shape[1]))
        moving_split_lin=np.reshape(moving_split,(-1,moving_split.shape[2],moving_split.shape[3]))
        xcorr2=lambda a,b: fftconvolve(a, np.rot90(b,k=2))
        c = np.zeros( (fixed_split_lin.shape[1]*2-1, fixed_split_lin.shape[2]*2-1) )
        
        for i in range( np.int32(np.round(fixed_split_lin.shape[0]/subsample_rate))): # check for int32-ng
            if np.max( fixed_split_lin[idx[i]], axis=None)>0:
                c=c+xcorr2( np.double(fixed_split_lin[idx[i]]), np.double(moving_split_lin[idx[i]]))
                
        shift_yx = np.unravel_index(np.argmax(c), c.shape)
        yoffset = -np.array([(shift_yx[0]+1-fixed_split_lin.shape[1])/resize_factor])
        xoffset = -np.array([(shift_yx[1]+1-fixed_split_lin.shape[2])/resize_factor])
        idx_minxy = np.argmin(np.abs(xoffset) + np.abs(yoffset))
        tform = ski.transform.SimilarityTransform( translation=[xoffset[idx_minxy], yoffset[idx_minxy]])
    
        logging.debug(f'transform calculated for {infile} to {fixed_file} Applying...')
        moving_aligned=np.zeros_like(moving)
        
        for i in range(moving.shape[0]):
            moving_aligned[i,:,:] = np.expand_dims( ski.transform.warp((np.squeeze(moving[i,:,:])),
                                                   tform,
                                                   preserve_range=True),
                                                   0)
        #,output_shape=(moving.shape[1],moving.shape[2])),0)# check if output size specification is necessary -ng
      
        moving_aligned=uint16m(moving_aligned)
        moving_aligned_full=moving_aligned.copy()

        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, moving_aligned_full)
        logging.debug(f'done writing {outfile}')




    
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

    regcycle_ski( infiles=args.infiles,  
                  outfiles=args.outfiles,
                  template=args.template, 
                  stage=args.stage, 
                  cp=cp )
    
    logging.info(f'done processing output to {outdir}')