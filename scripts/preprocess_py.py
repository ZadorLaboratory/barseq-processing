#!/usr/bin/env python
#
#
# Combination background, regchannels, and bleedthrough. 
#
#
import argparse
import logging
import os
import sys
import datetime as dt

import cv2
import numpy as np
import skimage as ski
import tifffile as tf

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def background_cv2_single(image, radius, num_c=4):
    '''
    
    '''
    I=image.copy()
    I_filtered=np.zeros_like(I)
    # save any extra channels. 
    I_rem=I[num_c:,:,:]
    I=I[0:num_c,:,:]
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
    for i in range(len(I)):
        bck=cv2.morphologyEx(I[i,:,:], cv2.MORPH_OPEN, kernel= k)
        I_filtered[i,:,:]=I[i,:,:]-np.expand_dims(bck,0)
    # restore extra channel(s)
    I_filtered[num_c:,:,:]=I_rem    
    I_filtered=uint16m(I_filtered)
    return I_filtered

def regchannels_ski_single(image, channel_shift, is_affine):
    '''
    
    '''
    n_channels = len(channel_shift)
    I=image.copy()
    Ishifted=np.zeros_like(I)
    # Save extra channel(s)
    I_rem=I[n_channels:,:,:]
    I=I[0:n_channels,:,:]
    for i in range(channel_shift.shape[0]):
        if is_affine:
            # refine this later on-ng
            tform=channel_shift[i] 
        else:
            # remember this takes -shifts-ng  
            tform=ski.transform.SimilarityTransform(translation = -channel_shift[i,:])              
        It=ski.transform.warp(np.squeeze(I[i,:,:]), 
                                    tform, 
                                    preserve_range=True, 
                                    output_shape=(I.shape[1],I.shape[2]))
        Ishifted[i,:,:]=np.expand_dims(It,0)
    # Put back extra channel(s)
    Ishifted[n_channels:,:,:]=I_rem    
    Ishifted=uint16m(Ishifted)
    return Ishifted

def bleedthrough_np_single(image, chprofile):

    n_channels = len(chprofile)
    I=image.copy()
    Icorrected=np.zeros_like(I)
    Ishifted2 = np.float64( I[0:n_channels,:,:] )
    I_rem=I[n_channels:,:,:]
    A = np.transpose(chprofile)
    B = np.reshape( Ishifted2 , (n_channels, -1), order='F')
    I_solved = np.linalg.solve( A, B ) 
    Icorrected=np.reshape( I_solved , 
                            ( n_channels, Ishifted2.shape[1], Ishifted2.shape[2]),
                            order='F')       
    Icorrected=uint16m(Icorrected)
    Icorrected=np.append(Icorrected, I_rem, axis=0)
    return Icorrected



def preprocess_py( infiles, outfiles, stage=None, cp=None):
    '''
    Perform background, regchannels, and bleedthrough correction.
    
    hyb denoised    = 6 channels
    geneseq denoised = 5 channels. 
    [geneseq]
    channels=G,T,A,C,BF

    [hyb]
    channels=GFP,YFP,TxRed,Cy5,DAPI,BF
       
    num_initial_c=5,
    num_later_c=4
    num_c=4

    num_c_hyb=5

    geneseq: 
    hyb: num_c=num_c_hyb

    preprocess():
    process_geneseq_cycle(  num_initial_c=num_initial_c,
                            num_later_c=num_later_c,
                            num_c=num_c)
    process_hyb_cycle(      num_c=num_c_hyb,
                            is_affine=is_affine)
    
    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'preprocess-geneseq'

    # Get parameters for all steps.
    mode = get_config_list(cp, stage, 'modes')
    mode = mode[0]
    radius = int(cp.get('cv2','radius'))
    output_dtype = cp.get( stage,'output_dtype')
    logging.debug(f'output_dtype={output_dtype} radius = {radius} ')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')
    chshift_file = cp.get(microscope_profile,f'channel_shift_{mode}')
    chshift_path = os.path.join(resource_dir, chshift_file)
    is_affine = cp.getboolean(stage,'is_affine')
    logging.debug(f'chshift_path = {chshift_path} is_affine={is_affine}')
    chshift = load_df(chshift_path, as_array=True)
    n_shift_channels = len(chshift)
    logging.debug(f'loaded channel shift. len={n_shift_channels} ')
    chprofile_file = cp.get(microscope_profile, f'channel_profile_{mode}')
    chprofile_path = os.path.join(resource_dir, chprofile_file)
    chprofile = load_df(chprofile_path, as_array=True)
    num_prof_channels = len(chprofile)
    logging.debug(f'chprofile_file={chprofile_file} num_channels={num_prof_channels}')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')                
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        # Background correction
        I = read_image( infile)      
        if radius > 0:
            Ibacksub = background_cv2_single(I, radius)
        else:
            Ibacksub = I       
        logging.debug(f'Done background. mode={mode} n_channnels={len(Ibacksub)}')
        
        logging.debug(f'Do regchannels...')
        # Regchannels
        # Register channels within each image. 
        IShifted = regchannels_ski_single(image, channel_shift)
        logging.debug(f'Done with regchannels. mode={mode} n_channnels={len(Ishifted)}')

        logging.debug(f'Do bleedthrough...')       
        ICorrected = bleedthrough_np_single(IShifted, chprofile)
        logging.debug(f'Done bleedthrough. mode={mode} n_channnels={len(ICorrected)}')
        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, Icorrected)
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

    preprocess_py( infiles=args.infiles, 
                     outfiles=args.outfiles, 
                     cp=cp )
    (outdir, file) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')