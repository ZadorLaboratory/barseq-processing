#!/usr/bin/env python
#
# Create RGB output.
import argparse
import logging
import os
import sys
import datetime as dt

import numpy as np

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def rbgbout_hyb( infiles, outfiles, stage=None, cp=None):
    '''
    
    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'rgbout-hyb'

    # Get parameters for all steps.
    mode = get_config_list(cp, stage, 'modes')
    mode = mode[0]
    output_dtype = cp.get( stage,'output_dtype')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')

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
        Ishifted = regchannels_ski_single(image=I, channel_shift=chshift)
        logging.debug(f'Done with regchannels. mode={mode} n_channnels={len(Ishifted)}')

        logging.debug(f'Do bleedthrough...')       
        Icorrected = bleedthrough_np_single(Ishifted, chprofile)
        logging.debug(f'Done bleedthrough. mode={mode} n_channnels={len(Icorrected)}')
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

    rbgbout_hyb( args.infiles, 
                args.outfiles, 
                stage=args.stage, 
                cp=cp)

    (outdir, file) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')