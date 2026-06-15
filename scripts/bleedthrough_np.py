#!/usr/bin/env python
#
import argparse
import logging
import os
import sys

import datetime as dt

import numpy as np
import tifffile as tf

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def bleedthrough_np( infiles, outfiles, stage=None, cp=None):
    '''
    
    '''
    if cp is None:
        cp = get_default_config()
        
    if stage is None:
        stage = 'bleedthrough'
    logging.debug(f'stage={stage}')

    # Get parameters for all steps.
    image_type = cp.get(stage, 'image_type')
    output_dtype = cp.get( stage,'output_dtype')
    channel_names =  get_config_list(cp, image_type, 'channels')
    select_channels = get_config_list(cp, stage, 'channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    num_c = len(select_channels)

    mode = get_config_list(cp, stage, 'modes')
    mode = mode[0]
    output_dtype = cp.get( stage,'output_dtype')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')

    chprofile_file = cp.get(microscope_profile, f'channel_profile_{mode}')
    chprofile_path = os.path.join(resource_dir, chprofile_file)
    chprofile = load_df(chprofile_path, as_array=True)
    num_prof_channels = len(chprofile)
    logging.debug(f'chprofile_file={chprofile_file} num_channels={num_prof_channels}')
    image_types = cp.get('barseq','image_types').split(',')
    num_channels = len(chprofile)
    logging.debug(f'chprofile_file={chprofile_file} num_channels={num_channels}')

    for i, infile in enumerate( infiles ):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        I = read_image(infile)
        I = I.copy()
        Icorrected = np.zeros_like(I)
        Ishifted2 = np.float64( I[0:num_channels,:,:] )
        I_rem=I[ num_channels: , : , : ]
        A = np.transpose(chprofile)
        B = np.reshape( Ishifted2 , (num_channels, -1), order='F')
        I_solved = np.linalg.solve( A, B ) 
        Icorrected=np.reshape( I_solved , 
                                ( num_channels, Ishifted2.shape[1], Ishifted2.shape[2]),
                                order='F')       
        Icorrected=uint16m(Icorrected)
        Icorrected=np.append(Icorrected, I_rem, axis=0)

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

    bleedthrough_np( infiles=args.infiles, 
                     outfiles=args.outfiles,
                     stage=args.stage, 
                     cp=cp )
    (outdir, file) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')