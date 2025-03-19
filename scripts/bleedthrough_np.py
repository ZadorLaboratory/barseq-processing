#!/usr/bin/env python
#
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

def bleedthrough_np( infiles, outdir, stage=None, cp=None):
    '''
    image_type = [ geneseq | bcseq | hyb ]

    is_affine=0
    num_initial_c=5 # initial channels in first round of sequencing to read
    num_later_c=4 # subsequent cycle channels to read
    num_c=4 # channels to perform bleedthrough-correction and channel alignment, 
    needs more elegant solution--ng
    subsample_rate=4
    do_coarse=0
    resize_factor=2
    block_size=128
    chshift_filename='chshift20x-20220218.mat'
    chprofile_filename='chprofile20x-50-30-20-40-20220218.mat'
    radius=31
    local=1

    def bleedthrough_linear(Is,num_c,config_pth,fname,pth,name,writefile):
        Is=Is.copy()
        Icorrected=np.zeros_like(Is)
        chprofile=scipy.io.loadmat(os.path.join(config_pth,fname))['chprofile20x']
        Ishifted2=np.float64(Is[0:num_c,:,:])
        I_rem=Is[num_c:,:,:]
        Icorrected=(np.reshape((np.linalg.solve(np.transpose(chprofile),((np.reshape(Ishifted2,(num_c,-1),order='F'))))),(num_c,Ishifted2.shape[1],Ishifted2.shape[2]),order='F'))
        Icorrected=uint16m(Icorrected)
        Icorrected=np.append(Icorrected,I_rem,axis=0)
        if writefile:
            tfl.imwrite(os.path.join(pth,name),Icorrected,photometric='minisblack')
        return Icorrected
        
        Ibcksub_shifted=channel_alignment(Ibcksub,chshift_filename,config_pth,pth,'bck_sub'+filename,num_c,is_affine,0) # last argument is to not to write intermediate file-ng CHANNEL ALIGN-NG
        Ibcksub_shifted_btcorr=bleedthrough_linear(Ibcksub_shifted,num_c,config_pth,chprofile_filename,pth,fixed_filename[i],1) # BLEEDTHROUGH CORRECTION-NG
        _,gtforms_ind=geneseq_cycle_alignment_block_correlation(fixed_filename[i],templatename,subsample_rate,do_coarse,resize_factor,block_size,num_c,num_cr,pth,'aligned','aligned'+filename)
        gtforms.append(gtforms_ind)

    '''
    if cp is None:
        cp = get_default_config()
        
    if stage is None:
        stage = 'bleedthrough'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')


    image_types = cp.get('barseq','image_types').split(',')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')
    chprofile_file = cp.get(microscope_profile,'channel_profile')
    chprofile_path = os.path.join(resource_dir, chprofile_file)
    chprofile = load_df(chprofile_path, as_array=True)
    num_channels = len(chprofile)
    logging.debug(f'chprofile_file={chprofile_file} num_channels={num_channels}')

    for infile in infiles:
        (dirpath, base, ext) = split_path(os.path.abspath(infile))
        logging.debug(f'handling {infile}')

        # I=tf.imread(infile)
        I = read_image(infile)
        I=I.copy()
        Icorrected=np.zeros_like(I)
        Ishifted2 = np.float64( I[0:num_channels,:,:] )
        I_rem=I[num_channels:,:,:]
        A = np.transpose(chprofile)
        B = np.reshape( Ishifted2 , (num_channels, -1), order='F')
        I_solved = np.linalg.solve( A, B ) 
        Icorrected=np.reshape( I_solved , 
                                ( num_channels, Ishifted2.shape[1], Ishifted2.shape[2]),
                                order='F')
        
        #Icorrected=(np.reshape((np.linalg.solve(np.transpose(chprofile),((np.reshape(Ishifted2,(num_channels,-1),order='F'))))),(num_channels,Ishifted2.shape[1],Ishifted2.shape[2]),order='F'))
        
        Icorrected=uint16m(Icorrected)
        Icorrected=np.append(Icorrected, I_rem, axis=0)

        logging.debug(f'done processing {base}.{ext} ')
        outfile = f'{outdir}/{base}.{ext}'
        #tf.imwrite(outfile, Icorrected, photometric='minisblack')
        write_image(outfile, Icorrected )
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

    bleedthrough_np( infiles=args.infiles, 
                    outdir=outdir, 
                    cp=cp )
    
    logging.info(f'done processing output to {outdir}')