#!/usr/bin/env python
#
# Print useful image information for one or more files. 
#  
import argparse
import logging
import os
import pprint
import shutil
import sys

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def get_image_info(infiles):
    '''
    
    '''
    for infile in infiles:
        logging.debug(f'handling {infile}')
        infile_image = read_image(infile)
        #logging.debug(f'{infile_image}')
        if len(infile_image.shape) == 2:
            print(f'{infile}: Flat image. {infile_image.shape} sum={infile_image.sum()}')

        elif len(infile_image.shape) >= 3:
            n_channels = len(infile_image)
            print(f'{infile}: {n_channels} channels.')
            for i in range(n_channels):
                chi = infile_image[i]
                shp = chi.shape
                dtp = str( chi.dtype )
                csum = chi.sum()
                print(f'    [{i}] {shp} sum={csum} {dtp}')
          

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

    parser.add_argument('infiles' ,
                        metavar='infiles', 
                        type=str,
                        nargs='+',
                        default=None, 
                        help='Image files. ')     

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    get_image_info(args.infiles)