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
        n_channels = len(infile_image)
        print(f'{infile}: {n_channels} channels.')
        for i in range(n_channels):
            chi = infile_image[i]
            shp = chi.shape
            dtp = str( chi.dtype )
            print(f'    [{i}] {shp} {dtp}')
          

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

    parser.add_argument('infile1' ,
                        metavar='infile1', 
                        type=str,
                        help='Image files. ')     

    parser.add_argument('infile2' ,
                        metavar='infile2', 
                        type=str, 
                        help='Image files. ')

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    ident, msg, min_similarity = do_compare_images(args.infile1, args.infile2)
    print(f'min similarity={min_similarity}')
    print(msg)