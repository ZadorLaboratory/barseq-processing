#!/usr/bin/env python
#
# https://imageio.readthedocs.io/en/v2.9.0/userapi.html
#    2.36.1 
#
import argparse
import logging
import os
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *

import scipy as sp
import numpy as np
import imageio.v2 as imageio
import tifffile as tf


def image_test(infiles):
    
    for infile in infiles:
        tif = tf.TiffFile(infile) 
        n_pages = len(tif.pages)
        logging.info(f'handling {infile} {n_pages} pages/channels')
        page = tif.pages[0]
        #page.shape  ->  (1080, 1280)
        # from a 1280 (w) x 1080 (h) microscopy image.  
        #  1,2,3 = None, Inch, Centimeter

        for i, page in enumerate( tif.pages ):
            XResolution = page.tags['XResolution'].value
            YResolution = page.tags['YResolution'].value
            ImageWidth = page.tags['ImageWidth'].value
            # height
            ImageLength = page.tags['ImageLength'].value
            shape = page.shape
            sizes = page.sizes
            s = f'{infile} page [{i}]\n   TIFF format info: '
            for d in sizes.keys():
                s += f'{d}={sizes[d]} '
            s += f'shape={shape}'
            print(s)
            #for t in page.tags:
            #    print(t)
            logging.debug(f'getting image as array...')
            a = page.asarray()
            shape = a.shape
        
            s = "   Numpy array info:"
            s += f'shape={shape}'
            print(s)

            #  height, width 
            (h,w) = shape
            
            s = ''
            logging.debug(f'calculating image sum...')
            imgsum = a.sum()
            #upperleft [ rows, columns ] 
            logging.debug(f'subsetting quadrants...')
            hi = int(h/2)
            wi = int(w/2)
            ul = a[  :hi,   :wi ]
            ur = a[  :hi, wi:   ]
            ll = a[hi:,     :wi ]
            lr = a[hi:,   wi:   ]
            logging.debug(f'calculating means...')
            s += f'   mean(UL) = {int(ul.mean())} '
            s += f'   mean(UR) = {int(ur.mean())} '
            s += f'   mean(LL) = {int(ll.mean())} '
            s += f'   mean(LR) = {int(lr.mean())}\n'

            logging.debug(f'calculating relative intensities...')
            s += f'   rel(UL) = { int( (ul.sum() / imgsum) * 100 ) } '
            s += f'   rel(UR) = { int( (ur.sum() / imgsum) * 100 ) } '
            s += f'   rel(LL) = { int( (ll.sum() / imgsum) * 100 ) } '
            s += f'   rel(LR) = { int( (lr.sum() / imgsum) * 100 ) }\n'

            #sp.stats.hmean(lr, axis=None)
            logging.debug(f'calculating harmonic means...')
            s += f'   hmean(UL) = {int(sp.stats.hmean(ul, axis=None) )} '
            s += f'   hmean(UR) = {int(sp.stats.hmean(ur, axis=None) )} '
            s += f'   hmean(LL) = {int(sp.stats.hmean(ll, axis=None) )} '
            s += f'   hmean(LR) = {int(sp.stats.hmean(lr, axis=None) )}\n'                      
            print(s)



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

    
    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None, 
                    type=str, 
                    help='outdir. output base dir if not given.')


    
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

      
    outdir = os.path.abspath('./')
    if args.outdir is not None:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    image_test( infiles=args.infiles )
    
    logging.info(f'done processing output to {outdir}')