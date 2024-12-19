#!/usr/bin/env python
#
# Script to use MIST (via ImageJ/FIJI with MIST plugins to stitch images.
#
#    1.     --filenamePattern <string containing filename pattern>
#    2.     --filenamePatternType <ROWCOL/SEQUENTIAL>
#    3a.    --gridOrigin <UL/UR/LL/LR>  -- Required only for ROWCOL or SEQUENTIAL
#    3b.    --numberingPattern <VERTICALCOMBING/VERTICALCONTINUOUS/HORIZONTALCOMBING/HORIZONTALCONTINUOUS> 
#                -- Required only for SEQUENTIAL
#    4.     --gridHeight <#>
#    5.     --gridWidth <#>
#    6.     --imageDir <PathToImageDir>
#    7a.     --startCol <#> -- Required only for ROWCOL
#    7b.     --startRow <#> -- Required only for ROWCOL
#    7c.     --startTile <R> -- Required only for SEQUENTIAL
#    8.     --programType <AUTO/JAVA/FFTW> -- Highly recommend using FFTW
#    9.     --fftwLibraryFilename libfftw3f.dll -- Required for FFTW program type
#    9a.     --fftwLibraryName libfftw3f -- Required for FFTW program type
#    9b.     --fftwLibraryPath <path/to/library> -- Required for FFTW program type

#    Example execution:
#    java.exe -jar MIST_-2.1-jar-with-dependencies.jar --filenamePattern img_r{rrr}_c{ccc}.tif --filenamePatternType ROWCOL --gridHeight 5 --gridWidth 5 --gridOrigin UR --imageDir C:\Users\user\Downloads\Small_Fluorescent_Test_Dataset\image-tiles --startCol 1 --startRow 1 --programType FFTW --fftwLibraryFilename libfftw3f.dll --fftwLibraryName libfftw3f --fftwLibraryPath C:\Users\user\apps\Fiji.app\lib\fftw
#
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

import pandas as pd
import numpy as np

import tifffile
import PIL
from PIL import Image, ImageSequence
#from ScanImageTiffReader import ScanImageTiffReader, ScanImageTiffReaderContext

import imagej
import skimage


def stitch_mist_test( infiles, outdir, image_type='geneseq', cp=None):
    '''
    image_type = [ geneseq | bcseq | hyb ]
    
    stitch images from the Small_Flourescent_Test_Dataset that comes with MIST
    
    '''
    if cp is None:
        cp = get_default_config()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
    
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_types = cp.get('barseq','image_types').split(',')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'image_types={image_types} channels={image_channels}')
    output_dtype = cp.get('stitch','output_dtype')

    fiji_path = os.path.abspath( os.path.expanduser( cp.get('mist','fiji_path')))
    logging.debug(f'fiji_path = {fiji_path}')
    # config with imageJ
    logging.info(f'initializing fiji at {fiji_path}...')
    ij=imagej.init(fiji_path)
    logging.debug(f'fiji version = {ij.getVersion()}')

def stitch_mist( infiles, outdir, image_type='geneseq', cp=None):
    '''
    infiles:    list of images for a single position to be stitched a single image. 
        e.g. MAX_Pos1_000_000.denoised.tif MAX_Pos1_000_001.denoised.tif ...
    output:    MAX_Pos1_stitched.denoised.tif 
    
    image_type = [ geneseq | bcseq | hyb ]
    

    '''
    if cp is None:
        cp = get_default_config()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
    
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_types = cp.get('barseq','image_types').split(',')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'image_types={image_types} channels={image_channels}')
    output_dtype = cp.get('stitch','output_dtype')
 
    logging.debug(f'output_dtype={output_dtype}')
       
    models = []
    if image_type in image_types:
        logging.debug(f'handling image_type={image_type}')
        for probe in model_channels:
            name = model_stem+probe
            logging.debug(f'loading model {name} from {basedir}')
            models.append( N2V(config=None, name=model_stem+probe, basedir=basedir) )
        
    logging.debug(f'got {len(models)} N2V models for {model_channels}')    
    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]}')




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

    stitch_mist_test( infiles=args.infiles, 
                 outdir=outdir, 
                 cp=cp )
    
    logging.info(f'done processing output to {outdir}')