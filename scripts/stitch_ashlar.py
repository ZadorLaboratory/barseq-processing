#!/usr/bin/env python
#
#  https://academic.oup.com/bioinformatics/article/38/19/4613/6668278#401879063
#
# manual code to deal with stitching, but fileseries ont filepattern :-(
#   https://github.com/labsyspharm/ashlar/issues/166 
#
#  https://forum.image.sc/t/ashlar-how-to-pass-multiple-images-to-be-stitched/49864/69?page=3
#
#
#  Undocumented command line hack...
#  ashlar 'fileseries|/path/to/images|pattern=img_{series:3}.tif|width=5|height=3|pixel_size=0.3|overlap=0.1'
#
# 
#
import argparse
import logging
import os
import sys

import datetime as dt

from configparser import ConfigParser

import numpy as np

import ashlar
from ashlar import filepattern, reg
from tifffile import imread, imwrite, TiffFile, TiffWriter

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *


def process_axis_flip(reader, flip_x, flip_y):
    metadata = reader.metadata
    # Trigger lazy initialization.
    _ = metadata.positions
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    metadata._positions *= [sy, sx]


class SingleTiffWriter:
    '''
    Test setup. Only writes first channel from mosaic. 
    
    '''

    def __init__(self, mosaic, outfile, verbose=False):
        self.mosaic = mosaic
        self.outpath = outfile
        self.verbose = verbose

    def run(self):
        pixel_size = self.mosaic.aligner.metadata.pixel_size
        resolution_cm = 10000 / pixel_size
        v = ashlar._version.get_versions()['version']
        software = f"Ashlar v{v}"
        #for ci, channel in enumerate(self.mosaic.channels):
        channel = 0
        if self.verbose:
            logging.info(f"Assembling channel {channel}:")
        img = self.mosaic.assemble_channel(channel)
        img = uint16m(img)
        with TiffWriter(self.outpath, bigtiff=True) as tiff:
            tiff.write(
                data=img,
                software=software.encode("utf-8"),
                resolution=(resolution_cm, resolution_cm, "centimeter"),
                photometric="minisblack",
            )
        img = None

def stitch_ashlar( infiles, outdir, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    
    microscope_profile = cp.get('experiment','microscope_profile')
    pixel_size = float( cp.get(microscope_profile, 'pixel_size') )
    
    #horizontal_overlap=23
    #vertical_overlap=23
    overlap = float( cp.get('tile','horizontal_overlap') )
    flip_y = cp.getboolean('ashlar','flip_y')
    flip_x = cp.getboolean('ashlar', 'flip_x')
    pattern= cp.get('ashlar','pattern')
        
    logging.debug(f'microscope_profile={microscope_profile} flip_x={flip_x} flip_y={flip_y}')
    
    fpr = filepattern.FilePatternReader(
                '/Users/hover/project/barseq/run_barseq/BC726126.5.out/regcycle/bcseq01/',
                pattern=pattern,
                overlap=overlap,
                pixel_size=pixel_size
               )
    logging.debug(f'reader: path={fpr.path} pattern={fpr.pattern} overlap={fpr.overlap} pixel_size={fpr.metadata.pixel_size}')

    logging.debug(f'doing axis flip flip_x={flip_x} flip_y={flip_y} ')
    process_axis_flip(fpr, flip_x, flip_y)

    logging.debug(f'making edge_aligner...')
      
    edge_aligner = reg.EdgeAligner(fpr, do_make_thumbnail=False, verbose=True )
    
    logging.debug(f'running edge_aligner...')
    edge_aligner.run()
    logging.debug(f'ran edge_aligner...')
    
    mshape = edge_aligner.mosaic_shape
    logging.debug(f'mosaic shape = {mshape}')
    
    mosaic = reg.Mosaic( edge_aligner, mshape, verbose=True )

    outfile = f'{outdir}/MAX_Pos1-flipped.ome.tif'

    #writer = reg.TiffListWriter([mosaic] )
    #    mosaics, output_path_format, verbose=not quiet, **writer_args
    writer = SingleTiffWriter( mosaic, outfile , verbose=True)
    writer.run()
    logging.debug(f'wrote {outfile}? ...')


    #mosaics = []
    #for j in range(1, 2):
    #    aligners.append(
    #        reg.LayerAligner(readers[j], aligners[0], channel=j, filter_sigma=15, verbose=True)
    #    )
    #    aligners[j].run()
    #    print("aligners[0].mosaic_shape", aligners[0].mosaic_shape, aligners[0])
    #    mosaic = reg.Mosaic(
    #        aligners[j], aligners[0].mosaic_shape, None, **mosaic_args
    #    )
    #    mosaics.append(mosaic)
    
    
    #aligner0 = reg.EdgeAligner(readers[0], channel=0, filter_sigma=15, verbose=True)
    #aligner0.run()
    
    #mosaic_args = {}
    #mosaic_args['verbose'] = True
    #mosaic_args['flip_mosaic_y'] = True
    #aligners = []
    #aligners.append(aligner0)
    
    #mosaics = []
    #for j in range(1, 2):
    #    aligners.append(
    #        reg.LayerAligner(readers[j], aligners[0], channel=j, filter_sigma=15, verbose=True)
    #    )
    #    aligners[j].run()
    #    print("aligners[0].mosaic_shape", aligners[0].mosaic_shape, aligners[0])
    #    mosaic = reg.Mosaic(
    #        aligners[j], aligners[0].mosaic_shape, None, **mosaic_args
    #    )
    #    mosaics.append(mosaic)
    #print(type(mosaic))
    #writer = reg.PyramidWriter(mosaics, r"/home/kuldeep/Downloads/Inferencing_with_Button/16*16.ome.tif",
    #                           verbose=True)
    #writer.run()




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

    stitch_ashlar( infiles=args.infiles, 
                   outdir=outdir, 
                   cp=cp )
    
    logging.info(f'done processing output to {outdir}')