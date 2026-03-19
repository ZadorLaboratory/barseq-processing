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
#  writing metadata:  https://forum.image.sc/t/python-tifffile-ome-full-metadata-support/56526/10 
#
#
#
#
#
#
import argparse
import json
import logging
import os
import re
import sys


import datetime as dt

from configparser import ConfigParser
from joblib import dump, load

import numpy as np

import ashlar
from ashlar import filepattern, reg, thumbnail
from tifffile import imread, imwrite, TiffFile, TiffWriter

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *

# Disable annoying Java logging...
logging.getLogger('kivy').setLevel(logging.WARNING)

def process_axis_flip(reader, flip_x, flip_y):
    metadata = reader.metadata
    # Trigger lazy initialization.
    _ = metadata.positions
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    metadata._positions *= [sy, sx]

def make_ashlar_pattern(image_pattern, basename, extension):
    '''
    XXX: This is currently brittle and ad-hoc. Need to abstract later...
    
    image_pattern = MAX_Pos{pos}_{col:03}_{row:03}$
    basename = MAX_Pos1_002_003

    prefix = MAX_Pos1
    ashlar_pattern = MAX_Pos1_{col:03}_{row:03}.tif

    return prefix, ashlar_pattern
    
    '''
    logging.debug(f'inbound image_pattern={image_pattern} basename={basename} ext={extension}')
    pi = image_pattern.find('{pos}') + 5
    # extract needed suffix, and remove end-of-string symbol
    suffix = image_pattern[pi:-1]
    # build new specific ashlare_pattern by finding initial match to full first number 
    # which could be multiple digits...
    
    pattern = image_pattern.replace('.', '\\.')
    pattern = pattern.replace('(', '\\(')
    pattern = pattern.replace(')', '\\)')
    regex = re.sub(r'{([^:}]+)(?:[^}]*)}', r'(?P<\1>.*?)', pattern)
    logging.debug(f'filtered regex={regex}')
    m = re.match(regex, basename)
    if m:
        gd = m.groupdict()
        logging.debug(f'groupdict={gd}')
        row = int(gd['row'])
        col = int(gd['col'])
        pos = int(gd['pos'])
        logging.debug(f'pos={pos} row={row} col={col}')        
        (b,e) = m.span('pos')
        prefix = basename[:e]
        logging.debug(f'prefix={prefix}')
        ashlar_pattern = f'{prefix}{suffix}.{extension}'
        logging.debug(f'made ashlar_pattern = {ashlar_pattern} prefix={prefix}')
    else:
        logging.error('unable to parse basename to pattern...')
    return ashlar_pattern, prefix 


class SingleTiffWriter:
    '''
    Test setup. Only writes first channel from mosaic. 
    
    '''

    def __init__(self, mosaic, outfile, verbose=False):
        self.mosaic = mosaic
        self.outfile = outfile
        self.verbose = verbose

    def run(self):
        pixel_size = self.mosaic.aligner.metadata.pixel_size
        resolution_cm = 10000 / pixel_size
        v = ashlar._version.get_versions()['version']
        software = f"Ashlar v{v}"
        images = []
        for ci, channel in enumerate(self.mosaic.channels):
            #channel = 0
            if self.verbose:
                logging.info(f"Assembling channel {channel}:")
            img = self.mosaic.assemble_channel(channel)
            img = uint16m(img)
            images.append(img)
            img = None
            logging.debug(f'Added channel {channel} to image list.')

        fullimage = np.dstack(images)
        logging.debug(f'dstack() -> {fullimage.shape}')
        # produces e.g. shape = ( 3200,3200,5)
        fullimage = np.rollaxis(fullimage, -1)
        logging.debug(f'rollaxis() -> {fullimage.shape}')
        # produces e.g. shape = ( 5, 3200, 3200)  

        with TiffWriter(self.outfile, bigtiff=True) as tiff:
            tiff.write(
                data=fullimage,
                software=software.encode("utf-8"),
                resolutionunit='centimeter',
                resolution=(resolution_cm, resolution_cm),
                photometric="minisblack",
            )
        logging.debug(f'done.')


def stitch_ashlar( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'stitch'

    # We know arity is single, so we can grab the outfile to check paths.
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
    
    # get params
    microscope_profile = cp.get('experiment','microscope_profile')
    pixel_size = float( cp.get(microscope_profile, 'pixel_size') )
    
    channels = [ c.strip() for c in cp.get(stage,'channels').split(',') ]
    channels = [ int(c) for c in channels ]
    overlap = float( cp.get('tile','horizontal_overlap') )
    flip_y = cp.getboolean('ashlar','flip_y')
    flip_x = cp.getboolean('ashlar', 'flip_x')
    logging.debug(f'microscope_profile={microscope_profile} flip_x={flip_x} flip_y={flip_y} channels={channels}')
    image_pattern=cp.get('barseq','image_pattern')
    image_ext = cp.get('barseq','image_ext')
    #pattern= cp.get('ashlar','pattern')        
    (dirpath, base, label, ext) = split_path( infiles[0] )
    logging.debug(f'image_pattern={image_pattern} base={base} image_ext={image_ext}')
    (pattern, prefix ) = make_ashlar_pattern( image_pattern, base, ext )
    logging.debug(f'generated ashlar pattern={pattern} prefix={prefix}')

    logging.info(f'stitch indir = {dirpath} pattern={pattern} to {outdir} ...')
    fpr = filepattern.FilePatternReader(
                dirpath,
                pattern=pattern,
                overlap=overlap,
                pixel_size=pixel_size
               )

    logging.debug(f'reader: path={fpr.path} pattern={fpr.pattern} overlap={fpr.overlap} pixel_size={fpr.metadata.pixel_size}')
    logging.debug(f'doing axis flip flip_x={flip_x} flip_y={flip_y} ')
    process_axis_flip(fpr, flip_x, flip_y)
    logging.debug(f'making edge_aligner...')
    edge_aligner = reg.EdgeAligner(fpr, do_make_thumbnail=True, verbose=True )
    edge_aligner.make_thumbnail()     
    outthumb = f'{outdir}/{prefix}_thumb.tif'
    #logging.debug(f'writing thumbnail {outthumb}...')
    #imwrite( outthumb, edge_aligner.reader.thumbnail )   
    logging.debug(f'running edge_aligner...')
    edge_aligner.run()
    logging.debug(f'ran edge_aligner...')
    mshape = edge_aligner.mosaic_shape
    logging.debug(f'mosaic shape = {mshape}')
    mosaic = reg.Mosaic( edge_aligner, mshape, verbose=True )

    # POS.stitched.tif
    # Image stitched according to Ashlar results.  
    (outdir, base, label, ext) = split_path(outfile)
    out_tiff = os.path.join(outdir, f'{base}.stitched.tif')
    writer = SingleTiffWriter( mosaic, out_tiff , verbose=True)
    writer.run()
    logging.debug(f'wrote {out_tiff}(s) ...')

    # POS.positions.tsv
    # This is intended to be human readable.
    out_tsv = os.path.join(outdir, f'{base}.positions.tsv' )   
    df = pd.DataFrame(edge_aligner.positions)
    logging.info(f'writing positions to {out_tsv}')
    df.to_csv(out_tsv, sep='\t') 

    # POS.tform_original.joblib
    # This is primary outfile, used to determine completion.
    infile_names = [ os.path.split(ifn)[1] for ifn in infiles ] 
    Tfull={}
    T={}
    for i, tilename in enumerate(infile_names):
        T['position']=[df.iloc[i,1],df.iloc[i,0]]
        T['grid']=[0,0]
        Tfull[tilename]=T
        T={}

    dump(Tfull, outfile )
    logging.info(f'writing position dict to {outfile}')

    # POS.tform_original.json
    # This is intended to be human readable.
    out_json = os.path.join(outdir, f'{base}.tform_original.json' )
    with open(out_json, 'w') as f:
        json.dump(Tfull, f, indent=4 )
 




# NOTEBOOK CODE
def merge_ashlar_results(pth, transform_rescale_factor=0.5, num_c=4):
    """
    Stitching function:
    1. ASHLAR based stitching results are encoded in a global dictionary 
    2. For each position (slice)-position of tiles and their names is saved as a sub-dictionary
    3. One final dictionary with positions as keys is stored as tforms_original file
    4. Calls function to rescale transformation
    """ 
    [folders,pos,_,_]=get_folders(pth)
    unique_pos=nsort(np.unique(pos))
    folder_names=np.array(folders)
    Texp={}
    Tfull={}
    for n_pos in unique_pos:
        T={}
        df=pd.read_csv(os.path.join(pth,'MAX_'+n_pos+'.positions.tsv'), sep='\t')
        pos_id=np.array([i for i,name in enumerate(pos) if name==n_pos])
        for ids in pos_id:
            tilename=folder_names[ids]+'.tif'
            T['position']=[df.iloc[ids,2],df.iloc[ids,1]]
            T['grid']=[0,0]
            Tfull[tilename]=T
            T={}
        Texp[n_pos]=Tfull
        Tfull={}
    dump(Texp,os.path.join(pth,'processed','tforms_original.joblib'))
    sx,sy=rescale_transformation(pth,folders,unique_pos,pos,transform_rescale_factor,num_c)
    return sx,sy

def rescale_transformation(pth,folders,unique_pos,pos,rescale_factor=0.5,num_c=4):
    """
    Stitching function:
    1. Reads the original transformation dictionary
    2. Downscales the coordinates as per rescale_factor and write in as new key-value pairs per tile in the original dictionary
    3. Writes the modified dictionary
    
    """ 

    folder_names=np.array(folders)
    T=load(os.path.join(pth,'processed','tforms_original.joblib'))
    sx=[]
    sy=[]
    for n_pos in unique_pos:
        pos_id=np.array([i for i,name in enumerate(pos) if name==n_pos])
        for ids in pos_id:
            tilename=folder_names[ids]+'.tif'
            T[n_pos][tilename]["ref_pos"]=[T[n_pos][tilename]["position"][0]*rescale_factor,T[n_pos][tilename]["position"][1]*rescale_factor]
            sx.append(T[n_pos][tilename]["ref_pos"][0])
            sy.append(T[n_pos][tilename]["ref_pos"][1])
        
    #pprint.pprint(T)
    dump(T,os.path.join(pth,'processed','tforms_rescaled'+str(rescale_factor).replace('.','p')+'.joblib'))
    return sx,sy



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

    stitch_ashlar( infiles=args.infiles, 
                   outfiles=args.outfiles, 
                   cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')