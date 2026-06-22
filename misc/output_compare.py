#!/usr/bin/env python
#
# Compare MATLAB/Pybarseq output to Python pipeline output. 
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

N_GENESEQ_CYCLES = 7
N_HYB_CYCLES = 1

TILENAMES = [ 
        'MAX_Pos1_000_000',
        'MAX_Pos1_000_001',
        'MAX_Pos1_001_000',
        'MAX_Pos1_001_001',
        'MAX_Pos2_000_000',
        'MAX_Pos2_000_001',
        'MAX_Pos2_001_000',
        'MAX_Pos2_001_001'
]

PBS_PREFIXES = ['original/',
                'backsub/backsub',
                'chalign/chalign',
                'bleedthrough/bleedthrough',
                'aligned/aligned',
]

BSP_PREFIXES = ['denoised',
                'background',
                'regchannels',
                'bleedthrough',
                'regcycle'
                ]


def do_compare_output(outdir1, outdir2):
    '''
    outdir1  pybarseq
    outdir2  barseq-processing
    '''
    outdir1 = os.path.abspath(outdir1)
    outdir2 = os.path.abspath(outdir2)

    identical = True
    # Handle preprocessing
    for i, pbs_prefix in enumerate(PBS_PREFIXES):
        if identical:
            bsp_prefix = BSP_PREFIXES[i]
            for gcycle in list( range(1, N_GENESEQ_CYCLES + 1)):
                fname = f'n2vgeneseq0{gcycle}.tif'
                for tilename in TILENAMES:
                    file1 = f'{outdir1}/processed/{tilename}/{pbs_prefix}{fname}'
                    rel1 = os.path.relpath(file1)
                    file2 = f'{outdir2}/{bsp_prefix}/geneseq0{gcycle}/{tilename}.tif'
                    rel2 = os.path.relpath(file2)
                    logging.info(f'comparing {file1} and {file2}')
                    ident, msg, min_sim = do_compare_images(file1, file2)
                    if ident:
                        print( f' {rel1} == {rel2}' )
                    else:
                        identical = False
                        print( f' {rel1} != {rel2} min_sim = {min_sim}' )
            
            for hcycle in list(range(1, N_HYB_CYCLES + 1)):
                fname = f'n2vhyb0{hcycle}.tif'
                for tilename in TILENAMES:
                    file1 = f'{outdir1}/processed/{tilename}/{pbs_prefix}{fname}'
                    rel1 = os.path.relpath(file1)
                    file2 = f'{outdir2}/{bsp_prefix}/hyb0{hcycle}/{tilename}.tif'
                    rel2 = os.path.relpath(file2)
                    logging.info(f'comparing {file1} and {file2}')
                    ident, msg, min_sim = do_compare_images(file1, file2)
                    if ident:
                        print( f' {rel1} == {rel2}' )
                    else:
                        identical = False
                        print( f' {rel1} != {rel2} min_sim = {min_sim}' )
        else:
            print(f'outputs have diverged. Exitting.')
            sys.exit(1)

    # Handle segmentation. 
    # processed/<tile>/aligned/cell_mask_cyto3.tif
    #   vs
    # segment/hyb/<tile>.cp_mask_cyto3.tif 
    for hcycle in list(range(1, N_HYB_CYCLES + 1)):
        for tilename in TILENAMES:
            file1 = f'{outdir1}/processed/{tilename}/aligned/cell_inp2.tif'
            rel1 = os.path.relpath(file1)
            file2 = f'{outdir2}/segment/hyb/{tilename}.cellpose_input.tif'
            rel2 = os.path.relpath(file2)
            logging.info(f'comparing {file1} and {file2}')
            ident, msg, min_sim = do_compare_images(file1, file2)
            if ident:
                print( f' {rel1} == {rel2}' )
            else:
                identical = False
                print( f' {rel1} != {rel2} min_sim = {min_sim}' )

    for hcycle in list(range(1, N_HYB_CYCLES + 1)):
        for tilename in TILENAMES:
            file1 = f'{outdir1}/processed/{tilename}/aligned/cell_mask_cyto3.tif'
            rel1 = os.path.relpath(file1)
            file2 = f'{outdir2}/segment/hyb/{tilename}.cp_mask_cyto3.tif'
            rel2 = os.path.relpath(file2)
            logging.info(f'comparing {file1} and {file2}')
            ident, msg, min_sim = do_compare_images(file1, file2)
            if ident:
                print( f' {rel1} == {rel2}' )
            else:
                identical = False
                print( f' {rel1} != {rel2} min_sim = {min_sim}' )

    return identical


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

    parser.add_argument('outdir1' ,
                        metavar='outdir1', 
                        type=str,
                        help='MATLAB/Pybarseq output tree. ')     

    parser.add_argument('outdir2' ,
                        metavar='outdir2', 
                        type=str, 
                        help='barseq-processing output tree.')

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    identical = do_compare_output(args.outdir1, args.outdir2)
    if identical:
        print('Output trees are identical. ')
    else:
        print('Output trees differ. ')