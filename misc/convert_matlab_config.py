#!/usr/bin/env python
#
#

import argparse
import json
import logging
import os
import pprint
import sys
import traceback

from configparser import ConfigParser

import pandas as pd
import scipy

from scipy.io import loadmat
import h5py

STR_ANNOT = [ 'BC_name', 'config_name',
               'gene_name', 'grid_direction', 'pth', 'pth_prev'   ]
INT_ANNOT = [ 'BC_refch','ball_radius', 'batch_num', 'boxsize', 'ball_radius_bc', 'br', 
              'chradius', 'count_thresh', 'dilation_radius', 'err_corr_thresh', 'gene_refch',
               'hyb_reg_ch',  'intensity_scaling', 'nuclearch', 
               'radius', 'reg_ch', 'reg_cycle',  'soma_thresh',
               'startingsliceidx', 'subsample_rate' ]
FLOAT_ANNOT = [ 'overlap', 'pixelsize' , 'rescale_factor', 'rgb_intensity' ]
FLOAT_LIST_ANNOT = [ 'dapi_shift', ]
INT_LIST_ANNOT = [  'hybbgn', 'hybthresh', 'rolthresh', 'tilesize' ]
BOOL_ANNOT = [ 'do_coarse', 'is_barcoded' , 'is_mm' ,'local_registration', 'run_all',
                'use_predefined_thresh', 'use_previous_settings'   ] 
IGNORE_ANNOT = [ 'chprofile20x', 'chprofilehyb', 'chshift20x', 'chshifthyb', 
                'code_folder', 'codebook_name', 'codebookhyb_name'  ]

def dump_matlab_config(infile):
    '''
    converts top-level config file from .mat to .tsv
    '''
    infile = os.path.abspath(infile)
    dirpath, filename = os.path.split(infile)
    base, ext = os.path.splitext(filename)
    outfile = os.path.join(dirpath, f'{base}.json')
    logging.debug(f'dumping {infile} to {outfile}')

    output_dict = {}

    try:    
        annots = loadmat(infile)
        for label in STR_ANNOT:
            try:
                data = annots[label]
                val = str( data[0])
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}')

        for label in INT_ANNOT:
            try:
                data = annots[label]
                val = int( data[0])
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}')

        for label in FLOAT_ANNOT:
            try:
                data = annots[label]
                val = float( data[0])
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}')

        for label in FLOAT_LIST_ANNOT:
            try:
                data = annots[label]
                val = list( data[0])
                val = [ float(x) for x in val ]
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}') 

        for label in INT_LIST_ANNOT:
            try:
                data = annots[label]
                val = list( data[0])
                val = [ int(x) for x in val ]
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}') 

        for label in BOOL_ANNOT:
            try:
                data = annots[label]
                val = int( data[0][0])
                if val:
                    val = True
                else:
                    val = False
                logging.debug(f'found label={label} val={val}')
                output_dict[label] = val
            except KeyError:
                logging.debug(f'no label={label}')


    except Exception as e:
        logging.error(f'loadmat() did not work. hd5? ')

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)

   
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
    
    parser.add_argument('infile' ,
                        metavar='infile', 
                        type=str,
                        default=None, 
                        help='BARseq .MAT config file')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    logging.debug(f'infile={args.infile}')
     
    dump_matlab_config(args.infile)

    