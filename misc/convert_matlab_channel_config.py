#!/usr/bin/env python
#
# h5py example
#
#  infile = 'v73data.mat'
#  f = h5py.File(infile, "r") 
#  list( f.keys())
#  ['#refs#', '#subsystem#', 'chshift_20x']
#  f['chshift_20x'].shape
# (4,1)
#  r1 = f['chshift_20x'][0][0]
#  h1 = f[r1]
#  d1 = h1[:] 
#  type(d1)
#  #numpy.ndarray
#  d1.shape
# (1,6) 
#
#  converting mapseq barcodematrix files via h5py
#
#  f = h5py.File(infile, "r")
#  bcmatrix = np.array(f['barcodematrix']) 
#  df = pd.DataFrame(bcmatrix.T) 
#  df.to_csv('bcmatrix.tsv', sep='\t')
#
#  refbarcodes = np.array(f['refbarcodes'])  
#  rdf = pd.DataFrame( refbarcodes.T )   # ascii integers, 1 per column
#  cdf = pd.DataFrame()
#  for col in rdf.columns:
#     cdf[col] = rdf[col].apply(chr)
#
#  def apply_join(row):
#      return ''.join(row)
#
#  joined = cdf.apply(apply_join, axis=1)
#  joined = joined.astype('string')

import argparse
import logging
import os
import sys
import traceback

from configparser import ConfigParser

import pandas as pd
import scipy

from scipy.io import loadmat
import h5py

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

ANNOT_LIST = [ 'chprofile20x','chshift20x']

def dump_matlab_configs(infiles):
    '''
    converts file(s) from .mat to .tsv

    annots to dump:
        chprofile20x
        chshift20x
    
    '''
    for infile in infiles: 
        infile = os.path.abspath(infile)
        dirpath, filename = os.path.split(infile)
        base, ext = os.path.splitext(filename)
        outfile = os.path.join(dirpath, f'{base}.tsv')
        logging.debug(f'dumping {infile} to {outfile}')
        try:    
            annots = loadmat(infile)
            for label in ANNOT_LIST:
                try:
                    data = annots[label]
                    logging.debug(f'found label={label}')
                    df = pd.DataFrame(data)
                    if label == 'chshift20x':
                        df.columns = ['tx','ty']
                    df.to_csv(outfile, sep='\t')
                    logging.info(f'wrote TSV to {outfile}')
                except KeyError:
                    logging.debug(f'no label={label}')
        except Exception as e:
            logging.error(f'loadmat() did not work. hd5? ')
   
   
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
                        help='BARseq .MAT config file(s)')
       

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    logging.debug(f'infile={args.infiles}')
      
        
    # create and handle 'real' 'spikein' and 'normalized' barcode matrices...
    dump_matlab_configs(args.infiles)

    