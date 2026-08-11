#!/usr/bin/env python
# take matlab codebook and convert to Pandas/TSV 
#
import argparse
import joblib
import logging
import os
import pprint
import sys
import traceback

from configparser import ConfigParser

import numpy as np
import pandas as pd
import scipy

from scipy.io import loadmat

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.utils import *

def format_config(cp):
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    s = pprint.pformat(cdict, indent=4)
    return s

def dump_hyb_codebook(infile, outfile, n_cycles=7):
      
    infile = os.path.abspath(os.path.expanduser(infile))
    outfile = os.path.abspath(os.path.expanduser(outfile))
    (odirpath, obase, olabel, oext) = split_path(os.path.abspath(outfile))

    logging.debug(f'infile={infile} outfile={outfile}')    
    num_channels = 4

    mcodebook = scipy.io.loadmat(infile)['codebookhyb']
    genes=np.array([str(x[0][0]) for x in mcodebook], dtype=str)
    logging.debug(f'mcodebook={mcodebook}')

    mcblist = list(mcodebook)
    lol = []
    for e in mcblist:
        elist = list(e)
        gene = str(elist[0][0])
        seq = str(elist[1][0][0])
        lol.append([gene,seq])
    df = pd.DataFrame(lol, columns=['gene','sequence'])
    df.to_csv(outfile, sep='\t')
    logging.debug(f'got codebook len={len(df)} e.g. {df.iloc[0]}')


def load_codebook_file(infile):
    df = pd.read_csv(infile, sep='\t', index_col=0)
    return df


   
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

    parser.add_argument('-t', '--test', 
                        action="store_true", 
                        default=False,
                        dest='test', 
                        help='test TSV loading and conversion.')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')    

    parser.add_argument('-n','--n_cycles', 
                        metavar='n_cycles',
                        required=False,
                        default=7,
                        type=int, 
                        help='Number of cycles to make codebook.')    



    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    required=True,
                    type=str,
                    )     

    parser.add_argument('infile',
                        metavar='infile',
                        type=str,
                        help='Single BARseq bardenser .mat file.')
       

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)    
    logging.debug(f'Running with config. {args.config}: {cdict}')
    logging.debug(f'infile={args.infile} outfile={args.outfile}')
       
    dump_hyb_codebook(args.infile, args.outfile, args.n_cycles)

    # Test loading the newly-made file...
    #logging.info(f'testing loading of {args.outfile}')
    #df = load_codebook_file(args.outfile)
    #codeflat, R, C, J, genes, pos_unused_codes = make_codebook_object(df)
    #logging.info(f'genes =\n{genes}\npos_unused_codes ={pos_unused_codes}')
    