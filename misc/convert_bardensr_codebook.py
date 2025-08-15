#!/usr/bin/env python
# take matlab codebook and convert to Pandas/TSV 
#

import argparse
import logging
import os
import pprint
import sys
import traceback

from configparser import ConfigParser

import pandas as pd
import scipy

from scipy.io import loadmat

def format_config(cp):
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    s = pprint.pformat(cdict, indent=4)
    return s

def dump_bardensr_codebook(infile, outfile):
      
    infile = os.path.abspath(os.path.expanduser(infile))
    outfile = os.path.abspath(os.path.expanduser(outfile))
    logging.debug(f'dumping {infile} to {outfile}')    

    matcodebook = scipy.io.loadmat(infile)['codebook']
    logging.debug(f'matcodebook={matcodebook}')
    mcblist = list(matcodebook)
    lol = []
    for e in mcblist:
        elist = list(e)
        gene = str(elist[0][0])
        seq = str(elist[1][0])
        lol.append([gene,seq])
    df = pd.DataFrame(lol, columns=['gene','sequence'])
    df.to_csv(outfile, sep='\t')
    logging.debug(f'got codebook len={len(df)} e.g. {df.iloc[0]}')
    
    

   
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
       
    dump_bardensr_codebook(args.infile, args.outfile)
    