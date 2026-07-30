#!/usr/bin/env python

import argparse
import itertools
import logging
import os
import random
import sys

import pandas as pd

ALPHABET = ['A','C','G','T']

def add_codebook_unused(infile, outfile, n_unused=5):
    logging.debug('Loading infile={infile} ...')
    df = pd.read_csv(infile, sep='\t', index_col=0)
    logging.debug(f'Loaded codebook with {len(df)} entries...')

    n_bases = len(df['sequence'][0] )
    result = [''.join(p) for p in itertools.product(ALPHABET, repeat=n_bases)]
    logging.debug(f'There are {len(result)} permutations of {ALPHABET} of length {n_bases}')

    codebook_set = set(df['sequence'])
    full_set = set(result)
    unused = list( full_set - codebook_set )
    unused_sample = random.sample(unused, k=n_unused)
    logging.debug(f'Got sample of {n_unused} unused sequences: {unused_sample}')

    new_gene = []
    new_sequence = []

    for i, unused_seq in enumerate(unused_sample):
        new_gene.append( f'unused-{i+1}')
        new_sequence.append( unused_seq)
    new_data = pd.DataFrame( {'gene': new_gene, 'sequence' :  new_sequence } )
    logging.debug(f'new_data= {new_data}')
    df = pd.concat( [df, new_data], ignore_index=True)
    df.to_csv(outfile, sep='\t' )
    logging.info(f'Wrote combined dataframe to {outfile}')
    


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

    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    required=True,
                    type=str,
                    )     

    parser.add_argument('-n','--n_unused', 
                        metavar='n_unused',
                        required=False,
                        default=5,
                        type=int, 
                        help='Number of unused sequences to add.')    

    parser.add_argument('infile',
                        metavar='infile',
                        type=str,
                        help='Single BARseq gemeseq codebook file.')
       

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    add_codebook_unused(args.infile, args.outfile, args.n_unused)