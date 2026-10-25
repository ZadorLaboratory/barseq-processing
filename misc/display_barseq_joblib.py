#!/usr/bin/env python
#
import argparse

import logging
import os
import sys
import joblib

from natsort import natsorted

import numpy
import pandas

IND="  "
LIST_THRESH = 11

def dump_barseq_joblib(infiles):
    for infile in infiles:
        obj = joblib.load(infile)
        print(f'Joblib: {infile}')
        handle_child(obj, level=1)

def handle_child(obj, level=0):
        if type(obj) == dict:
            dks = list(obj.keys())
            dks = natsorted(dks)
            for k in dks:
                o = obj[k]
                if type(o) == numpy.ndarray:
                    print(f'{IND*level}{k} -> ndarray: shape={o.shape}')
                elif type(o) == pandas.core.series.Series:
                    print(f'{IND*level}{k}: Series len={len(o)}')
                elif type(o) == dict:
                    print(f'{IND*level}{k}:')
                    handle_child(o, level=level+1)
                elif type(o) == list:
                    print(f'{IND*level}{k}:')
                    if len(o) < LIST_THRESH:
                        handle_child(o, level=level+1)    
                    else:
                        print(f'{IND*level}{k}: LIST len={len(o)}')

        elif type(obj) == list:
            if len(obj) < LIST_THRESH:
                for elem in obj:
                    handle_child(elem, level=level+1)
            else:
                print(f'{IND*level}LIST len={len(obj)}')            

        elif type(obj) == pandas.core.series.Series:
            print(f'{IND*level}Series len={len(obj)}')      

        elif type(obj) == str:
            print(f'{IND*level}{obj}')

        elif type(obj) == numpy.float64 :
            print(f'{IND*level}{obj}')

        elif type(obj) == numpy.ndarray :
            print(f' numpy.ndarray: shape={obj.shape}')
        else:
            print(f'unhandled type: {type(obj)}')

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
                        help='BARseq .joblib file(s)')
       

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    logging.debug(f'infile={args.infiles}')
      
        
    # create and handle 'real' 'spikein' and 'normalized' barcode matrices...
    dump_barseq_joblib(args.infiles)