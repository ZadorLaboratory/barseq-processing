#!/usr/bin/env python
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
                        #nargs='*',
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')

    parser.add_argument('-H','--halt', 
                        metavar='halt',
                        required=False,
                        default=None,
                        type=str, 
                        help=f'Stage name to stop after.')

    parser.add_argument('-L', '--list', 
                        action="store_true",
                        default=False, 
                        dest='list', 
                        help='List workflow stages and exit.')

    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None,  
                    type=str, 
                    help='Outdir.')
    
    parser.add_argument('indir', 
                    metavar='indir',
                    nargs='?',
                    default=None,
                    type=str, 
                    help='input file base dir, containing [bc|gene]seq, hyb dirs.') 
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        loglevel = 'debug'
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   
        loglevel = 'info'
    
    logging.debug(f'args = {args}') 

    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    #logging.debug(f'Running with config={args.config}:\n{cdict}')

    indir = os.path.abspath('./')
    if args.indir is not None:
        indir = os.path.expanduser( os.path.abspath(args.indir))
    logging.debug(f'indir={indir}')
    
    outdir = os.path.abspath('./')
    if args.outdir is not None:
        outdir = os.path.expanduser( os.path.abspath( args.outdir ) )
    os.makedirs(outdir, exist_ok=True)
    logging.debug(f'outdir={outdir}')
    
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    if not args.list:
        logging.debug(f'list is false')
    else:
        stagelist = get_stagelist_info(cp)
        print(stagelist)
        sys.exit(0) 

    run_workflow( indir=indir, 
                  outdir=outdir,
                  cp=cp,
                  halt=args.halt
                )
    
    logging.info(f'done processing output to {outdir}')
 