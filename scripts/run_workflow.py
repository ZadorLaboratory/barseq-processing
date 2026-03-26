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

def process_all(indir, outdir=None, expid=None, cp=None):
    '''
    CSHL BARseq pipeline invocation
    Overall "business logic", even idiosyncratic, is capture here.

    Top level function to call into sub-steps...
    @arg indir          Top level input directory. Cycle directories below.  
    @arg outdir         Top-level output directory. Stage directories created below.  
    @arg expid          Label/tag/run_id, may be used to access run/experiment-specific config. 
    @arg cp             ConfigParser object defining stage and implementation behavior. 
     
    '''
    if cp is None:
        cp = get_default_config()
        
    if expid is None:
        expid = cp.get('project','project_id')
    
    logging.info(f'Processing experiment {expid} directory={indir} to {outdir}')
    bse = BarseqExperiment(indir, outdir, cp)
    logging.debug(f'got BarseqExperiment metadata: {bse}')
    
    # In sequence, perform all pipeline processing steps
    # maptypes are tileset, cycle, position
    try:
        stage_list = get_config_list(cp, 'experiment','stages')
        n_stages = len(stage_list)
        logging.info(f'got stage_list={stage_list}')
        for i, stage in enumerate( stage_list ):
            maptype = cp.get(stage, 'maptype')
            modes = get_config_list(cp, stage, 'modes')
            stage_no = i + 1
            logging.info(f'[ {stage_no}/{n_stages} ] Running stage={stage} maptype={maptype} modes={modes}')
            if maptype == 'position':
                process_stage_position_map(indir, outdir, bse, stage=stage, cp=cp)
            elif maptype == 'cycle':
                process_stage_cycle_map(indir, outdir, bse, stage=stage, cp=cp )
            elif maptype == 'tileset':
                process_stage_tileset_map(indir, outdir, bse, stage=stage, cp=cp)
            logging.info(f'[ {stage_no}/{n_stages} ] Done stage={stage}')
        logging.info(f'Done running workflow.') 

    except Exception as ex:
        logging.error(f'got exception {ex}')
        logging.error(traceback.format_exc(None))


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
    
    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None, 
                    type=str, 
                    help='outdir. output base dir if not given.')
    
    parser.add_argument('indir', 
                    metavar='indir',
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
    
    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    logging.debug(f'Running with config={args.config}:\n{cdict}')

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

    process_all( indir=indir, 
                 outdir=outdir,
                 cp=cp )
    
    logging.info(f'done processing output to {outdir}')
 
 

   