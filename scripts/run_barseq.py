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
    # placing output in sub-directories by stage. 
    try:
        # denoise indir, outdir, ddict, cp=None
        #sub_outdir = f'{outdir}/denoised'
        logging.info(f'denoising. indir={bse.inputdir} outdir ={outdir}')
        
        process_stage_allfiles_map(indir, outdir, bse, stage='denoise-geneseq', cp=cp) 
        process_stage_allfiles_map(indir, outdir, bse, stage='denoise-hyb', cp=cp)
        process_stage_allfiles_map(indir, outdir, bse, stage='denoise-bcseq', cp=cp)
        
        
        #process_stage_allimages(bse.expdir, sub_outdir, bse, stage='denoise-geneseq', cp=cp)
        #process_stage_allimages(bse.expdir, sub_outdir, bse, stage='denoise-hyb', cp=cp)
        #process_stage_allimages(bse.expdir, sub_outdir, bse, stage='denoise-bcseq', cp=cp)        
        logging.info(f'done denoising.')
        
        #new_indir = sub_outdir        
        #sub_outdir = f'{outdir}/background'
        #process_stage_allimages(new_indir, sub_outdir, bse, stage='background', cp=cp)
        #logging.info(f'done background.')
        
        #new_indir = sub_outdir        
        #sub_outdir = f'{outdir}/regchannels'        
        #process_stage_allimages(new_indir, sub_outdir, bse, stage='regchannels', cp=cp)
        #logging.info(f'done registering image channels')

        #new_indir = sub_outdir        
        #sub_outdir = f'{outdir}/bleedthrough'        
        #process_stage_allimages(new_indir, sub_outdir, bse, stage='bleedthrough', cp=cp)
        #logging.info(f'done applying bleedthrough profiles.')
  
        # keep this new_indir for all registration steps. 
        #logging.info(f'registering images within and across cycles...')
        #logging.info(f'registering all geneseq')
        #new_indir = sub_outdir        
        #sub_outdir = f'{outdir}/regcycle'        
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='regcycle-geneseq', cp=cp) 
        #logging.info(f'done regcycle-geneseq.')

        # keep this new_indir and outdir for all registration steps.                 
        #logging.info(f'registering hyb to geneseq[0]')
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='regcycle-hyb', cp=cp)
        #logging.info(f'done regcycle-hyb')

        # keep this new_indir and outdir for all registration steps.      
        #logging.info(f'registering bcseq[0] to geneseq[0]')
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='regcycle-bcseq-geneseq', cp=cp)

        # keep this new_indir and outdir for all registration steps.         
        #logging.info(f'registering all bcseq to bcseq[0]')
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='regcycle-bcseq', cp=cp)
        #logging.info(f'done registering images.')
      
        # keep this new_indir for all basecall steps. 
        #new_indir = sub_outdir
        #sub_outdir = f'{outdir}/basecall'
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='basecall-geneseq', cp=cp) 
        #logging.info(f'done basecall-geneseq.')

        #process_stage_tilelist_map(new_indir, outdir, bse, stage='basecall-geneseq', cp=cp) 
        #logging.info(f'done basecall-geneseq.')


        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='basecall-hyb', cp=cp) 
        #logging.info(f'done basecall-hyb.')

        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='basecall-bcseq', cp=cp) 
        #logging.info(f'done basecall-hyb.')

        #sub_outdir = f'{outdir}/segment'
        #process_stage_tilelist(new_indir, sub_outdir, bse, stage='segment-', cp=cp) 
        #logging.info(f'done basecall-hyb.')
        


        # Run stitching at end, as it is many-to-one
                
        #new_indir = sub_outdir
        #sub_outdir = f'{outdir}/stitched'
        #process_stage_positionlist(new_indir, sub_outdir, bse, stage='stitch', cp=cp)
      
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
 
 

   