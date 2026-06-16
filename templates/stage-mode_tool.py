#!/usr/bin/env python
# 
# Generic template for stage scripts in workflow processing. 
# 
#
#
#  Multi-argument commands 
#  https://stackoverflow.com/questions/36166225/using-the-same-option-multiple-times-in-pythons-argparse
#
#
#
import argparse
import joblib
import json
import logging
import os
import sys

import datetime as dt
from natsort import natsorted as nsort 

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def stage_function(infile):
    '''


    '''
    return infile



def stage_mode_tool( infiles, outfiles, template=None, stage=None, cp=None ):
    '''

    Standard function signature. 

    @arg infiles    input file(s)
    @arg outdir     TOP LEVEL out directory
    @arg template   optional file to use as template against infiles, 
                    otherwise register to first. 
    @arg cp         ConfigParser object
    @arg stage      stage label in cp
    
    '''
    if cp is None:
        cp = get_default_config()
    
    if stage is None:
        stage = 'calc-params'


    arity = cp.get(stage, 'arity')

    # For image-handling stages, we determine channels to act upon.
    # By default, we want to pass through any un-selected channels 
    # unchanged to the output.  
    image_type = cp.get(stage, 'image_type')
    output_dtype = cp.get( stage,'output_dtype')
    channel_names =  get_config_list(cp, image_type, 'channels')
    select_channels = get_config_list(cp, stage, 'channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    num_c = len(select_channels)

    # For specialized, stage-specific configuration lookup, we may need mode, 
    # resource directory, and filename construction information, e.g. 
    #  
    mode = get_config_list(cp, stage, 'modes')
    mode = mode[0]
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')
    # config_file = cp.get(microscope_profile, f'config_file_{mode}')
    # config_file_path = os.path.join(resource_dir, config_file)
    # stage_config = load_df(config_file_path, as_array=True)
    # logging.debug(f'config_file_path={config_file_path}')
    logging.debug(f'resource_dir = {resource_dir} microscope_profile = {microscope_profile}')

    # If there is a need to aggregate/combine info from multiple chunks, 
    # set up global data structures here. 


    # infiles, outfiles, and template are now multi-argument lists-of-lists
    for i, infilelist in enumerate(infiles):
        outfilelist = outfiles[i]

        template_file = None
        if template is not None:
            template_file = template[i]

        logging.debug(f'infiles={infilelist} outfiles={outfilelist} template_file = {template_file}')
        
        # If we know arity is single, we can grab the singleton outfile 

        if arity == 'single':
            outfile = outfilelist[0]
            (outdir, file) = os.path.split(outfile)
        elif arity == 'parallele':
            outdir = os.path.split(outfilelist[0])

        # Always ensure/create output directory
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')

        # Gather configuration and log. 
        logging.info(f'handling stage={stage} to outdir={outdir}')
        logging.info(f'handling {len(infilelist)} input files e.g. {infilelist[0]} ')
        (dirpath, base, label, ext) = split_path(os.path.abspath(infilelist[0]))
        (prefix, subdir) = os.path.split(dirpath)
        logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')

        # Handle this input/output map...
        output = stage_function(infile)

        # Write output. If an image, e.g., but could be other formats... 
        write_image(outfile, output)
        
        # with open(outfile, 'w' ) as f:
        #   json.dump(output, f)
        # logging.info(f'wrote json to {outfile}')  


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
    
    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')

    parser.add_argument('-t','--template', 
                    metavar='template',
                    default=None,
                    required=False, 
                    type=str, 
                    help='Optional template file for function')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        action='append',
                        nargs ="+",
                        type=str,
                        help='All files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    action='append',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='One or more outfiles.')
       
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

    #(outdir, file) = os.path.split(args.outfiles[0])
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    stage_mode_tool( infiles=args.infiles,  
                     outfiles=args.outfiles,
                     template=args.template, 
                     stage=args.stage, 
                     cp=cp )
    
    #logging.info(f'done processing output to {outdir}')