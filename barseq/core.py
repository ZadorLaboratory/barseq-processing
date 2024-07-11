import logging
import os
import re

from collections import defaultdict
from configparser import ConfigParser


import numpy as np
import scipy

import tifffile as tif

from barseq.utils import *


def get_default_config():
    dc = os.path.expanduser('~/git/barseq-processing/etc/barseq.conf')
    cp = ConfigParser()
    cp.read(dc)
    return cp

def process_maxproj_files(infiles, cp=None, outdir=None ):
    '''
    parse filenames by config regex. 
    create maxproj by position and channel. 
    pos I, Z, C
    
    '''
    
    if cp is None:
        cp = get_default_config()
    
    if outdir is None:
        afile = infiles[0]
        filepath = os.path.abspath(afile)    
        dirname = os.path.dirname(filepath)
        outdir = dirname
    outdir = os.path.abspath(outdir)
    logging.debug(f'making outdir if needed: {outdir} ')
    os.makedirs(outdir, exist_ok=True)    
    logging.info(f'handling {len(infiles)} files...')

    outmap = parse_filenames(infiles, cp)

    # for each position, make maxproj for each channel, and combine
    plist = list(outmap.keys())
    plist.sort()
   
    for pos in plist:
        logging.debug(f'handling pos={pos}')
        chlist = list(outmap[pos].keys())
        chlist.sort()        
        channel_stack = []
        outfile = f'{outdir}/{pos}.tif'
        for ch in chlist:
            flist = outmap[pos][ch]
            flist.sort()
            logging.info(f'making max_proj file {outfile}')
            logging.debug(f'outmap[{pos}][{ch}]={flist}')
            stack = []
            for f in flist:
                logging.debug(f'reading {f} ...')
                image = tif.imread(f)
                stack.append(image)
            maxproj = np.max(stack, axis=0)
            channel_stack.append(maxproj)
            maxproj_stack = np.array(channel_stack)
            logging.debug(f'writing {outfile}...')            
            tif.imwrite(outfile, maxproj_stack)
        
        
    

def parse_filenames(infiles, cp):
    '''
    parse filenames into position + channel sets. 
    return list of mappings of new filename to input files. 
    {  
        newfile1 : 
            { c1 : [in1, in2, in3, in4 ] },
            { c2 : [in1, in2, in3, in4 ] },            
            { c3 : [in1, in2, in3, in4 ] },
            { c4 : [in1, in2, in3, in4 ] },            
            { c5 : [in1, in2, in3, in4 ] },            
       newfile2 : 
            { c1 : [in1, in2, in3, in4 ] },
            { c2 : [in1, in2, in3, in4 ] },            
            { c3 : [in1, in2, in3, in4 ] },
            { c4 : [in1, in2, in3, in4 ] },            
            { c5 : [in1, in2, in3, in4 ] },       
    }
    
    '''
    mp = cp.get('maxproj','microscope_profile')
    tile_regex = cp.get(mp, 'tile_regex')      
    
    # build mapping to files for each position with channels. 
    outmap = defaultdict(lambda: defaultdict(list))
    
    for infile in infiles:
        dp, base, ext = split_path(infile)
        logging.debug(f'base={base} ext={ext}')
        m = re.search(tile_regex, base)
        pos = m.group(1)
        z = m.group(2)
        channel = m.group(3)
        logging.debug(f'pos={pos} z={z} channel={channel}')        
        newbase = f'max_pos_{pos}'
        newchan = f'channel{channel}'
        chlist = outmap[newbase]
        chlist[newchan].append(infile)
    
    logging.debug(f'outmap len={len(outmap)} keys={outmap.keys()} sub={outmap[ next(iter(outmap.keys()))] }')
    return outmap

            
    
        
    
        


