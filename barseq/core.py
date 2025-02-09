import itertools
import logging
import math
import os
import pprint
import re
import sys
import traceback

from collections import defaultdict
from configparser import ConfigParser

import scipy
import numpy as np
import pandas as pd
import tifffile as tif

from scipy.sparse import dok_matrix, csr_matrix, lil_matrix, csc_matrix, coo_matrix, bsr_matrix

from barseq.utils import *

class BarseqExperiment():
    
    def __init__(self, indir, cp=None):
        '''
        Methods and data structure to keep and deliver
        sets of files in groupings as needed. 
        Centralized metadata. 
        Abstract out mode identifiers from directory names, paths from cycles, and names from position tilesets. 
        
        ddict   (directory dict)    keys = modes, values= list of cycle directories. 
        tdict   (tile dict)         keys = modes, values= list of lists:  cycles, tiles   
        pdict   (position dict)      
        
        All real file paths are stored as relative paths in the object, to allow mapping from 
        subdir to subdir by steps in the pipeline. 
        
        Goal is to 
        1. Fully validate (for missing images) before proceeding. 
        2. Allow retrieving...
            -- All tiles in flat form, grouped by chunksize (for flat processing). 
            -- All tiles, grouped by position, and grouped by chunksize (for stitching).
            -- All tiles within a mode, but across cycles, for a tilename (for registration) 
        3. 
        
        '''
        self.expdir = os.path.abspath( os.path.expanduser(indir))
        self.cp = cp
        if cp is None:
            self.cp = get_default_config()
            
        # geneseq, bcseq, hyb ,n order. 
        self.modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]
        # create directory dict. 
        # mapping from modality to directory names
        self.ddict = self._parse_experiment_indirs(indir, cp = self.cp)
        
        # ordered lists of files by cycle
        self.cdict = self._parse_experiment_cycles(cp=self.cp)
        # self.tdict = self._parse_experiment_tiles(self.ddict, cp=self.cp)
        
        # tiles grouped by position
        self.pdict = self._parse_experiment_images(cp=self.cp)
        
        logging.debug('BarseqExperiment metadata object intialized. ')

    def __repr__(self):
        s = f'BarseqExperiment: \n'
        for mode in self.modes:
            ncyc = len(self.cdict[mode])
            ntiles = 0
            for cyc in self.pdict[mode]:
                for p in list( cyc.keys()):
                    ntiles += len(cyc[p].flatten())          
            s += f'  mode={mode}\tncycles={ncyc}\tntiles={ntiles}\n'
            cyclelist = self.pdict[mode]
            for i, cycle in enumerate( cyclelist):
                s += f'    cycle[{i}]\n'
                skeys = list( cycle.keys())
                skeys.sort()
                for p in skeys:
                    (x,y) = cycle[p].shape
                    s += f'     pos={p} tiles={x*y} [{x}x{y}]\n'
        return s

    def _parse_experiment_indirs(self, indir, cp=None):
        '''
        determine input data structure and files. 
        return dict of lists of dirs by cycle 
                  
        '''    
        if cp is None:
            cp = get_default_config()
        
        re_list = []
        pdict = {}
        ddict = {}
        modes = [ x.strip() for x in cp.get('experiment','modes').split(',') ]
        for mode in modes:
            p = re.compile( cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
                    
        #tif = re.compile( cp.get('barseq','tif_regex'))       
        dlist = os.listdir(indir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        return ddict

        
    def _parse_experiment_cycles(self, cp=None):
        '''
        make dict of dicts of modes and cycle directory names to all tiles within. 
        
        .cdict = { <mode> ->  list of cycles -> list of relative filepaths
        
        
        '''
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')
                
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in self.ddict[mode]:
                cyclelist = []
                cycledir = f'{self.expdir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    m = re.search(image_regex, base)
                    if m is not None:
                        rfile = f'{d}/{base}.{ext}'
                        cyclelist.append(rfile)
                    else:
                        logging.warning(f'file {f} did not pass image regex.')
                cdict[mode].append(cyclelist)
        return cdict

       
    def _parse_experiment_images(self, cp=None):
        '''
        sets of files, grouped by position 

        '''
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')        
        logging.debug(f'image_regex={image_regex}')
        pdict = {}
        # 
        # pdict[mode] -> cyclist[0] -> posdict['1'] ->  dok_matrix
        #
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = self.cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{self.expdir}/{rfile}')
                    dp, base, ext = split_path(afile)
                    logging.debug(f'dp={dp} base={base} ext={ext} for file={afile}')
                    m = re.search(image_regex, base)
                    if m is not None:
                        pos = m.group(1)
                        x = m.group(2)
                        y = m.group(3)
                        x = int(x)
                        y = int(y)
                        logging.debug(f'mode={mode} cycle={i} pos={pos} x={x} y={y} type(pos)={type(pos)}')
                        logging.debug(f'cycdict.keys() = {list( cycdict.keys() )}')
                        pos = str(pos).strip()
                        try:    
                            posarray = cycdict[pos] 
                            logging.debug(f'success. got posarray for cycle[{i}] position {pos}')
                        
                        except KeyError:
                            logging.debug(f'KeyError: creating new position dict for {pos} type(pos)={type(pos)}')
                            cycdict[pos] = lil_matrix( (50,50), dtype='S128' )
                            logging.debug(f'type = {type( cycdict[pos]) }')
                              
                        fname = f'{rfile}'
                        logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                        cycdict[pos][x,y] = fname 
                    else:
                        logging.warning(f'File {afile} fails regex match.')
                pdict[mode].append(cycdict)
                
            logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sarray = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sarray)} ")
                    pnew = self._fix_sparse(sarray)
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        return pdict


    def _fix_sparse(self, sarray):
        '''
        remove empty rows and columns. convert to normal ndarray. 
        '''
        logging.debug(f'input type = {type(sarray)}')
        darray = sarray.toarray()
        nan_cols = np.all(darray == b'', axis = 0)
        nan_rows = np.all(darray == b'', axis = 1)        
        darray = darray[:,~nan_cols]
        darray = darray[~nan_rows,:]
        return darray
        
    def get_tileset(self, mode=None, chunksize=None):
        ''' 
            returns list of all tile image files across cycles for mode
        '''
        tlist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
            
        for m in modes:
            for cyc in self.pdict[m]:
                for p in list( cyc.keys()):
                    for t in cyc[p].flatten():
                        t = t.decode('UTF-8')
                        tlist.append(t)
        return tlist     


    def get_cycleset(self, mode=None):
        '''
         get ordered list of cycles where elements are lists relative paths.  
         optionally restrict to single mode
         if modes must be handled distinctly, then caller must cycle through nodes. 
                 
        '''
        clist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
            
        for mode in modes:
            for c in self.cdict[mode]:
                    clist.append(c)
        return clist 


    def get_imageset(self, mode='bcseq'):
        '''
        Get list of (ordered) lists of images for a single tile across cycles for a single mode. 
        '''
        ilist = [ ]
        # use first cycle as template
        for i in range(0,len(self.cdict[mode][0])):
            #logging.debug(f'{i}')
            flist = []     
            for cyc in self.cdict[mode]:
                flist.append(cyc[i])
            ilist.append(flist)
        return ilist
                
        
    def get_positionset(self, mode=None, cycle=None):
        '''  
            returns list of lists all tile image file, optionally for a mode
            
        '''               
        plist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
            
        for m in modes:
            for cyc in self.pdict[m]:
                for p in list( cyc.keys()):
                    tlist = []
                    for t in cyc[p].flatten():
                        t = t.decode('UTF-8')                       
                        tlist.append(t)
                    plist.append(tlist)
        return plist
    
    
    def validate(self):
        '''
            -- confirms that there corresponding tiles in all cycles of each mode. 
            --             
            @return True if valid, False otherwise   logs warnings as check is made. 
        '''
        return True

    
    def validate_target(self, target_dir ):
        '''
            --  Confirms that target directory contains files parallel to all those in
                Experiment directory. To be used between pipeline stages to confirm all 
                outputs were successfully created. 
                
            @ arg target    Top-level directory of tree to confirm.
            @return True if valid, False otherwise 
        
        '''
        return True
        
    
def get_default_config():
    dc = os.path.expanduser('~/git/barseq-processing/etc/barseq.conf')
    cp = ConfigParser()
    cp.read(dc)
    return cp

    
def get_script_dir():
    logging.debug(f'getting current script name {sys.argv[0]}')
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    logging.debug(f'script_dir = {script_dir}')
    return script_dir




def process_stage_all_images(indir, outdir, bse, stage='background', cp=None, force=False):
    '''
    process any stage that acts on all images singly, batched by cycle directory. 
    
    @arg indir    is top-level input directory (with cycle dirs below)
    @arg outdir   outdir is top-level out directory (with cycle dirs below)
    @arg stage    which pipeline stage should be executed. 
    @arg bse      bse is BarseqExperiment metadata object with relative file/mode layout

    @return None
    handle all images in all modes in parallel to outdir. 
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir}')
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)

    current_env = os.environ['CONDA_DEFAULT_ENV']

    logging.info(f'current_env = {current_env} tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir}')

    # cycle, directory mappings
    ddict = bse.ddict
    # cycle files
    clist = bse.get_cycleset()  # all modes, all tiles
 
    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []
        
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        dirlist = bse.ddict[mode]
        cyclist = bse.get_cycleset(mode)

        # handle batches of cycle directories...
        for i, dirname in enumerate(dirlist):
            sub_outdir = f'{outdir}/{dirname}'
            logging.debug(f'handling mode={mode} dirname={dirname} sub_outdir={sub_outdir}')
            num_files = 0
            flist = cyclist[i]

            if conda_env == current_env :
                logging.debug(f'same envs needed, run direct...')
                cmd = ['python', script_path,
                           log_arg,
                           '--config' , runconfig ,                            
                            ]
            else:
                logging.debug(f'different envs. user conda run...')
                cmd = ['conda','run',
                           '-n', conda_env , 
                           'python', script_path,
                           log_arg, 
                           '--config' , runconfig , 
                           ]
            cmd.append('--stage')
            cmd.append(f'{stage}')
            cmd.append( '--outdir ' )
            cmd.append( f'{sub_outdir}')                
            for rname in flist:
                infile = f'{indir}/{rname}'
                outfile = f'{outdir}/{rname}'
                if not os.path.exists(outfile):
                    cmd.append(infile)
                    num_files += 1
                else:
                    logging.debug(f'outfile {outfile} exists. omitting file.')
            if num_files > 0:
                command_list.append(cmd)
                n_cmds += 1
            else:
                logging.debug(f'all outfiles exist. omitting command.')
            
            logging.info(f'handled {indir}/{dirname}')
        logging.info(f'created {n_cmds} commands for mode={mode}')
    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')


def process_stage_tilelist(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, writing each to parallel (cycle) output 
    subdirs.
    
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.   
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'handling stage={stage} types={bse.modes} indir={indir} outdir={outdir}')

    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )

    logging.info(f'tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir} template_mode={template_mode} template_source={template_source}  ')

    # cycle, directory mappings
    ddict = bse.ddict
    # cycle files
    clist = bse.get_cycleset()  # all modes, all tiles
 
    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []
    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        dirlist = bse.ddict[mode]
        tilelist = bse.get_imageset(mode)
        
        # Use first cycle as template. 
        if template_mode is not None:    
            template_list = bse.get_cycleset(template_mode)[0]
        else:
            template_list = bse.get_cycleset(mode)[0]
        
        # default template source to input directory. 
        template_path = indir
        if template_source == 'input':
            logging.debug(f'template_source={template_source}')
            template_path = indir
        elif template_source == 'output':
            logging.debug(f'template_source={template_source}')
            template_path = outdir
        else:
            logging.warning(f'template_source not specified. defaulting to indir. ')
        
        # Handle batches by tile index. Define template...
        for i, flist in enumerate( tilelist):
            logging.debug(f'handling mode={mode} tile_index={i} n_images={len(flist)} num_cycles={num_cycles}')

            template_rpath = template_list[i]
            num_files = 0
            if conda_env == current_env :
                logging.debug(f'same envs needed, run direct...')
                cmd = ['python', script_path,
                           log_arg,
                           '--config' , runconfig, 
                            ]
            else:
                logging.debug(f'different envs. user conda run...')
                cmd = ['conda','run',
                           '-n', conda_env , 
                           'python', script_path,
                           log_arg, 
                           '--config' , runconfig ,                            
                           ]            
            cmd.append('--stage')
            cmd.append(f'{stage}')
            cmd.append( '--outdir ' )
            cmd.append( f'{outdir}' )            
            cmd.append( f'--template')
            cmd.append( f'{template_path}/{template_rpath}')            
            
            for j, rname in enumerate(flist):
                if j < num_cycles:
                    infile = f'{indir}/{rname}'
                    outfile = f'{outdir}/{rname}'
                    if not os.path.exists(outfile):
                        cmd.append(infile)
                        num_files += 1
                    else:
                        logging.debug(f'outfile {outfile} exists. omitting file.')
                else:
                    logging.debug(f'{j} !< {num_cycles} omitting file.')
            if num_files > 0:
                command_list.append(cmd)
                n_cmds += 1
            else:
                logging.debug(f'all outfiles exist. omitting command.')            
            logging.info(f'handled tileset {i}')

                
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')



def parse_exp_indir(indir, cp=None):
    '''
    determine input data structure and files. 
    return dict of lists of dirs by cycle 
    
     geneseq | bcseq | hyb
    
    As probe types are added, expand this list. 
    
    ddict = { 'geneseq' : 
                [ { 'geneseq01' : ['MAX_Pos1_002_001.tif', 'MAX_Pos1_000_003.tif'] }'
    
    
    
    
    
    '''    
    if cp is None:
        cp = get_default_config()
        
    bcp = re.compile( cp.get('barseq','bc_regex'))
    gsp = re.compile( cp.get('barseq','gene_regex'))
    hyp = re.compile( cp.get('barseq','hyb_regex'))
    tif = re.compile( cp.get('barseq','tif_regex'))
            
    pdict = { bcp : 'bcseq',
              gsp : 'geneseq',
              hyp : 'hyb'            
             }
        
    ddict = { 'bcseq'   : [],
              'geneseq' : [],
              'hyb'     : []
            } 
    
    dlist = os.listdir(indir)
    dlist.sort()
    for d in dlist:
        for p in pdict.keys():
            if p.search(d) is not None:
                k = pdict[p]
                ddict[k].append(d)
    return ddict


