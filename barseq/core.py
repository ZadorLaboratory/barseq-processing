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
    '''
        Methods and data structure to keep and deliver
        sets of files in groupings as needed.
        Generates maps of stage-to-stage file relationships as required.  
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
        3. Allow checking for existing output? 

        Nomenclature/conventions:
        
        mode     top-level category of image subsets
        cycle    a mode has one or more cycles, consisting of positions made up of images
        image    a single image file
        tile     a set of images that represent a single FOV across cycles
        position a set of adjacent images across cycles 


        Assumes path/naming hierarchy:
        
        EXP123/

            M1C1/           M1C2/               M2C1/
                P1T1.<ext>     P1T1.<ext>           P1T1.<ext>
                P1T2.<ext>     P1T2.<ext>           P1T2.<ext>
                P2T1.<ext>     P2T1.<ext>           P2T1.<ext>
                P2T2.<ext>     P2T2.<ext>           P2T2.<ext>

            M1/                                 M2/
                P1.<label>.<ext>                    P1.<label>.<ext>
                P2.<label>.<ext>                    P2.<label>.<ext>
 
        Key methods/ conventions. 
        return elements are relative PATHS under EXP123, not simply file names. 
        This allows retention of subdirectory hierarchy on outputs. 
         
        get_Xlist  -> flat list
        get_Xset   -> list of lists, structured appropriately. 
        
        get_filelist()
            Flat list of all files. 
            Unit of work: individual file
            [ M1C1/P1T1, M1C1/P1T2, M1C1/P2T1, M1C1/P2T2, 
              M1C2/P1T1, M1C2/P1T2, M1C2/P2T1, M1C2/P2T2,
              M2C1/P1T1, M2C1/P1T2, M2C1/P2T1, M2C1/P2T2
             ]
        
        get_cycleset( mode=M1)  
            Files grouped by cycle, otherwise unstructured. 
            Unit of work: individual file. 
        
            [ [ M1C1/P1T1, M1C1/P1T2, M1C1/P2T1, M1C1/P2T2 ], 
              [ M1C2/P1T1, M1C2/P1T2, M1C2/P2T1, M1C2/P2T2 ] 
            ]        
        
        get_positionset( mode=M1 )
            Files grouped by position value, otherwise unstructured. 
            Unit of work: all files for position (e.g. for stitching). 
            [ [ M1C1/P1T1, M1C1/P1T2], [ M1C1/P2T1, M1C1/P2T2 ], 
              [ M1C2/P1T1, M1C2/P1T2], [ M1C2/P2T1, M1C2/P2T2 ], 
            ]
        
        get_tileset( mode=M1 )
            Files grouped by tile value, ordered by cycle.
            Unit of work: all images for given tile, typically within mode. 
                e.g. for registration. 
            [  [ M1C1/P1T1, M1C2/P1T1 ], 
               [ M1C1/P1T2, M1C2/P1T2 ],
               [ M1C1/P2T1, M1C2/P2T1 ],
               [ M1C1/P2T2, M1C2/P2T2 ]
            ] 
    '''
    
    def __init__(self, indir, outdir, cp=None):
        '''
        @arg indir     overall input data directory
        @arg outdir    ovarall output working directory
        @arg cp        experiment config
                
        '''       
        self.cp = cp
        if cp is None:
            self.cp = get_default_config()
        self.inputdir = os.path.abspath( os.path.expanduser(indir))
        self.outputdir = os.path.abspath( os.path.expanduser(outdir)) 
        self.modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]

        # cache parsed file map trees. 
        # input key is 'input'
        self.stageinfo = {}
        
        (ddict, cdict, pdict) = self.parse_stage_dir()
        logging.debug(f'ddict = {ddict} cdict={cdict} pdict={pdict}')
        
        self.ddict = ddict
        self.cdict = cdict
        self.pdict = pdict

        logging.debug('BarseqExperiment metadata object intialized. ')


    def parse_stage_dir(self, stage=None):
        '''
        Top-level combined method, handling dirs, cycles, and positions. 
        '''
        re_list = []
        pdict = {}
        ddict = {}
        modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]
        for mode in modes:
            p = re.compile( self.cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
        
        if stage is None:
            parse_dir = self.inputdir
            file_regex = self.cp.get( 'barseq' , 'file_regex')
        else:
            stagedir = self.cp.get(stage, 'stagedir' )
            parse_dir = os.path.join( self.outputdir, stagedir )
            file_regex = self.cp.get( stage , 'file_regex')
        logging.debug(f'parse directory is {parse_dir}')
        
        dlist = os.listdir(parse_dir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        logging.debug(f'directory dict = {ddict}')
      
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in ddict[mode]:
                cyclelist = []
                cycledir = f'{parse_dir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    m = re.search(file_regex, base)
                    if m is not None:
                        rfile = f'{d}/{base}.{ext}'
                        cyclelist.append(rfile)
                    else:
                        logging.warning(f'file {f} did not pass image regex.')
                cdict[mode].append(cyclelist)        

        pdict = {}
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{parse_dir}/{rfile}')
                    dp, base, ext = split_path(afile)
                    base = base.rsplit('.',1)[0]
                    logging.debug(f'dp={dp} base={base} ext={ext} for file={afile}')
                    m = re.search(file_regex, base)
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
                            cycdict[pos] = SimpleMatrix()
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
                    sm = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    pnew = sm.to_ndarray()
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        
        if stage is not None:
            logging.debug(f'caching stage info for {stage}')
            self.stageinfo[stage] = (ddict, cdict, pdict)
        
        return (ddict, cdict, pdict)    
            

    def get_filelist(self, mode=None, stage=None, chunksize=None):
        ''' 
        returns FLAT list of ALL files across ALL cycles for mode(s)
        '''
        tlist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
        
        if stage is None:
            pdict = self.pdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
            except KeyError:
                (ddict, cdict, pdict) = self.parse_stage_dir( stage)
        
        for m in modes:
            for cyc in pdict[m]:
                for p in list( cyc.keys()):
                    for t in cyc[p].flatten():
                        try:
                            t = t.decode('UTF-8')
                        except:
                            t = str(t)
                        tlist.append(t)
        return tlist     



    def get_cycleset(self, mode=None, stage=None):
        '''
         get ordered list of cycles where elements are flat lists of relative 
         paths of ALL images in that cycle  
         optionally restrict to single mode.
         if modes must be handled differently, then 
         caller must cycle through modes explicitly 
                 
        '''
        clist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]

        if stage is None:
            cdict = self.cdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
            except KeyError:
                (ddict, cdict, pdict) = self.parse_stage_dir( stage )

        for mode in modes:
            for c in self.cdict[mode]:
                    clist.append(c)
        return clist 


    def get_tileset(self, mode='bcseq', stage=None):
        '''
        Get list of (ordered) lists of images for a single tile across cycles for a single mode. 
        '''
        ilist = [ ]
        
        if stage is None:
            cdict = self.cdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
            except KeyError:
                (ddict, cdict, pdict) = self.parse_stage_dir( stage )
        
        # use first cycle as template
        for i in range(0,len(cdict[mode][0])):
            #logging.debug(f'{i}')
            flist = []     
            for cyc in cdict[mode]:
                flist.append(cyc[i])
            ilist.append(flist)
        return ilist

    def get_positionset_map(self, 
                        mode='geneseq', 
                        stage=None, 
                        label=None, 
                        ext=None, 
                        arity='single',
                        instage=None
                        ) :
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. 
        
        return similar format as get_tileset() except each element is a tuple (input_rpath, output_rpath)
        if arity is 'single' output_rpath is a single item, with leading directory set to <mode>
        
        By default both rpaths the same. 
        
        label=None, ext=None, arity='parallel'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
            ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif']
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'] 
          )
        ]

        label='spots, ext='csv, arity='single'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
             'geneseq/MAX_Pos1_003_004.spots.csv'
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            'geneseq/MAX_Pos1_000_001.spots.csv' 
          )
        ]        

        '''
        positionset_list = self.get_positionset(mode=mode, stage=instage)
        
        output_list = []
        output_elem = None 
        
        for ps in positionset_list:
            if arity == 'parallel':
                output_elem = []
                if (ext is not None) or (label is not None):
                    for rpath in ps:
                        (subdir, base, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join(subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join(subdir, f'{base}.{ext}')
                        output_elem.append(out_rpath) 
                else:
                    output_elem = ps.copy()
                    
            elif arity == 'single':
                (subdir, base, current_ext) = parse_rpath( ps[0] )
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        output_elem = os.path.join(mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join(mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join(mode , f'{base}.{ext}')
                    
            logging.debug(f'tileset output={(ps, output_elem)}')        
            output_list.append( (ps, [ output_elem ]) )
        logging.debug(f'made list of {len(output_list)} tilesets.')     
        return output_list

                
    def get_positionset(self, mode=None, stage=None, cycle=None):
        '''  
        Creates list of lists all tile files in sets.
        Each position consists of 1 or more tiles. 
        '''               
        positionlist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]

        if stage is None:
            pdict = self.pdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
            except KeyError:
                (ddict, cdict, pdict) = self.parse_stage_dir( stage )
            
        for m in modes:
            for cyc in self.pdict[m]: 
                for p in list( cyc.keys()):
                    tlist = []
                    for t in cyc[p].flatten():
                        #t = t.decode('UTF-8')                       
                        t = str(t)
                        tlist.append(t)
                    positionlist.append(tlist)
        return positionlist


    def get_filelist_map(self,                         
                         mode='geneseq', 
                         stage=None, 
                         label=None, 
                         ext=None, 
                         arity='parallel',
                         instage=None):
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. Initial input default. 
        

        To remain consistent with tileset handling, output is still a list of tuples, where 
        each element of the tuple is a list. 
        
        by default, for a flat, 2-element file list with parallel logic we get:
        [  
            ( 
                ['geneseq01/MAX_Pos1_000_000.tif',
                 'geneseq01/MAX_Pos1_000_001.tif',],
                ['geneseq01/MAX_Pos1_000_000.tif',
                 'geneseq01/MAX_Pos1_000_001.tif',]
            )
        ]
                 
        '''
        file_list = self.get_filelist(mode=mode, stage=instage)
        #
        # list for input-output tuples
        output_list = []
        stagedir = self.cp.get(stage, 'stagedir')
        if arity == 'parallel':
            infile_list = []
            outfile_list = []
            for rpath in file_list:
                infile_list.append(rpath)
                if (ext is not None) or (label is not None):
                    (subdir, base, current_ext) =  parse_rpath(rpath)
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        out_rpath = os.path.join( subdir, f'{base}.{label}.{ext}')
                    else:
                        out_rpath = os.path.join( subdir, f'{base}.{ext}')
                    outfile_list.append( out_rpath )
                else:
                    outfile_list.append( rpath )
            output_list.append( ( infile_list, outfile_list) )        
                
        elif arity == 'single':
            # Use first input rpath as model for output_rpath
            # Assume mode output dir (not numbered cycle dir)
            (subdir, base, current_ext) = parse_rpath( file_list[0] )
            if (ext is not None) or (label is not None):
                if ext is None:
                    ext = current_ext
                if label is not None:
                    output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                else:
                    output_elem = os.path.join( mode, f'{base}.{ext}')
            else:
                output_elem = os.path.join( mode, f'{base}.{ext}')
            
            output_list = [ (file_list, [ output_elem] ) ]        
            logging.debug(f'filelist output={(ts, output_elem)}')        
            output_list.append( (ts, [ output_elem ])  )
        logging.debug(f'made list of {len(output_list)} filemaps')     
        return output_list        
        

    def get_cycleset_map(self,                         
                         mode='bcseq', 
                         stage=None, 
                         label=None, 
                         ext=None, 
                         arity='parallel',
                         instage=None):
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. Initial input default. 
        
        
         get ordered list of cycles where elements are flat lists of relative 
         paths of ALL images in that cycle  
         optionally restrict to single mode.
         if modes must be handled differently, then 
         caller must cycle through modes explicitly 
                 
        '''
        cycle_list = self.get_cycleset(mode=mode, stage=instage)

        output_list = []
        output_elem = None 
        stagedir = self.cp.get(stage, 'stagedir')

        for cs in cycle_list:
            if arity == 'parallel':
                infile_list = []
                outfile_list = []
                for rpath in cs:
                    infile_list.append(rpath)
                    if (ext is not None) or (label is not None):
                        (subdir, base, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join( subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join( subdir, f'{base}.{ext}')
                        outfile_list.append( out_rpath )
                    else:
                        outfile_list.append( rpath )
                output_list.append( ( infile_list, outfile_list) )        
                    
            elif arity == 'single':
                # Use first input rpath as model for output_rpath
                # Assume mode output dir (not numbered cycle dir)
                (subdir, base, current_ext) = parse_rpath( cycle_list[0] )
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join( mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join( mode, f'{base}.{ext}')
                        
                logging.debug(f'filelist output={( cs, output_elem)}')        
                output_list.append( ( cs , [output_elem] )  )
        logging.debug(f'made list of {len(output_list)} filemaps')     
        return output_list        
        
        
    def get_tileset_map(self, 
                        mode='bcseq', 
                        stage=None, 
                        label=None, 
                        ext=None, 
                        arity='parallel',
                        instage=None
                        ) :
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. 
        
        return similar format as get_tileset() except each element is a tuple (input_rpath, output_rpath)
        if arity is 'single' output_rpath is a single item, with leading directory set to <mode>
        
        By default both rpaths the same. 
        
        label=None, ext=None, arity='parallel'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
            ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif']
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'] 
          )
        ]

        label='spots, ext='csv, arity='single'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
             'geneseq/MAX_Pos1_003_004.spots.csv'
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            'geneseq/MAX_Pos1_000_001.spots.csv' 
          )
        ]        

        '''
        tileset_list = self.get_tileset(mode=mode, stage=instage)
        
        output_list = []
        output_elem = None 
        
        for ts in tileset_list:
            if arity == 'parallel':
                output_elem = []
                if (ext is not None) or (label is not None):
                    for rpath in ts:
                        (subdir, base, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join(subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join(subdir, f'{base}.{ext}')
                        output_elem.append(out_rpath) 
                else:
                    output_elem = ts.copy()
                    
            elif arity == 'single':
                (subdir, base, current_ext) = parse_rpath( ts[0] )
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        output_elem = os.path.join(mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join(mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join(mode , f'{base}.{ext}')
                    
            logging.debug(f'tileset output={(ts, output_elem)}')        
            output_list.append( (ts, [ output_elem ]) )
        logging.debug(f'made list of {len(output_list)} tilesets.')     
        return output_list

    def parse_stage_indirs(self, indir, stage='basecall', cp=None):
        '''
        make a map of a stage output directory, for use
        in generating stage-by-stage processing commands. 
        
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
                           
        stagedir = os.path.join(indir, stage)
        logging.debug(f'scanning stagedir={stagedir}')
        dlist = os.listdir(stagedir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        return ddict    
    

    def parse_stage_cycles(self, indir, ddict, stage='basecall', cp=None):
        '''
        make dict of dicts of modes and cycle directory names to all files within. 
        
        .cdict = { <mode> ->  list of cycles -> list of relative filepaths
        
        '''
        if cp is None:
            cp = get_default_config()
                
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in ddict[mode]:
                cyclelist = []
                cycledir = f'{indir}/{stage}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    rfile = f'{d}/{base}.{ext}'
                    cyclelist.append(rfile)
                cdict[mode].append(cyclelist)
        return cdict    

    def parse_stage_files(self, indir, cdict, stage='basecall', cp=None):
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
            cycfilelist = cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{self.expdir}/{rfile}')
                    dp, base, ext = split_path(afile)
                    # Take base as string up to first dot. 
                    base = base.split('.', 1)[0]
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
                            #cycdict[pos] = lil_matrix( (50,50), dtype='S128' )
                            #cycdict[pos] = coo_matrix( (50,50), dtype='S128' )
                            cycdict[pos] = SimpleMatrix()
                            logging.debug(f'type = {type( cycdict[pos]) }')
                              
                        fname = f'{rfile}'
                        logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                        cycdict[pos][x,y] = fname 
                    else:
                        logging.warning(f'File {afile} fails a regex match.')
                pdict[mode].append(cycdict)
                
            logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sm = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    #pnew = self._fix_sparse(sarray)
                    pnew = sm.to_ndarray()
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        return pdict

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
          
    
    def validate(self):
        '''
            -- confirms that there images corresponding to all tiles in all cycles of each mode. 
            --             
            @return True if valid, False otherwise   logs warnings as check is made. 
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


def process_stage_file_map(indir, outdir, bse, stage='denoise-geneseq', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    @arg indir          Top-level experiment input directory
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    
    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        
        file_map = bse.get_filelist_map(mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage
                                       )
        logging.debug(f'file_map= {file_map}')
        
        # Use first cycle as template. 
        #if template_mode is not None:    
        #    template_list = bse.get_cycleset(template_mode)[0]
        #else:
        #    template_list = bse.get_cycleset(mode)[0]
        #    logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                for i, fname in enumerate( output_list):
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.warning(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')


def process_stage_position_map(indir, outdir, bse, stage='stitch', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    @arg indir          Top-level experiment input directory, ignored if instage is not None
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all files in a related list, with output to parallel folders.
    Assumes one or more input files to process.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    
    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        
        file_map = bse.get_positionset_map(mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage
                                       )
        logging.debug(f'file_map= {file_map}')
        
        # Use first cycle as template. 
        #if template_mode is not None:    
        #    template_list = bse.get_cycleset(template_mode)[0]
        #else:
        #    template_list = bse.get_cycleset(mode)[0]
        #    logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                for i, fname in enumerate( output_list):
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.warning(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')

def process_stage_cycle_map(indir, outdir, bse, stage='stitch', cp=None, force=False):
    '''
    process any stage that handles a list of files per entire cycle, following input-output map. 
    
    @arg indir          Top-level experiment input directory, ignored if instage is not None
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all files in a related list, with output to parallel folders.
    Assumes one or more input files to process.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    
    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        
        file_map = bse.get_cycleset_map(mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage
                                       )
        logging.debug(f'file_map= {file_map}')
        
        # Use first cycle as template. 
        #if template_mode is not None:    
        #    template_list = bse.get_cycleset(template_mode)[0]
        #else:
        #    template_list = bse.get_cycleset(mode)[0]
        #    logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                logging.debug(f'arity=parallel output_list length={len(output_list)}')
                for i, fname in enumerate( output_list):
                    logging.debug(f'making outfile outdir={outdir} stagedir={stagedir} fname={fname}')
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.warning(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')



def process_stage_tilelist_map(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    num_cycles = int(cp.get(stage, 'num_cycles'))

    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None

    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

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
        #dirlist = bse.ddict[mode]
        #tilelist = bse.get_tileset(mode)
        
        tilelist = bse.get_tileset_map(mode='geneseq', 
                                       stage=stagedir, 
                                       label='spots',
                                       ext='csv',
                                       arity='single',
                                       
                                       )
        logging.debug(f'tilelist= {tilelist}')
        
        # Use first cycle as template. 
        #if template_mode is not None:    
        #    template_list = bse.get_cycleset(template_mode)[0]
        #else:
        #    template_list = bse.get_cycleset(mode)[0]
        #    logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        # Handle batches by tile index. Define template...
        for i, fmap in enumerate( tilelist):
            (input_list, output_list) = fmap
    
            logging.debug(f'handling mode={mode} tile_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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
            if len(output_list) > 1:
                cmd.append( '--outdir ' )
                cmd.append( f'{outdir}' )
            elif len(output_list) == 1:
                cmd.append( '--outfile ')
                outfile = os.path.join(outdir, stagedir, output_list[0]   )
                cmd.append( outfile )
            
            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            for rpath in input_list:
                infile = os.path.join( indir , rpath )
                cmd.append( f' {infile} ')

            # Check for ALL output. Any missing re-runs. 
            output_complete = True 
            for rpath in output_list:
                outfile = os.path.join(outdir, stagedir, rpath )
                output_complete = os.path.exists(outfile) and output_complete
            if not output_complete:
                command_list.append(cmd)
                cmdstr = ' '.join(cmd)            
                logging.info(f'tileset {i} cmdstr={cmdstr}')
            else:
                logging.info(f'tileset {i} output complete. skipping command ')    
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')


def parse_rpath(rpath):
    '''
    Assumes relative path with single subdir
       'geneseq03/MAX_Pos1_003_002.tif'
       
    '''
    (subdir, filename) = rpath.split('/')
    (base, ext) =  filename.rsplit('.', 1)
    return (subdir, base, ext)
    

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


