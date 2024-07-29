import itertools
import logging
import os
import re

from collections import defaultdict
from configparser import ConfigParser

import scipy
import numpy as np
import pandas as pd
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
        
        
def parse_filenames(infiles, cp=None):
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
    if cp is None:
        cp = get_default_config()
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


def make_tilesets(infiles, outdir, cp=None):
    '''
    Given 4-point coordinates in file, generate tile positions. 
    
    Assumes infile is x,y,z coordinates, 4 per position x,y in mm.  
    Finds midpoint and range of all x,y,z
    Tiles assuming given pixel size in mm. 
    Tiles with desired overlap. 
    
    '''
    COLS = [ 'min_x', 'max_x', 'mid_x', 'min_y', 'max_y', 'mid_y', 'min_z', 'max_z', 'mid_z', 'pos_id' ]
    
    if cp is None:
        cp = get_default_config()
    
    overlap= float(cp.get('tile','overlap') )

    mp = cp.get('tile','microscope_profile')
    fov_pixels_x= int( cp.get(mp, 'fov_pixels_x'))   
    fov_pixels_y= int( cp.get(mp, 'fov_pixels_x')) 
    pixel_size= float( cp.get(mp, 'pixel_size') )
    logging.debug(f'profile={mp} fov={fov_pixels_x}x{fov_pixels_y}px pixel_size={pixel_size}/um')
    
    if outdir is None:
        afile = infiles[0]
        filepath = os.path.abspath(afile)    
        dirname = os.path.dirname(filepath)
        outdir = dirname
    outdir = os.path.abspath(outdir)
    logging.debug(f'making outdir if needed: {outdir} ')
    os.makedirs(outdir, exist_ok=True)    
    logging.info(f'handling {len(infiles)} files...')
    
    
    data_lol = []
    for infile in infiles:
        posdf = pd.read_csv(infile, sep=';', header=None, names=['x','y','z'])
        logging.debug(f'got position set for {int(len(posdf)/ 4)} positions (4 points each)')
        pos_id = 1
        for i in range(0,len(posdf),4):
            rows = posdf[i:i+4]
            logging.debug(f'\n{rows}')
            data_list = calc_bounds(rows, x='x',y='y', z='z')
            logging.debug(f'data_list =\n{data_list}')
            data_list.append(str( pos_id) )
            data_lol.append(data_list)
            pos_id += 1
    
    sdf = pd.DataFrame(data_lol, columns= COLS )
    logging.debug(f'slice dataframe =\n{sdf}')
    
    sdf = calc_n_tiles(sdf, fov_pixels_x, fov_pixels_y, pixel_size, overlap)
    logging.debug(f'slice dataframe w/ tilecount =\n{sdf}')

    sdf.drop(['min_x', 'max_x',  'min_y', 'max_y','min_z', 'max_z'], axis=1, inplace=True)

    tdf = tile_slices(sdf)
    logging.debug(f'complete tile list=\n{tdf}')
    

def calc_bounds(ptsdf, x='x', y='y', z='z'):
    '''
    assumes points dataframe with 2 or more points. 
    returns list of 3D bounds and midpoint for each set.  
    '''            
    minx = min(ptsdf[x])
    maxx = max(ptsdf[x])
    try:   
        midx = (minx + maxx) / 2
    except DivideByZeroException:
        midx= 0    
    
    miny = min(ptsdf[y])
    maxy = max(ptsdf[y])
    try:
        midy = (miny + maxy) / 2
    except DivideByZeroException:
        midy= 0
    
    minz = min(ptsdf[z])
    maxz = max(ptsdf[z])
    
    try:
        midz = (minz + maxz) / 2
    except DivideByZeroException:
        midz= 0    
    
    logging.debug(f'x midpoint of {minx} {maxx} = {midx} y midpoint of {miny} {maxy} = {midy}    ')
    dlist = [ minx, maxx, midx, miny, maxy, midy, minz, maxz, midz ]
    return dlist


def calc_n_tiles(sdf, fov_pixels_x=3200, fov_pixels_y=3200, pixel_size=.33, overlap=.15):
    '''
    Calculate x and y axis tile count to cover, given pixel size and overlap.
    
    @args
    sdf:  slice DF format:
        min_x   max_x    mid_x   min_y   max_y    mid_y    min_z    max_z     mid_z
    0  46.737  51.348  49.0425 -26.162 -18.684 -22.4230  3672.45  3676.22  3674.335
    1  34.320  38.908  36.6140 -26.003 -18.882 -22.4425  3666.73  3675.72  3671.225
    2  45.950  50.616  48.2830   0.852   8.563   4.7075  3671.21  3676.37  3673.790
    
    integer pixel counts. 
    pixel_size in um. 1/1000 of coordinate units.  
    
    @return 
    For each, adds n_tiles_x, n_tiles_y, fov_x, fov_y, overlap columns to each row
    coordinates in mm.    
    
    '''
    
    fov_x = ( fov_pixels_x * pixel_size ) / 1000
    fov_y = ( fov_pixels_y * pixel_size ) / 1000 
    logging.debug(f'fov_x={fov_x}mm fov_y={fov_y}mm')
    
    # how many tiles in x to cover, how many tiles in y to cover, given fov and overlap?
    n_tiles_x_vals = []
    n_tiles_y_vals = []
    for i, row in sdf.iterrows():
        logging.debug(f'\n{row}')
        # handle x axis
        mid_x = row['mid_x']
        max_x = row['max_x']
        edge_x = mid_x + ( fov_x / 2 )   # edge of center tile
        overlap_x = fov_x * overlap      # in um
        extra_tiles_x = 0                # tiles more than center tile to right of center
        while edge_x < max_x:
            logging.debug(f'edge_x = {edge_x} < max_x = {max_x}')
            extra_tiles_x +=1
            edge_x += (fov_x - overlap_x)
        n_tiles_x = 1 + ( 2 * extra_tiles_x)
        n_tiles_x_vals.append(n_tiles_x)
        # handle y axis
        mid_y = row['mid_y']
        max_y = row['max_y']
        edge_y = mid_y + ( fov_y / 2 )   # edge of center tile
        overlap_y = fov_y * overlap      # in um
        extra_tiles_y = 0                # tiles more than center tile to right of center
        while edge_y < max_y:
            logging.debug(f'edge_y = {edge_y} < max_y = {max_y}')
            extra_tiles_y +=1
            edge_y += (fov_y - overlap_y)
        n_tiles_y = 1 + ( 2 * extra_tiles_y)
        n_tiles_y_vals.append(n_tiles_y)
            
    sdf['x_tiles'] = pd.Series(n_tiles_x_vals)
    sdf['y_tiles'] = pd.Series(n_tiles_y_vals)
    sdf['fov_x'] = fov_x
    sdf['fov_y'] = fov_y
    sdf['overlap'] = overlap
    return sdf

def tile_slices(sdf):
    '''
    Take a set of slices in DF, 
        min_x   max_x    mid_x   min_y   max_y    mid_y    min_z    max_z     mid_z  pos_id  x_tiles  y_tiles
    0  46.737  51.348  49.0425 -26.162 -18.684 -22.4230  3672.45  3676.22  3674.335     1      5        9
    1  34.320  38.908  36.6140 -26.003 -18.882 -22.4425  3666.73  3675.72  3671.225     2      5        9
    
    
    @return
    pos_id tile_id pos_x  pos_y  pos_z 
    1        1
    1        2
    1        3
    1        4
    
    '''
    dflist = []
    for i, row in sdf.iterrows():
        tiledf = tile_slice(row)
        dflist.append(tiledf)
    logging.debug(f'assembled list of {len(dflist)} slice DFs.')
    logging.debug(f'slicedf = \n{dflist[0]}')
    tiledf = merge_dfs(dflist)
    tiledf.sort_values(['pos_id','tile_id'], inplace=True )
    tiledf.reset_index(drop=True, inplace=True)
    return tiledf     

    
def tile_slice(row):
    '''
    Define tiles of a single slice from a slice DF: 
    Calculates x/y slope of z positions. 
    
    min_x   max_x    mid_x   min_y   max_y    mid_y    min_z    max_z     mid_z pos_id  x_tiles  y_tiles  fov_x  fov_y  overlap
 0  46.737  51.348  49.0425 -26.162 -18.684 -22.4230  3672.45  3676.22  3674.335      1        5        9  1.056  1.056     0.15

    '''
    COLS = [ 'pos_id', 'tile_id', 'pos_x', 'pos_y', 'pos_z' ]
    pos_id = str( row['pos_id'])
    testdata= [ [ pos_id, 1, 47.2 , -25.2, row['mid_z'] ], 
                [ pos_id, 2, 48.2 , -24.2, row['mid_z'] ],  
                [ pos_id, 3, 49.2 , -23.2, row['mid_z'] ],       
                [ pos_id, 4, 50.2 , -22.2, row['mid_z'] ],
              ]
    df = pd.DataFrame(testdata, columns=COLS)

    logging.debug(f'slice_tiles=\n{df}') 
    return df







             

    
        
        
                
    
        
    
        


