import itertools
import logging
import math
import os
import re
import sys
import traceback

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

    
def get_script_dir():
    logging.debug(f'getting current script name {sys.argv[0]}')
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    logging.debug(f'script_dir = {script_dir}')
    return script_dir


def process_barseq_all(indir, outdir=None, expid=None, cp=None):
    '''
    Top level function to call into sub-steps...
    indir is top-level indir 
    outdir is top-level out directory. 
    expid is label/tag/run_id, may be used to access run/experiment-specific config. 
    cp is combined ConfigParser object. 
    
    overall "business logic", even idiosyncratic, is capture here. 
    
    input/output dirs for each step of pipeline.  
        denoised
        registered
        stitched
        basecalled
        segmented
        ...
        
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'Processing experiment directory={indir} to {outdir}')
    
    ddict = parse_exp_indir(indir, cp)
    logging.debug(f'got ordered cycle data dictionary: {ddict}')
    
    # In sequence, perform all pipeline processing steps
    # placing output in sub-directories by stage. 
    try:
        # denoise indir, outdir, ddict, cp=None
        sub_outdir = f'{outdir}/denoised'
        process_denoise(indir, sub_outdir, ddict, cp=cp)
        
        #process_registration()
        
        new_indir = sub_outdir        
        sub_outdir = f'{outdir}/stitched'
        #process_stitching(new_indir, sub_outdir, ddict, cp=cp)
        
        #process_basecalls()
        
        #process_segmentation()
  
    except Exception as ex:
        logging.error(f'got exception {ex}')
        logging.error(traceback.format_exc(None))


def process_denoise(indir, outdir, ddict, cp=None):
    '''
    indir is top-level in directory
    outdir is top-level out directory
    ddict is dictionary of probes/cycles
    
    handle de-noising of all modalities, all cycles, all images.
    
    general approach to sub-conda environments...
    process = subprocess.Popen(
    "conda run -n ${CONDA_ENV_NAME} python script.py".split(), , stdout=subprocess.PIPE
    )
    output, error = process.communicate()

    '''
    logging.debug(f'handling cycle types={list( ddict.keys())}')
    
    if cp is None:
        cp = get_default_config()

    # general parameters
    tool = cp.get('denoise','tool')
    conda_env = cp.get('denoise','conda_env')
    
    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    
    script_name = f'denoise_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    indir = os.path.expanduser( os.path.abspath(indir) )

    mpp = re.compile( cp.get('maxproj','maxproj_regex'))

    logging.info(f'tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir}')
    logging.debug(f'ddict = {ddict}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    
    command_list = []
    
    for mtype in ddict.keys():
        logging.info(f'handling modality type {mtype}')
        for idir in ddict[mtype]:
            sub_idir = f'{indir}/{idir}'
            sub_outdir = f'{outdir}/{idir}'
            logging.info(f'handling {sub_idir}->{sub_outdir} with image type {mtype}')
            infiles = os.listdir(sub_idir)         
            
            cmd = ['conda','run',
                   '-n', conda_env , 
                   'python', script_path,
                   log_arg,  
                   '--outdir', sub_outdir,                      
                   ]           
            # only process files that match expected pattern.             
            for fname in infiles:
                if mpp.search(fname) is not None:
                    infile = f'{sub_idir}/{fname}'
                    cmd.append(infile)

            #try:
            #    run_command_shell(cmd)
            #except NonZeroReturnException as nzre:
            #    logging.error(f'problem with input files.')
            #    logging.error(traceback.format_exc(None))
            #    raise    
            command_list.append(cmd)
            logging.info(f'handled {sub_idir}')
        logging.debug(f'handled probe type {mtype}')

    logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
    jstack = JobStack()
    jstack.setlist(command_list)
    jset = JobSet( max_processes = n_jobs, jobstack = jstack)
    logging.debug(f'running jobs...')
    jset.runjobs()
    
    logging.info(f'done with denoising...')


def process_registration( cp=None):
    logging.warning('registration not implemented.')
        
def process_stitching(indir, outdir, ddict, cp=None):
    '''
    indir is top-level in directory
    outdir is top-level out directory


    
    '''
    logging.debug(f'handling cycle types={list( ddict.keys())}')
    if cp is None:
        cp = get_default_config() 
    

        
def process_basecalls( cp=None):
    logging.warning('basecalls not implemented.')
        
def process_segmentation( cp=None):
    logging.warning('segmentation not implemented.')







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
                #sddict = {}
                #flist = []
                #allfiles = os.listdir(f'{indir}/{d}/')
                #for f in allfiles: 
                #    if tif.search(f) is not None:
                #        logging.debug(f'{f} is an image file.')
                #        flist.append(f)
                #sddict[d] = flist
                #ddict[k].append(sddict)
                ddict[k].append(d)
    return ddict
        

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
        
        
def parse_filenames_maxproj(infiles, cp=None):
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
    tile_regex = cp.get(mp, 'image_regex')      
    
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


def parse_filenames_positions(infiles, cp=None):
    '''
    parse filenames into position + channel sets. 
    keep only filename not path.  
    {  
        xy1 : 
            { c1 : [in1, in2, in3, in4 ] },
            { c2 : [in1, in2, in3, in4 ] },            
            { c3 : [in1, in2, in3, in4 ] },
            { c4 : [in1, in2, in3, in4 ] },            
            { c5 : [in1, in2, in3, in4 ] },            
       xy2 : 
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
    tile_regex = cp.get(mp, 'image_regex')      
    
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
        newbase = f'xy{pos}'
        newchan = f'c{channel}'
        chlist = outmap[newbase]
        #chlist[newchan].append(f'{base}{ext}')
        chlist[newchan].append(infile)
    logging.debug(f'outmap len={len(outmap)} keys={outmap.keys()} sub={outmap[ next(iter(outmap.keys()))] }')
    return outmap


def make_tilesets(infiles, outdir, cp=None):
    '''
    Given 4-point coordinates in file, generate tile positions. 
    
    Assumes infile is x,y,z coordinates, 4 per position x,y in mm.  
    Finds midpoint and range of all x,y
    Finds equation of best-fit plane in z
    Tiles assuming given pixel size in mm. 
    Tiles with desired overlap. 
    
    '''
    COLS = [ 'min_x', 'max_x', 'mid_x', 'min_y', 'max_y', 'mid_y', 'z_a','z_b','z_c','z_d','pos_id' ]
    
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
            # calc max box x/y boundaries. 
            data_list = calc_bounds(rows, x='x',y='y')
            logging.debug(f'data_list =\n{data_list}')
            # add zplane coefficients
            zplane_list =  calc_zplane(rows, x='x',y='y', z='z')
            logging.debug(f'zcoeff_list =\n{zplane_list}')
            data_list.extend(zplane_list)
            # add position id
            data_list.append(str( pos_id) )
            data_lol.append(data_list)
            pos_id += 1
    
    sdf = pd.DataFrame(data_lol, columns= COLS )
    logging.debug(f'slice dataframe =\n{sdf}')
    
    sdf = calc_n_tiles(sdf, fov_pixels_x, fov_pixels_y, pixel_size, overlap)
    logging.debug(f'slice dataframe w/ tilecount =\n{sdf}')
    sdf.drop(['min_x', 'max_x',  'min_y', 'max_y'], axis=1, inplace=True)
    tdf = tile_slices(sdf)
    logging.debug(f'complete tile list=\n{tdf}')
    return tdf
    

def calc_bounds(ptsdf, x='x', y='y'):
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

    logging.debug(f'x midpoint of {minx} {maxx} = {midx} y midpoint of {miny} {maxy} = {midy}    ')
    dlist = [ minx, maxx, midx, miny, maxy, midy ]
    return dlist

def calc_zplane(posdf, x='x', y='y',z='z'):
    '''
    z = a*x + b*y + c
    
    given input list of 3 or more points (x,y,z) for a given position, 
        calculate zplane coefficients A, B, C 

    https://stackoverflow.com/questions/18552011/3d-curvefitting/18648210#18648210
    Best-fit linear plane, for the Eq: z = a*x + b*y + c.
        See: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    scipy.linalg.lstsq()

    Methods:
    -- linalg.lstsq() minimizes least-squares *vertical* distance from points to fit plane 
    -- Some other methods minimize least-squares *perpendicular* distance from points to fit plane


    '''
    #    a = 4.5
    #    b = 4.3
    #    c = 1020
    #        x       y        z
    #    0  49.902 -26.162  3676.22
    #    1  51.348 -23.167  3672.45
    #    2  49.420 -18.684  3672.45
    #    3  46.737 -20.540  3675.28
    #                        0                   1                   2
    #xyz = np.array([[1.1724546888698482, 0.67037911349217505, 1.6014525241637045], 
    #                [2.0029440384631063, 1.2163076402918147, -1.1082409593302032], 
    #                [-0.87863180025363918, 1.261853987259635, 1.1598532675831237], 
    #                [1.177240124012167, 0.90163100927998241, -1.1405108476689563] ])
    
    xyz = posdf.to_numpy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    A = np.c_[x, y, np.ones(xyz.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, z)    
    # Coefficients in the form: a*x + b*y + c*z + d = 0.
    a, b, c, d = C[0], C[1], -1., C[2]
    return [float(a) ,float(b),float(c),float(d)]


def get_z( x, y, a, b, c, d):
    '''
    a*x + b*y + c*z + d = 0
    a*x + b*y + d = -c*z   
    ((a*x + b*y + d) / -c ) = z 
    z = ((a*x + b*y + d) / -c ) 
    '''
    z = (-1) * ( (a*x + b*y + d ) / c )
    logging.debug(f'got {z} for x={x} and y={y}')
    return z


def calc_n_tiles(sdf, fov_pixels_x=3200, fov_pixels_y=3200, pixel_size=.33, overlap=.15):
    '''
    Calculate x and y axis tile count to cover, given pixel size and overlap.
    
    @args
    sdf:  slice DF format:
        min_x   max_x    mid_x   min_y   max_y    mid_y   
    0  46.737  51.348  49.0425 -26.162 -18.684 -22.4230  
    1  34.320  38.908  36.6140 -26.003 -18.882 -22.4425  
    2  45.950  50.616  48.2830   0.852   8.563   4.7075  
    
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
    Calculates z value by zplane coefficients. 
    
     mid_x    mid_y   z_a        z_b      z_c     z_d          pos_id  x_tiles  y_tiles  fov_x  fov_y    overlap
 0  49.0425 -22.4230   2.124911  0.207182  -1.0   3594.470049  1        5        9       1.056  1.056     0.15

    '''
    logging.debug(f'tile_slice(row) row=\n{row}')
    mid_x = row['mid_x']
    mid_y = row['mid_y']
    fov_x = row['fov_x']
    fov_y = row['fov_y']
    x_tiles = row['x_tiles']
    y_tiles = row['y_tiles']        
    overlap = row['overlap']
    overlap_x = fov_x * overlap      # in mm
    overlap_y = fov_y * overlap      # in mm
    
    # upper left tile is primary. 
    # find middle of it to start. 
    # stride is fov - (overlap * fov) 
    stride_x = fov_x - (overlap_x * fov_x)
    stride_y = fov_y - (overlap_y * fov_y)
    cent_x =  mid_x - (math.floor(x_tiles / 2 ))      
    cent_y =  mid_y + (math.floor(y_tiles / 2 ))
    logging.debug(f'origin tile center = ({cent_x},{cent_y})')
    # calculate center points for all tiles, left to right, top to bottom. 
    
    pos_id = str( row['pos_id'])  
    pos_data_lol = []
    tile_id = 1 
    for xi in range(0, x_tiles):
        for yi in range(0, y_tiles):
            xpos = cent_x + (xi * stride_x )
            ypos = cent_y + (yi * stride_y )
            zpos = get_z(xpos, ypos, row['z_a'],row['z_b'],row['z_c'],row['z_d'] )
            pos_data = [  pos_id, tile_id, xi, yi, xpos, ypos, zpos ]
            logging.debug(f'created tile [{xi},{yi}]: ({xpos}mm,{ypos}mm,{zpos}um)')
            pos_data_lol.append(pos_data)
            tile_id += 1
            
    COLS = [ 'pos_id', 'tile_id', 'tile_x', 'tile_y', 'pos_x', 'pos_y', 'pos_z' ]
    df = pd.DataFrame(pos_data_lol, columns=COLS)
    logging.debug(f'slice_tiles=\n{df}') 
    return df



def check_focus(infiles, outdir=None, cp=None):
    '''
    https://forum.image.sc/t/find-focus-in-a-z-stack/8255/2
    https://forum.image.sc/t/select-in-focus-image-from-z-stack/721/2
    https://imagej.net/plugins/microscope-focus-quality 
    
    ROI : Region Of Interest
    
    Single image methods:
    -- -- imagej DNN plugin.
    
    Z-stack methods:
    -- autofocus hyperstack. richard mort. "normalized variance"
    -- hyper with stacks. william ashby
    -- xiaoyin: downsize for speed, remove brightest 0.05%, select image with maximum mean intensity. 
    -- yuan: ?
    -- cristian: ? 
    
    '''
    logging.debug(f'infiles={infiles}')
    logging.debug(f'checking focus on {len(infiles)} files...')
    if cp is None:
        cp = get_default_config()
    mp = cp.get('maxproj','microscope_profile')
    tile_regex = cp.get(mp, 'image_regex')   
    
    map = parse_filenames_positions(infiles)
    logging.debug(f'{pprint_dict(map)}')
    return None







             

    
        
        
                
    
        
    
        


