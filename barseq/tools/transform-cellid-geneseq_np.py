#!/usr/bin/env python
#
# Apply transforms
# 
# 
import argparse
import joblib
import logging
import math
import os
import re
import pprint
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from barseq.utils import *
from barseq.imageutils import *

def transform_cellid_np(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #             'tforms_final.joblib'
    #
    #. There may be more inputs that required, so only select relevant ones...
    # E.g.
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/all_segmentation.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/genehyb.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_original.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_rescaled0p5.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/geneseq/basecalls.joblib
    # 
    # lroi10x.joblib is main flag output. 

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'transform-cellid-geneseq'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    starting_fov_idx = cp.getint(stage, 'starting_fov_idx')
    dummy_cell_num = cp.getint(stage, 'dummy_cell_num')
    starting_slice_idx = cp.getint(stage, 'starting_slice_idx')
    image_regex = cp.get('barseq', 'file_regex')
    position_group = cp.getint('barseq', 'position_group')
    codebook_file = cp.get('geneseq','codebook_file')
    codebook = pd.read_csv(os.path.join(resource_dir, codebook_file), sep='\t', index_col=0) 


    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    #
    input_map = { 'basecalls'     : 'basecalls_geneseq.joblib',
                  'segmentation' : 'segmentation.joblib',
                  'tforms'       : 'tforms_final.joblib',
                  }

    (basecalls_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    basecalls=joblib.load(basecalls_file)
    seg = joblib.load(seg_file)
    tform_final =joblib.load(tforms_file)
    tilename_list = nsort( list(seg.keys() ))

    # Handle transform. tile coordinates to slice coordinates. 
    tile_dict={}
    for i, tilename in enumerate(tilename_list):
        logging.debug(f'handling {tilename}') 
        t={}
        tform=tform_final[tilename]        
        [x,y]=apply_transform(tform, basecalls[tilename]['lroi_y'], basecalls[tilename]['lroi_x'])
        t['lroi10x_x']=x
        t['lroi10x_y']=y
        tile_dict[tilename]=t

    # first output lroi10x_geneseq.joblib / tsv
    outfile = os.path.join(dir, f'lroi10x_geneseq.joblib')
    logging.info(f'Writing output to {outfile}')
    joblib.dump(tile_dict, outfile)

    outdf = make_tiledict_dataframe(tile_dict)
    outdf['mode'] = 'geneseq'
    outfile = os.path.join(dir, f'lroi10x_geneseq.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    outdf.to_csv(outfile, sep='\t')

    # Handle cell_id assignment
    pos_idx_map = make_pos_idx_map(tilename_list, image_regex, position_group)

    pos40x_x = [] 
    pos40x_y = []
    gene_id_list = []
    gene_list = []
    cell_id_list = []
    slice_id_list = []
   
    for i, tilename in enumerate(tilename_list):
        if tilename not in basecalls:
            logging.warning(f'no geneseq basecalls for tile {tilename}, skipping')
            continue
        tile_basecalls = basecalls[tilename]
        rows = np.asarray(tile_basecalls['lroi_x'])      # axis-0 (row, y)
        cols = np.asarray(tile_basecalls['lroi_y'])      # axis-1 (col, x)
        gene_id = np.asarray(tile_basecalls['gene_id'])
        for gid in tile_basecalls['gene_id']:
            gene = codebook['gene'][gid]
            gene_list.append(gene)        

        mask = seg[tilename]['dilated_labels']
        pos_idx = pos_idx_map[tilename]

        n = len(rows)

        # assign cellid: cell label at rolony pixel; 
        # global id = local + i * offset
        cellid_local = np.asarray(get_cellid(mask, rows, cols), dtype=np.int64)
        cellid_global = cellid_local + ( i * starting_fov_idx * dummy_cell_num )

        pos40x_x.append(cols)
        pos40x_y.append(rows)
        gene_id_list.append(tile_basecalls['gene_id'])
        slice_id_list.append(np.full(n, pos_idx + starting_slice_idx))
        #slice_idx.append(np.full(n, i))
        cell_id_list.append(cellid_global)

    cellid_out = {
        'pos40x_x': np.concatenate(pos40x_x) if pos40x_x else np.zeros(0),
        'pos40x_y': np.concatenate(pos40x_y) if pos40x_y else np.zeros(0),
        'gene_id': np.concatenate(gene_id_list) if gene_id_list else np.zeros(0),
        'gene'   : gene_list,
        'slice_id': np.concatenate(slice_id_list) if slice_id_list else np.zeros(0, dtype=int),
        'cell_id': np.concatenate(cell_id_list) if cell_id_list else np.zeros(0, dtype=np.int64),
    }
    logging.info(f'cellid: {len(cellid_out)} rolonies. Writing {outfile}')
    joblib.dump(cellid_out, outfile)
    logging.info('Done.')

    outdf = make_dict_dataframe(cellid_out)
    outdf['mode'] = 'geneseq'
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    outdf.to_csv(outfile, sep='\t')
    logging.info(f'Done.')
  
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
                    help='label for this stage config')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='outfile. ')
       
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
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    aggregate_transform_np( infiles=args.infiles, 
                            outfiles=args.outfiles,
                            stage=args.stage,  
                            cp=cp )
    logging.info(f'done processing output to {args.outfiles[0]}')
