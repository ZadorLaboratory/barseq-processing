#!/usr/bin/env python
#
# Assign each rolony to a cell, 
#
# Inputs (gathered via merge-dummy): 
#   basecalls-geneseq.joblib (merge/geneseq),
#   all_segmentation.joblib (dilated_labels) 
#
# Simplified output, moving toward Python datatypes. 
#
# Output: cellid_geneseq.joblib (merge/bcseq).
#
import argparse
import joblib
import logging
import os
import re
import sys
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from barseq.utils import *
from barseq.imageutils import *

def assign_cellid_np(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'cellid-geneseq'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    starting_fov_idx = cp.getint(stage, 'starting_fov_idx')
    dummy_cell_num = cp.getint(stage, 'dummy_cell_num')
    starting_slice_idx = cp.getint(stage, 'starting_slice_idx')
    image_regex = cp.get('barseq', 'file_regex')
    position_group = cp.getint('barseq', 'position_group')
    
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('experiment','resource_dir')))
    codebook_file = cp.get('geneseq','codebook_file')
    codebook = pd.read_csv(os.path.join(resource_dir, codebook_file), sep='\t', index_col=0) 

    input_map = {'basecalls': 'basecalls_geneseq.joblib',
                 'segmentation': 'all_segmentation.joblib',
                }
                 
    (basecalls_file, seg_file) = select_input_files(infiles, input_map)
    basecalls = joblib.load(basecalls_file)
    seg = joblib.load(seg_file)
    tilename_list = nsort(list(seg.keys()))
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
        cellid_global = cellid_local + i * starting_fov_idx * dummy_cell_num

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
    parser.add_argument('-d', '--debug', action="store_true", dest='debug', help='debug logging')
    parser.add_argument('-v', '--verbose', action="store_true", dest='verbose', help='verbose logging')
    parser.add_argument('-c', '--config', metavar='config', required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, help='config file.')
    parser.add_argument('-s', '--stage', metavar='stage', default=None, type=str,
                        help='label for this stage config')
    parser.add_argument('-i', '--infiles', metavar='infiles', nargs="+", type=str,
                        help='Gathered input joblib files.')
    parser.add_argument('-o', '--outfiles', metavar='outfiles', default=None, nargs="+", type=str,
                        help='Output file.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cp = ConfigParser()
    cp.read(args.config)

    assign_cellid_np(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
