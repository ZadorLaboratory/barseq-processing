#!/usr/bin/env python
#
# Produces list of cell_ids to remove for use by other stages. 
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
from scipy.spatial import cKDTree

from barseq.utils import *
from barseq.imageutils import *
 

def filter_cellid_np(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'cellid-geneseq'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    dirpath, base, label, ext = split_path(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # get parameters
    image_regex = cp.get('barseq','image_regex')
    position_group = cp.getint('barseq','position_group')
    rescale_factor=cp.getfloat(stage, 'rescale_factor')
    px = cp.getfloat( stage, 'px')
    box_half_width_um = cp.getint( stage, 'box_half_width_um')
    search_radius_um = cp.getint( stage, 'search_radius_um')


    # calculate secondary parameters
    pr = px / rescale_factor
    overlap_half_width = np.round(box_half_width_um / pr)
    search_radius = np.round(search_radius_um / pr)

    # gather inputs. 
    input_map = { 'matrix': 'matrix_geneseq.joblib',
                  'segmentation': 'transform_segment.joblib',
                }
                 
    (matrix_file, seg_file) = select_input_files(infiles, input_map)
    matrix = joblib.load(matrix_file)
    expmat = matrix['expmat']
    cells = joblib.load(seg_file)
    tilename_list = nsort(list(cells.keys()))

    # build expression dataframe, indexed by cell_id, get summary expression value 
    expdf = pd.DataFrame(data=matrix['expmat'], columns=list(matrix['genes']), index=matrix['cell_id'])
    total_expr = expdf.sum(axis=1)
    total_expr= pd.DataFrame(total_expr, columns=['total_expr'])
    total_expr.reset_index(inplace=True,names=['cell_id'])

    # map is indexes, starting at 0
    pos_idx_map = make_pos_idx_map(tilename_list, image_regex, position_group)
    cell_df = make_tiledict_dataframe(cells)
    #cell_df.rename(columns={'cellpos10x_x':'center_x', 'cellpos10x_y':'center_y' },inplace=True)
 

    for slice_id in list(df['slice_id'].unique()):
        sdf = df[df['slice_id'] == slice_id ]
        c = np.column_stack((sdf['center_x'], sdf['center_y']))
        tree = cKDTree( c )
        sparse_dist = tree.sparse_distance_matrix(tree, 
                                                  max_distance=search_radius, 
                                                  output_type='coo_matrix')
        cell_id = sparse_dist.row
        nearest_neigh_id = sparse_dist.col
        dist_vals = sparse_dist.data
        ddf = pd.DataFrame()
        ddf['cell_id'] = pd.Series(cell_id)
        ddf['neigh_id'] =  pd.Series(nearest_neigh_id)
        ddf['distance'] = pd.Series(dist_vals)
        sddf = ddf[ddf['distance'] > 0 ].reset_index(drop=True)
        # sub distance df contains only overlapping cellids. 
        # overlap when distance less than 8. not square, so may miss corner overlap
        odf =  sddf[sddf['distance'] < ( overlap_half_width * 2 )  ]

        # mdf = pd.merge(odf, total_expr, on="cell_id", how="left").fillna(0)
        # mdf = pd.merge(odf, total_expr, left_on="cell_id", right_on="cell_id", how="left")
        #mdf.rename(columns={'total_expr':'cell_expr'}, inplace=True)

        #mmdf = pd.merge(mdf, total_expr, left_on="neigh_id", right_on="cell_id", how="left").fillna(0)
        #mmdf.rename(columns={'total_expr':'neigh_expr'}, inplace=True)



    joblib.dump(cellid_out, outfile)
    logging.info('Done.')

    outdf = make_dict_dataframe(cellid_out)
    outdf['mode'] = 'hyb'
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

    filter_cellid_np(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
