#!/usr/bin/env python
#
# Produces list of cell_ids to remove for use by other stages. 
#
#   rescale_factor = 0.5
#   px = 0.33
#   box_half_width_um = 5
#   search_radius_um = 200
#
# 
#
import argparse
import joblib
import logging
import os
import re
import sys

from configparser import ConfigParser
from itertools import chain

import numpy as np
# import networkx as nx
from natsort import natsorted as nsort
from scipy.spatial import cKDTree

from barseq.utils import *
from barseq.imageutils import *


def filter_cellid_np(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'filter-cellids'

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
    starting_fov_idx = cp.getint(stage, 'starting_fov_idx')
    dummy_cell_num = cp.getint(stage, 'dummy_cell_num')
    starting_slice_idx = cp.getint(stage, 'starting_slice_idx')

    # calculate secondary parameters
    pr = px / rescale_factor
    overlap_half_width = np.round(box_half_width_um / pr)
    search_radius = np.round(search_radius_um / pr)

    # to use tree distance as final limit instead of bounding box calculation. 
    # hypotenuse of maximum box.
    tree_search_radius = math.sqrt( 2 *  math.pow(2 * overlap_half_width, 2) )


    # gather inputs. 
    input_map = { 'matrix': 'matrix_geneseq.joblib',
                  'segmentation': 'segmentation.joblib',
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

    # set dataframe index to cell_id to ease filtering. 
    total_expr.index = total_expr['cell_id']

    pos_id_map = make_pos_id_map(tilename_list, image_regex, position_group)
    cell_df = make_tiledict_dataframe(cells)
    cell_df['slice_id'] = cell_df['tile_name'].map(pos_id_map).astype('string')

    # Assign cell_ids from local cell_num
    tile_idx_map = make_tilename_idx_map(tilename_list)
    cell_df['tile_name_idx'] = cell_df['tile_name'].map(tile_idx_map)
    cell_df['cell_id'] = cell_df['cell_num'] + (cell_df['tile_name_idx'] * starting_fov_idx * dummy_cell_num )

    # extra values for box algorithm. 
    #cell_df['xmin'] = cell_df['cent_x'] - overlap_half_width
    #cell_df['xmax'] = cell_df['cent_x'] + overlap_half_width
    #cell_df['ymin'] = cell_df['cent_y'] - overlap_half_width
    #cell_df['ymax'] = cell_df['cent_y'] + overlap_half_width

    id_to_keep_all = []
    id_to_remove_all = [] 

    for slice_id in list(cell_df['slice_id'].unique()):
        sdf = cell_df[cell_df['slice_id'] == slice_id ]
        
        #
        #   Find neighbor candidates. 
        #
        c = np.column_stack((sdf['cent_x'], sdf['cent_y']))
        tree = cKDTree( c )
        #sparse_dist = tree.sparse_distance_matrix(tree, 
        #                                          max_distance=search_radius, 
        #                                          output_type='coo_matrix')
    
        # Simply use search radius to fully limit neighbor search. 
        sparse_dist = tree.sparse_distance_matrix(tree, 
                                                  max_distance=tree_search_radius, 
                                                  output_type='coo_matrix')

        cell_id = sparse_dist.row
        nearest_neigh_id = sparse_dist.col
        dist_vals = sparse_dist.data

        cell_ids = sdf.loc[cell_id]['cell_id'].copy()
        nearest_neigh_ids = sdf.loc[nearest_neigh_id]['cell_id'].copy()

        # Neighbors dataframe. 
        ddf = pd.DataFrame()
        ddf['cell_id'] = pd.Series(cell_ids, dtype="int64")
        ddf.reset_index(inplace=True, drop=True)
        ddf['neigh_id'] = pd.Series(nearest_neigh_ids, dtype="int64").reset_index(drop=True)
        ddf['distance'] = pd.Series(dist_vals, dtype='float64')
        odf = ddf[ddf['distance'] > 0 ].reset_index(drop=True)
        # sub distance df contains only overlapping cellids. 
        # overlap when distance less than 8. not square, so may miss corner overlap
        # odf =  ddf[ddf['distance'] < ( overlap_half_width * 2 ) ].copy()

        # Add tile info. Remove pairs in same tile.        
        lookup =  dict(zip( sdf['cell_id'], sdf['tile_name']) )
        odf['cell_tile'] = odf['cell_id'].map(lookup)
        odf['neigh_tile'] =  odf['neigh_id'].map(lookup)

        # Shouldn't happen, but make sure pairs are not in same tile. 
        odf = odf[ odf['cell_tile'] != odf['neigh_tile']]
        # odf.drop( ['cell_tile','neigh_tile' ], inplace=True,  axis=1)
        odf.reset_index(inplace=True, drop=True)

        # Use original notebook logic for cell_id overlap. 
        # find_overlap_box(odf)
        # Use nx connected components for cell_id overlap.  
        # find_overlap_nx(odf, total_expr)
        # odf is now all overlapping cells. 

        # Find groups and choose highest expression
        [_,rev_idx]  = np.unique( odf['cell_id'], return_counts=False, return_inverse=True)
        df = pd.DataFrame({ 'cell_id': odf['cell_id'],
                            'neigh_id': odf['neigh_id'] , 
                            'group': np.transpose(rev_idx)
                            } )        
        exp_map = dict(zip( total_expr['cell_id'],total_expr['total_expr']))
        df['cell_exp'] = df['cell_id'].map(exp_map).fillna(0)
        df['neigh_exp'] =  df['neigh_id'].map(exp_map).fillna(0)
        df = df.astype(int)
        idx_max = df.groupby('group')['neigh_exp'].idxmax()     
        df_ref = df.loc[idx_max]
        remove_cell = df_ref['neigh_exp'] > df_ref['cell_exp']
        id_to_remove = list(df_ref[remove_cell]['cell_id'] )
        id_to_remove_all.append(id_to_remove)

    # Flatten list of slice-specific remove lis
    id_to_remove_all = [item for sublist in id_to_remove_all for item in sublist]
    remove_df = pd.DataFrame({ 'cell_id': pd.Series(id_to_remove_all) })  

    joblib.dump(remove_df, outfile)
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    remove_df.to_csv(outfile, sep='\t')
    
    # Filter cell_df from segmentation. 
    cell_df.set_index(cell_df['cell_id'], inplace=True)
    filtered_cell_df = cell_df.loc[ ~cell_df.index.isin(remove_df['cell_id']) ]

    outfile = os.path.join(dir, f'filtered_cells.joblib')
    joblib.dump(filtered_cell_df, outfile)
    outfile = os.path.join(dir, f'filtered_cells.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    filtered_cell_df.to_csv(outfile, sep='\t')    
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
