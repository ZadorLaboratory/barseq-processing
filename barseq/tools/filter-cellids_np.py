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
import networkx as nx
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

    # set dataframe index to cell_id to ease filtering. 
    total_expr.index = total_expr['cell_id']

    pos_id_map = make_pos_id_map(tilename_list, image_regex, position_group)
    cell_df = make_tiledict_dataframe(cells)
    cell_df['slice_id'] = cell_df['tile_name'].map(pos_id_map).astype('string')

    id_to_keep = []
    id_to_remove = [] 

    for slice_id in list(cell_df['slice_id'].unique()):
        sdf = cell_df[cell_df['slice_id'] == slice_id ]
        c = np.column_stack((sdf['cent_x'], sdf['cent_y']))
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

        edges = list(odf.itertuples(index=False, name=None))
        dg = nx.Graph()
        dg.add_weighted_edges_from(edges)
        scc = nx.connected_components(dg)
        compsets = list( scc )
        handle_component_sets(compsets, total_expr )     

    joblib.dump(cellid_out, outfile)
    logging.info('Done.')

    outdf = make_dict_dataframe(cellid_out)
    outdf['mode'] = 'hyb'
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    outdf.to_csv(outfile, sep='\t')
    logging.info(f'Done.')


def handle_component_sets(component_list, expr_df ):
    '''
    inputs:
        -- list of cell_id connected components. 
        -- expression table
                cell_id  total_expr
            0        20000       17777
            1        20019           4
            2        20021           1

    output:
        -- keep_cellids, discard_cellids 
    '''
    keep_cellids = []
    discard_cellids = []

    n_present = 0
    n_missing = 0
    n_multi_sets = 0 
    n_compsets = len(compsets)

    for i, compset in enumerate( compsets) :
        logging.debug(f'[{i}/{n_compsets}]')
        comp_list = sorted(list(compset))
        sedf = None
        try:
            sedf = expr_df.loc[ comp_list ]
        except KeyError as ke:
            kerr_str = str(ke)
            (present, missing)  = parse_keyerror_diff(comp_list, kerr_str)
            logging.debug(f'{len(present)} present. {len(missing)} missing.')
            n_present += len(present)
            n_missing += len(missing)
            if n_present >= 2:
                n_multi_sets += 1
                sedf = expr_df.loc[ present ]
            else:
                logging.debug(f'no components in expression table.')
            
        # Calculate highest 
        if sedf is not None:
            


        # logging.debug(f'valid compset expr DF: {sedf}') 
    logging.debug(f'total present={n_present} total_missing={n_missing} n_multi_sets={n_multi_sets} ')

def parse_keyerror_diff(key_list, kerr_str):
    '''
        input: 
        key_list =  [1710, 2228, 9508, 12680, 15410, 16591, 20682, 21670, 27547, 30023]
        kerr_str= '[1710, 2228, 9508, 12680, 15410, 16591, 20682, 21670, 27547] not in index' 
        output: 
            present = [ 30023 ]
            missing = {1710, 2228, 9508, 12680, 15410, 16591, 20682, 21670, 27547}
    '''
    logging.debug(f'KeyError:{kerr_str}')
    present = []
    missing = []
    
    if kerr_str.startswith('"None'):
        #logging.debug(f'Error starts with None. All missing.')
        present = []
        missing = key_list
    elif kerr_str.startswith("'["):
        #logging.debug(f"Error starts with '['. Evaluating. ")
        kes = kerr_str[:kerr_str.find(']') + 1]
        kes = kes.replace("'","")
        kerr_list = eval(kes)
        kerr_set = set(kerr_list)
        key_set = set(key_list)
        present = key_set - kerr_set
        missing = key_set - present

        present = sorted(list(present))
        missing = sorted(list(missing))
    else:
        logging.warning(f'Key Error not parsable: {kerr_str}')
    return (present, missing) 






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
