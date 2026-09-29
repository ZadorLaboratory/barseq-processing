#!/usr/bin/env python
#
# Aggregate barcode rolonies: transform to 10x stitched coords, assign each rolony
# to a cell, and concatenate all FOVs into one bc-rolonies structure.
#
# Faithful port of MATLAB bc_to_10x.m + assign_bc2cell.m + organize_bc_rolonies.m.
# Reuses the shared helpers apply_transform (= transformPointsForward) and
# assign_rolony_to_cell/get_cellid (= mmassignrol2cell) so bc rolonies are handled
# exactly like the geneseq/hyb rolonies in aggregate-transform/aggregate-cellids.
#
# Inputs (gathered via merge-dummy): 
#   basecalls-bc.joblib (merge/bcseq),
#   all_segmentation.joblib (dilated_labels) 
#   tforms_final.joblib (merge/hyb).
#
# Output: rolonies_bcseq.joblib (merge/bcseq).
#
import argparse
import joblib
import logging
import os
import re
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from barseq.utils import *
from barseq.imageutils import *

def _vstack(arrs, ncols):
    arrs = [a for a in arrs if a is not None and len(a) > 0]
    if not arrs:
        return np.zeros((0, ncols))
    return np.concatenate(arrs, axis=0)


def aggregate_bcseq_np(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'aggregate-data-bcseq'

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

    input_map = {'bc_rol': 'basecalls-bcseq.joblib',
                 'seg'   : 'all_segmentation.joblib',
                 'tforms': 'tforms_final.joblib'}
                 
    (bc_rol_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    bc_rol = joblib.load(bc_rol_file)
    seg = joblib.load(seg_file)
    tform_final = joblib.load(tforms_file)

    # tile ordering MUST match aggregate-data (nsort over segmentation tiles) so the
    # per-FOV cell-id offset i*starting_fov_idx*dummy_cell_num lines up with filt_neurons.id.
    tilename_list = nsort(list(seg.keys()))
    pos_id_map = make_pos_id_map(tilename_list, image_regex, position_group)



    for i, tilename in enumerate(tilename_list):
        try:
            bc = bc_rol[tilename]
        except KeyError:
            logging.warning(f'no bc basecalls for tile {tilename}, skipping')
            continue

        pos40x_x, pos40x_y, pos10_x, pos10_y = [], [], [], []
        seq_l, qual_l, int_l, sig_l = [], [], [], []
        slice_l, fov_l, cellid_l = [], [], []
        n_cyc = None
        n_ch = None

        rows = np.asarray(bc['lroi_x'])      # axis-0 (row, y)
        cols = np.asarray(bc['lroi_y'])      # axis-1 (col, x)
        n = len(rows)
        if bc['seq'].size:
            n_cyc = bc['seq'].shape[1]
            n_ch = bc['sig'].shape[2]
        if n == 0:
            continue

        mask = seg[tilename]['dilated_labels']
        tform = tform_final[tilename]

        # assign_bc2cell: cell label at rolony pixel; global id = local + i*offset
        # (matches aggregate-data cellidall; rolonies outside cells -> i*offset, match no cell).
        cellid_local = np.asarray(get_cellid(mask, rows, cols), dtype=np.int64)
        cellid_global = cellid_local + i * starting_fov_idx * dummy_cell_num

        # bc_to_10x: transform (x=col, y=row) -> 10x stitched coords
        x10, y10 = apply_transform(tform, cols, rows)

        pos_id = pos_id_map[tilename]
        pos40x_x.append(cols)
        pos40x_y.append(rows)
        pos10_x.append(np.asarray(x10))
        pos10_y.append(np.asarray(y10))
        seq_l.append(bc['seq'])
        qual_l.append(bc['score'])
        int_l.append(bc['int'])
        sig_l.append(bc['sig'])
        slice_l.append(np.full(n, pos_id + starting_slice_idx))
        fov_l.append(np.full(n, i))
        cellid_l.append(cellid_global)

    if n_cyc is None:
        n_cyc, n_ch = 0, 4

    bc_out = {
        'pos40x_x': np.concatenate(pos40x_x) if pos40x_x else np.zeros(0),
        'pos40x_y': np.concatenate(pos40x_y) if pos40x_y else np.zeros(0),
        'pos_x': np.concatenate(pos10_x) if pos10_x else np.zeros(0),
        'pos_y': np.concatenate(pos10_y) if pos10_y else np.zeros(0),
        'seq': _vstack(seq_l, n_cyc).astype(np.int8),
        'qual': _vstack(qual_l, n_cyc),
        'int': _vstack(int_l, n_cyc),
        'sig': (np.concatenate([s for s in sig_l if s.size], axis=0)
                if any(s.size for s in sig_l) else np.zeros((0, n_cyc, n_ch))),
        'slice': np.concatenate(slice_l) if slice_l else np.zeros(0, dtype=int),
        'fov': np.concatenate(fov_l) if fov_l else np.zeros(0, dtype=int),
        'cellid': np.concatenate(cellid_l) if cellid_l else np.zeros(0, dtype=np.int64),
        'fov_names': tilename_list,
    }
    logging.info(f'bc-rolonies: {len(bc_out["seq"])} rolonies, {n_cyc} cycles. Writing {outfile}')
    joblib.dump(bc_out, outfile)
    logging.info('Done.')


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

    aggregate_bcseq_np(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
