#!/usr/bin/env python
#
# Merge per-FOV soma barcodes (basecall-soma-bcseq output) and assign GLOBAL cell ids.
#
# The global id must match aggregate-data exactly:
#   id = local_cellid + i * starting_fov_idx * dummy_cell_num
# where i is the 0-based index over natsorted tilenames (= filt_neurons.id convention).
# So add-somabc can match soma barcodes to filt_neurons by id.
#
import argparse
import joblib
import logging
import os
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from barseq.utils import *
from barseq.imageutils import *


def _concat(arrs):
    arrs = [a for a in arrs if a is not None and len(a) > 0]
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)


def merge_soma_bcseq_pd(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-basecall-soma-bcseq'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    starting_fov_idx = cp.getint(stage, 'starting_fov_idx')
    dummy_cell_num = cp.getint(stage, 'dummy_cell_num')
    basecall_channels = get_config_list(cp, stage, 'basecall_channels')
    basecall_channels.append('N')
 
    # gather all per-FOV tile dicts -> tilename: data
    tile_data = {}
    for infile in infiles:
        d = joblib.load(infile)
        for tilebase, data in d.items():
            tile_data[tilebase] = data

    tilename_list = nsort(list(tile_data.keys()))
    ids, seq, sig, score, seq_hd, sig_hd, score_hd = [], [], [], [], [], [], []
    for i, tilename in enumerate(tilename_list):
        data = tile_data[tilename]
        local = np.asarray(data['cellid'], dtype=np.int64)
        if len(local) == 0:
            continue
        ids.append(local + i * starting_fov_idx * dummy_cell_num)
        seq.append(data['seq'])
        sig.append(data['sig'])
        score.append(data['score'])
        seq_hd.append(data['seq_hd'])
        sig_hd.append(data['sig_hd'])
        score_hd.append(data['score_hd'])

    out = {
        'id': _concat(ids),
        'seq': _concat(seq),
        'sig': _concat(sig),
        'score': _concat(score),
        'seq_hd': _concat(seq_hd),
        'sig_hd': _concat(sig_hd),
        'score_hd': _concat(score_hd),
        'fov_names': tilename_list,
    }
    n = 0 if out['id'] is None else len(out['id'])
    logging.info(f'merged soma-bc: {n} cells across {len(tilename_list)} tiles. writing {outfile}')
    joblib.dump(out, outfile)

    # Make human-readable TSV.
    # Translate bases.
    # 
    keep_columns = ['id', 'seq', 'seq_hd' ]  
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')   

    dflist = []
    for base in list( tile_data.keys() ):
        df = pd.DataFrame()
        for data_col in list( tile_data[base].keys()):
            if data_col in keep_columns:
                df[data_col] = list( tile_data[base][data_col] )   
                if data_col == 'seq' or data_col == 'seq_hd':
                    seq_bases = []
                    for ar in tile_data[base][data_col]:
                        base_list = list(ar)
                        seq_bases.append( ''.join( [ basecall_channels[ y - 1 ] for y in base_list ] ))
                    df[f'{data_col}_sequence'] = seq_bases
                    df = df.drop([data_col], axis=1)
        df['tilename'] = base        
        dflist.append(df)
    outdf = merge_dfs(dflist)
    logging.info(f'writing basecall-soma-bcseq TSV: {outfile}')
    df.to_csv(outfile, sep='\t')        
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
                        help='Per-FOV soma-bc joblib files.')
    parser.add_argument('-o', '--outfiles', metavar='outfiles', default=None, nargs="+", type=str,
                        help='Output file.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cp = ConfigParser()
    cp.read(args.config)

    merge_soma_bcseq_pd(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')
