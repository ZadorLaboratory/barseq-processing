#!/usr/bin/env python
#
#  Filters previously determined overlapping cell_ids from mode matrix. 
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
        stage = 'filter-matrix-geneseq'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    dirpath, base, label, ext = split_path(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # get parameters
    stage_mode = cp.get(stage,'stage_mode')

    # gather inputs. 
    input_map = { 
            'filter': 'filtered_cellids.joblib',
            'matrix': f'matrix_{stage_mode}.joblib',
                }
                 
    (filter_file, matrix_file ) = select_input_files(infiles, input_map)
    matrix = joblib.load(matrix_file)
    matrix_df = pd.DataFrame(data=matrix['expmat'],columns=matrix['genes'],index=matrix['cell_id'])
    matrix_df.index.name='cell_id'
    remove_df = joblib.load(filter_file)

    filtered_matrix = matrix_df.loc[ ~matrix_df.index.isin(remove_df['cell_id']) ]

    joblib.dump(filtered_matrix, outfile)
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')
    logging.info(f'Writing data to TSV: {outfile} ')
    filtered_matrix.to_csv(outfile, sep='\t')
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
