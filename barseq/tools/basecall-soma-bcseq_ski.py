#!/usr/bin/env python
#
# Basecall soma (whole-cell) barcodes for a single FOV.
#
# Faithful port of MATLAB basecall_somas_xt.m -> mmbasecallcells_xt.m. Two methods
# per cell, both over the cell's pixels in the registered bc image stack:
#   method 1 (seq/sig/score): mean signal over ALL soma pixels per cycle,
#   method 2 (seq_hd/...):     mean over HIGH-QUALITY pixels only (per-pixel quality
#                              = max/L2 over channels; keep pixels whose 3rd-lowest
#                              cross-cycle quality > hd_thresh; require > hd_count).
# basecall = argmax over channels (1-4 = G,T,A,C; 5 = no call for method 2).
#
# The per-tile UNDILATED cellpose mask (segment output, cp_mask_cyto3.tif) arrives as
# --template; the bc cycle stack as --infiles. Output (joblib, one per FOV) is keyed
# by tile basename with LOCAL cell labels; merge-soma-bcseq applies the global offset.
#
import argparse
import logging
import os
import sys
import datetime as dt
from configparser import ConfigParser

import joblib
import numpy as np
import scipy.ndimage as ndi

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.utils import *
from barseq.imageutils import *


def basecall_soma_bcseq_ski(infiles, outfiles, template=None, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'basecall-soma-bcseq'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    image_type = cp.get(stage, 'image_type')
    channel_names = get_config_list(cp, image_type, 'channels')
    basecall_channels = get_config_list(cp, stage, 'basecall_channels')
    ch_idx = channel_names_index_map(basecall_channels, channel_names)
    num_c = len(ch_idx)
    hd_thresh = float(cp.get(stage, 'hd_thresh', fallback='0.85'))
    hd_count = int(cp.get(stage, 'hd_count', fallback='50'))

    (dirpath, base, ilabel, ext) = split_path(os.path.abspath(infiles[0]))
    logging.info(f'soma-bc {base}: {len(infiles)} cycles, hd_thresh={hd_thresh} hd_count={hd_count}')

    # ---- cell mask (undilated cellpose labels) ----
    maski = np.asarray(read_image(template)).astype(np.int64)
    cellid1 = np.unique(maski)
    cellid1 = cellid1[cellid1 != 0]
    n_cells = len(cellid1)

    n_cyc = len(infiles)
    H, W = maski.shape

    if n_cells == 0:
        out = {base: {'cellid': np.zeros(0, dtype=np.int64),
                      'seq': np.zeros((0, n_cyc), dtype=np.int8),
                      'sig': np.zeros((0, num_c, n_cyc)),
                      'score': np.zeros((0, n_cyc)),
                      'seq_hd': np.zeros((0, n_cyc), dtype=np.int8),
                      'sig_hd': np.zeros((0, num_c, n_cyc)),
                      'score_hd': np.zeros((0, n_cyc))}}
        joblib.dump(out, outfile)
        logging.info(f'{base}: no cells. wrote {outfile}')
        return

    # ---- read bc cycle stack: im[cycle][channel] ----
    im = np.zeros((n_cyc, num_c, H, W), dtype=np.float64)
    for m, sf in enumerate(infiles):
        im[m] = read_image(sf, ch_idx).astype(np.float64)

    # ---- method 1: mean signal over ALL soma pixels (grouped mean via ndimage) ----
    # sig1[cell, channel, cycle]
    sig1 = np.zeros((n_cells, num_c, n_cyc))
    for m in range(n_cyc):
        for ch in range(num_c):
            sig1[:, ch, m] = ndi.mean(im[m, ch], labels=maski, index=cellid1)
    seq1 = np.argmax(sig1, axis=1) + 1                      # (n_cells, n_cyc) in 1..num_c
    maxsig1 = np.max(sig1, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        score1 = maxsig1 / np.sqrt(np.sum(sig1 ** 2, axis=1))
    score1[np.isnan(score1)] = 0.5

    # ---- method 2: high-quality pixels only ----
    # per-pixel quality per cycle: max/L2 over channels
    with np.errstate(invalid='ignore', divide='ignore'):
        qpix = np.max(im, axis=1) / np.sqrt(np.sum(im ** 2, axis=1))   # (n_cyc, H, W)
    qpix = np.nan_to_num(qpix, nan=0.0)
    qsort = np.sort(qpix, axis=0)                           # ascending over cycles
    k = min(3, n_cyc) - 1                                   # 3rd-lowest (0-based)
    hd_pixels = qsort[k] > hd_thresh                        # (H, W) bool

    # label image restricted to hd pixels -> grouped mean over hd pixels only
    maski_hd = np.where(hd_pixels, maski, 0)
    hd_count_per_cell = ndi.sum(hd_pixels.astype(np.float64), labels=maski, index=cellid1)
    sig2 = np.zeros((n_cells, num_c, n_cyc))
    for m in range(n_cyc):
        for ch in range(num_c):
            mvals = ndi.mean(im[m, ch], labels=maski_hd, index=cellid1)
            sig2[:, ch, m] = np.nan_to_num(mvals, nan=0.0)

    enough = hd_count_per_cell > hd_count                   # (n_cells,)
    seq2 = np.argmax(sig2, axis=1) + 1                      # provisional
    maxsig2 = np.max(sig2, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        score2 = maxsig2 / np.sqrt(np.sum(sig2 ** 2, axis=1))
    score2 = np.nan_to_num(score2, nan=0.0)
    # cells without enough hd pixels -> no call (5), zero sig/score
    no_call = ~enough
    seq2[no_call, :] = 5
    score2[no_call, :] = 0.0
    sig2[no_call, :, :] = 0.0

    out = {base: {
        'cellid': cellid1.astype(np.int64),     # LOCAL labels; merge applies fov offset
        'seq': seq1.astype(np.int8),
        'sig': sig1,
        'score': score1,
        'seq_hd': seq2.astype(np.int8),
        'sig_hd': sig2,
        'score_hd': score2,
    }}
    logging.info(f'{base}: {n_cells} cells soma-basecalled. writing {outfile}')
    joblib.dump(out, outfile)
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
    parser.add_argument('-t', '--template', metavar='template', default=None, required=False,
                        type=str, help='per-tile cellpose soma mask (cp_mask_cyto3.tif)')
    parser.add_argument('-i', '--infiles', metavar='infiles', nargs="+", type=str,
                        help='All bc cycle image files for one tile.')
    parser.add_argument('-o', '--outfiles', metavar='outfiles', default=None, nargs="+", type=str,
                        help='outfile.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cp = ConfigParser()
    cp.read(args.config)

    basecall_soma_bcseq_ski(infiles=args.infiles, outfiles=args.outfiles,
                            template=args.template, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
