#!/usr/bin/env python
#
# Basecall barcode (bcseq) rolonies for a single FOV across all bc cycles.
#
# Faithful port of MATLAB basecall_barcodes_highres.m -> mmbasecallsinglerol_bgnsub.m
# (E:\git\barseq_helpers). Per FOV:
#   1. read bc cycle-1 (4 seq channels G,T,A,C), find rolonies per channel via an
#      h-maxima (prominence rolthresh) regional-max + per-component peak pick,
#   2. merge peaks across channels and de-duplicate adjacent peaks,
#   3. read EVERY cycle (optional gauss pre-smooth + ball-bgnrad top-hat) and read
#      out the 4-channel signal at each rolony position,
#   4. basecall = argmax over channels (1-4 = G,T,A,C); score = max/L2-norm.
#
# Output (joblib, one file per FOV): dict keyed by tile basename ->
#   {'lroi_x'(rows), 'lroi_y'(cols), 'seq'(N,Ncyc 1-4), 'score'(N,Ncyc),
#    'sig'(N,Ncyc,4), 'int'(N,Ncyc)}
# lroi_x/lroi_y use the same (row, col) convention as the geneseq/hyb basecalls so
# the shared aggregate helpers (get_cellid / apply_transform) apply unchanged.
#
import argparse
import logging
import os
import sys
import datetime as dt
from configparser import ConfigParser

import joblib
import numpy as np
import skimage
import scipy.ndimage as ndi
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import extrema

from barseq.utils import *
from barseq.imageutils import *


def _parse_per_channel(value, num_c):
    '''Parse a config value that may be a scalar or a [a,b,c,d] list into a
    length-num_c list of floats (MATLAB rolthresh can be scalar or per-channel).'''
    s = str(value).strip().strip('[]')
    parts = [p for p in s.replace(',', ' ').split() if p]
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        vals = vals * num_c
    return vals[:num_c]


def find_rolonies_one_channel(a, rolthresh):
    '''
    Port of the per-channel rolony finder in mmbasecallsinglerol_bgnsub (relaxed==0):
        CC = bwconncomp(imregionalmax(imreconstruct(max(a-rolthresh,0), a)))
        peak = pixel of max a within each connected component.
    imreconstruct(a-h, a) + imregionalmax == the h-maxima transform with h=rolthresh,
    i.e. skimage.morphology.extrema.h_maxima(a, rolthresh). Returns a list of
    (row, col) integer peak coordinates.
    '''
    a = np.asarray(a, dtype=np.float64)
    hmax = extrema.h_maxima(a, max(rolthresh, 1e-9))      # binary regional maxima w/ prominence
    lbl = label(hmax)
    peaks = []
    for region in regionprops(lbl):
        coords = region.coords                            # (npix, 2) rows,cols
        vals = a[coords[:, 0], coords[:, 1]]
        pk = coords[int(np.argmax(vals))]
        peaks.append((int(pk[0]), int(pk[1])))
    return peaks


def clear_overlapping_rolonies(lpeaks):
    '''
    Port of MATLAB lpeaks & ~imdilate(lpeaks, triu(ones(3))-diag([1 1 0])).
    The asymmetric 3x3 SE is [[0,1,1],[0,0,1],[0,0,1]]; MATLAB imdilate reflects the
    SE, so we dilate (scipy.ndimage, no implicit reflection) with the reflected SE to
    match MATLAB exactly. De-duplicates adjacent double-detections.
    '''
    se = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    se_reflected = se[::-1, ::-1]
    return lpeaks & np.logical_not(ndi.binary_dilation(lpeaks, structure=se_reflected))


def basecall_bcseq_ski(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'basecall-bcseq'

    # arity is single -> one output file for this FOV
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

    rolthresh = _parse_per_channel(cp.get(stage, 'rolthresh'), num_c)
    gaussrad = float(cp.get(stage, 'gaussrad', fallback='0'))
    bgnrad = int(cp.get(stage, 'bgnrad', fallback='0'))
    relaxed = get_boolean(cp.get(stage, 'relaxed', fallback='False'))
    logging.info(f'basecall-bcseq num_c={num_c} rolthresh={rolthresh} gaussrad={gaussrad} '
                 f'bgnrad={bgnrad} relaxed={relaxed} n_cycles={len(infiles)}')

    # tile basename (all infiles are the same tile across cycles)
    (dirpath, base, ilabel, ext) = split_path(os.path.abspath(infiles[0]))
    # MATLAB sorts the cycle files naturally; the harness already passes them in cycle order.
    seqfiles = list(infiles)

    # ---- 1. cycle-1: find rolonies per channel ----
    lim = read_image(seqfiles[0], ch_idx).astype(np.float64)   # (num_c, H, W)
    if gaussrad and gaussrad > 0:
        lim = np.stack([gaussian(lim[n], sigma=gaussrad, preserve_range=True)
                        for n in range(num_c)], axis=0)
    H, W = lim.shape[1], lim.shape[2]

    peak_rows, peak_cols = [], []
    for n in range(num_c):
        for (r, c) in find_rolonies_one_channel(lim[n], rolthresh[n]):
            peak_rows.append(r)
            peak_cols.append(c)

    # ---- 2. merge + de-duplicate peaks ----
    lpeaks = np.zeros((H, W), dtype=bool)
    if peak_rows:
        lpeaks[np.asarray(peak_rows), np.asarray(peak_cols)] = True
    lpeaks = clear_overlapping_rolonies(lpeaks)
    rows, cols = np.where(lpeaks)            # final rolony coords (row, col)
    n_rol = len(rows)
    logging.info(f'{base}: {n_rol} barcode rolonies')

    # ---- 3. read out signal across all cycles ----
    n_cyc = len(seqfiles)
    sig = np.ones((n_rol, n_cyc, num_c), dtype=np.float64)
    for m, sf in enumerate(seqfiles):
        im = read_image(sf, ch_idx).astype(np.float64)        # (num_c, H, W)
        if gaussrad and gaussrad > 0:
            im = np.stack([gaussian(im[n], sigma=gaussrad, preserve_range=True)
                           for n in range(num_c)], axis=0)
        if bgnrad and bgnrad > 0:
            im = np.stack([ball_tophat(im[n], bgnrad) for n in range(num_c)], axis=0)
        if n_rol:
            # sig[rolony, cycle, channel] = im[channel, row, col]
            sig[:, m, :] = im[:, rows, cols].T

    # ---- 4. basecall + score (MATLAB: max over channels, score = max / L2 norm) ----
    if n_rol:
        seq = np.argmax(sig, axis=2) + 1                      # 1..num_c
        maxsig = np.max(sig, axis=2)
        score = maxsig / np.sqrt(np.sum(sig ** 2, axis=2))
        score[np.isnan(score)] = 0.5
        intensity = maxsig                                    # bcint = max over channels
    else:
        seq = np.zeros((0, n_cyc), dtype=int)
        score = np.zeros((0, n_cyc))
        intensity = np.zeros((0, n_cyc))

    tile_data = {
        'lroi_x': np.asarray(rows),      # row index (axis 0) -- matches aggregate-cellids
        'lroi_y': np.asarray(cols),      # col index (axis 1)
        'seq': seq.astype(np.int8),
        'score': score,
        'sig': sig,
        'int': intensity,
    }
    out = {base: tile_data}
    logging.info(f'writing {outfile}')
    joblib.dump(out, outfile)

    # Make human-readable TSV.
    # Translate bases.
    # 
    keep_columns = ['lroi_x','lroi_y', 'seq', 'seq_hd' ]  
    dir, base, label, ext = split_path(outfile)
    outfile = os.path.join(dir, f'{base}.tsv')   

    df = pd.DataFrame()
    for data_col in list( tile_data.keys()):
        if data_col in keep_columns:
            df[data_col] = list( tile_data[data_col] )   
            if data_col == 'seq' or data_col == 'seq_hd':
                seq_bases = []
                for ar in tile_data[data_col]:
                    base_list = list(ar)
                    seq_bases.append( ''.join( [ basecall_channels[ y - 1 ] for y in base_list ] ))
                df[f'{data_col}_sequence'] = seq_bases
                df = df.drop([data_col], axis=1)
    df['tilename'] = base        
    logging.info(f'writing basecall-bcseq TSV: {outfile}')
    df.to_csv(outfile, sep='\t')
    logging.info(f'Done.')




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
                        type=str, help='unused for basecall-bcseq')
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
    logging.debug(f'Running with config={args.config}')

    basecall_bcseq_ski(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
