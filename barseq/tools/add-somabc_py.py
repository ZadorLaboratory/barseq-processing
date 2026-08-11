#!/usr/bin/env python
#
# Add soma barcodes to filt_neurons, flag barcoded cells, and add single-rolony
# barcodes. Faithful port of MATLAB add_somabc.m + filter_somabc_xt.m + add_singlebc.m.
#
# Inputs (gathered via merge-dummy-bc):
#   filt_neurons.joblib (aggregated/hyb)  -> {'filt_neurons': {...}}
#   soma-bc.joblib      (aggregated/bcseq) -> per-cell soma barcodes (global id)
#   bc-rolonies.joblib  (aggregated/bcseq) -> per-rolony seq + global cellid
# Output: filt_neurons-bc-somas.joblib (aggregated/hyb), filt_neurons augmented with
#   soma_bc*, is_barcoded, dom_bc*, all_bc*.
#
import argparse
import joblib
import logging
import os
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.utils import *
from barseq.imageutils import *


def lingseqcomplexity(seq):
    '''
    Port of MATLAB lingseqcomplexity.m (Trifonov 1990 linguistic complexity).
    seq is a string; returns the vector u of per-length vocabulary-usage ratios.
    '''
    L = len(seq)
    u = np.ones(L)
    for i in range(1, L + 1):
        subs = set(seq[n:n + i] for n in range(0, L - i + 1))
        c = len(subs)
        w = min(4 ** i, L - i + 1)
        u[i - 1] = c / w
    return u


def seq_complexity(seq_row):
    '''log10(prod(lingseqcomplexity(char(48+seq)))) -- MATLAB filter_somabc_xt.
    seq_row holds basecalls 1-4; char(48+x) makes the digit string '1'..'4'.'''
    s = ''.join(chr(48 + int(x)) for x in seq_row)
    return np.log10(np.prod(lingseqcomplexity(s)))


def hamming_count(a, b):
    '''number of differing positions between two equal-length int sequences.'''
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def collapse_barcodes(bc_rows, err_corr_thresh):
    '''
    Port of the add_singlebc Hamming-collapse: given all rolony barcodes in a cell
    (bc_rows, shape (n, ncyc)), find unique rows + counts, then greedily merge each
    (count-descending) into the nearest accepted barcode if Hamming distance (in #
    positions) <= err_corr_thresh, else accept as new. Returns (bclist, bccount).
    '''
    uniq, inv = np.unique(bc_rows, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    order = np.argsort(-counts)                  # descending
    uniq = uniq[order]
    counts = counts[order]

    bclist = [uniq[0]]
    bccount = [int(counts[0])]
    for m in range(1, len(uniq)):
        dists = [hamming_count(bl, uniq[m]) for bl in bclist]
        j = int(np.argmin(dists))
        if dists[j] <= err_corr_thresh:
            bccount[j] += int(counts[m])
        else:
            bclist.append(uniq[m])
            bccount.append(int(counts[m]))
    return bclist, bccount


def add_somabc_py(infiles, outfiles, stage=None, cp=None):
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'add-somabc'

    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    complexity_thresh = float(cp.get(stage, 'complexity_thresh'))
    score_thresh = float(cp.get(stage, 'score_thresh'))
    sig_thresh = float(cp.get(stage, 'signal_thresh'))
    count_thresh = get_config_none(cp, stage, 'count_thresh')
    count_thresh = None if count_thresh is None else int(count_thresh)
    err_corr_thresh = int(cp.get(stage, 'err_corr_thresh'))

    # NOTE: select_input_files returns files ordered by sorted KEY name:
    # bcrol < filt < soma.
    input_map = {'filt': 'filt_neurons.joblib',
                 'soma': 'soma-bc.joblib',
                 'bcrol': 'bc-rolonies.joblib'}
    (bcrol_file, filt_file, soma_file) = select_input_files(infiles, input_map)
    filt_data = joblib.load(filt_file)
    fn = filt_data['filt_neurons']
    soma = joblib.load(soma_file)
    bcrol = joblib.load(bcrol_file)

    fn_id = np.asarray(fn['id'], dtype=np.int64)
    n_neurons = len(fn_id)

    # ---- add_somabc: match each filt_neuron id to its soma barcode row ----
    soma_id = np.asarray(soma['id'], dtype=np.int64)
    id_to_row = {int(cid): r for r, cid in enumerate(soma_id)}
    I = np.array([id_to_row.get(int(c), -1) for c in fn_id])
    n_missing = int(np.sum(I < 0))
    if n_missing > 0:
        logging.warning(f'{n_missing}/{n_neurons} filt_neuron ids have no soma barcode '
                        f'(filled with no-call). MATLAB add_somabc would error here.')
    safe = np.where(I < 0, 0, I)

    def take(arr):
        out = arr[safe].copy()
        if n_missing > 0:
            out[I < 0] = 0
        return out

    fn['soma_bc'] = take(soma['seq'])
    fn['soma_bc_sig'] = take(soma['sig'])
    fn['soma_bc_score'] = take(soma['score'])
    fn['soma_bc_hd'] = take(soma['seq_hd'])
    fn['soma_bc_sig_hd'] = take(soma['sig_hd'])
    fn['soma_bc_score_hd'] = take(soma['score_hd'])

    # ---- filter_somabc_xt: complexity + 3rd-lowest score + 3rd-lowest signal ----
    soma_bc = fn['soma_bc']
    n_cyc = soma_bc.shape[1] if soma_bc.size else 0
    bc_complexity = np.array([seq_complexity(soma_bc[i]) for i in range(n_neurons)]) \
        if n_neurons and n_cyc else np.full(n_neurons, -np.inf)

    score1s = np.sort(fn['soma_bc_score'], axis=1)                       # ascending
    sig1m = np.sort(np.max(fn['soma_bc_sig'], axis=1), axis=1)           # max over channel, then sort over cycle
    k = min(n_cyc, 3) - 1 if n_cyc else 0
    pass_complexity = bc_complexity >= complexity_thresh
    high_score = score1s[:, k] >= score_thresh
    high_sig = sig1m[:, k] >= sig_thresh
    fn['is_barcoded'] = pass_complexity & high_score & high_sig
    logging.info(f'is_barcoded: {int(np.sum(fn["is_barcoded"]))}/{n_neurons} cells')

    # ---- add_singlebc ----
    dom_bc = [None] * n_neurons
    dom_bc_count = np.zeros(n_neurons, dtype=int)
    all_bc = [None] * n_neurons
    all_bc_count = [None] * n_neurons

    if count_thresh is None:
        # Sindbis path: dominant barcode = the soma barcode itself.
        for n in range(n_neurons):
            dom_bc[n] = soma_bc[n]
            all_bc[n] = soma_bc[n]
            dom_bc_count[n] = 1
            all_bc_count[n] = [1]
    else:
        # rabies single-bc path: collapse rolony barcodes assigned to each cell.
        bc_cellid = np.asarray(bcrol['cellid'], dtype=np.int64)
        bc_seq = np.asarray(bcrol['seq'])
        for n, cid in enumerate(fn_id):
            in_cell = bc_cellid == int(cid)
            if int(np.sum(in_cell)) > count_thresh:
                rows = bc_seq[in_cell, :]
                bclist, bccount = collapse_barcodes(rows, err_corr_thresh)
                jmax = int(np.argmax(bccount))
                if bccount[jmax] > count_thresh:
                    dom_bc[n] = np.asarray(bclist[jmax], dtype=np.int8)
                    dom_bc_count[n] = bccount[jmax]
                    all_bc[n] = np.asarray(bclist, dtype=np.int8)
                    all_bc_count[n] = list(bccount)
                else:
                    all_bc_count[n] = [0]
            else:
                all_bc_count[n] = [0]

    fn['dom_bc'] = np.array(dom_bc, dtype=object)
    fn['dom_bc_count'] = dom_bc_count
    fn['all_bc'] = np.array(all_bc, dtype=object)
    fn['all_bc_count'] = np.array(all_bc_count, dtype=object)

    out = {'filt_neurons': fn}
    logging.info(f'writing {outfile}')
    joblib.dump(out, outfile)

    # MATLAB run_barseq copies bc-rolonies.mat to output; it is already produced by
    # aggregate-bcseq in aggregated/bcseq. Re-dump alongside the final neurons for convenience.
    bcrol_out = os.path.join(outdir, 'bc-rolonies.joblib')
    if not os.path.exists(bcrol_out):
        joblib.dump(bcrol, bcrol_out)
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

    add_somabc_py(infiles=args.infiles, outfiles=args.outfiles, stage=args.stage, cp=cp)
    logging.info(f'done processing output to {args.outfiles[0]}')
