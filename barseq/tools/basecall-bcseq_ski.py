#!/usr/bin/env python
#
# Do basecalling on batches of images.
# used for bcseq

import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
from joblib import load, dump

import matplotlib.pylab as plt
import numpy as np

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *


def basecall_bcseq_ski( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
      
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-hyb'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')



# Claude Code

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

def _parse_per_channel_claude(value, num_c):
    '''Parse a config value that may be a scalar or a [a,b,c,d] list into a
    length-num_c list of floats (MATLAB rolthresh can be scalar or per-channel).'''
    s = str(value).strip().strip('[]')
    parts = [p for p in s.replace(',', ' ').split() if p]
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        vals = vals * num_c
    return vals[:num_c]


def find_rolonies_one_channel_claude(a, rolthresh):
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


def clear_overlapping_rolonies_claude(lpeaks):
    '''
    Port of MATLAB lpeaks & ~imdilate(lpeaks, triu(ones(3))-diag([1 1 0])).
    The asymmetric 3x3 SE is [[0,1,1],[0,0,1],[0,0,1]]; MATLAB imdilate reflects the
    SE, so we dilate (scipy.ndimage, no implicit reflection) with the reflected SE to
    match MATLAB exactly. De-duplicates adjacent double-detections.
    '''
    se = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    se_reflected = se[::-1, ::-1]
    return lpeaks & np.logical_not(ndi.binary_dilation(lpeaks, structure=se_reflected))


def basecall_bcseq_ski_claude(infiles, outfiles, stage=None, cp=None):
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
    logging.info('Done.')








# NOTEBOOK CODE

def basecall_barcodes_rolony(pth,relaxed=0,thresh=[30,30,30,30],prominence=[30,30,30,30],num_cycles=15,num_c=4):
    [folders,pos,x,y]=get_folders(pth)
    lroi_x_all=[]
    lroi_y_all=[]
    id_t_all=[]
    sig_t_all=[]
    score_t_all=[]
    seq_t_all=[]
    for folder in folders:
        print(f'BARCODE BASECALL FOLDER {folder}')
        pthw=os.path.join(pth,'processed',folder,'aligned')
        I=[]
        for i in range(num_cycles):
            I.append(tfl.imread(os.path.join(pthw,'alignedregn2vbcseq'+str("%0.2d"%(i+1))+'.tif'),key=range(0,4,1)))
        [lroi_x,lroi_y,id_t,sig_t,score_t,seq_t]=basecall_bc_one_image(pthw, num_c, I, thresh, prominence)
        lroi_x_all.append(lroi_x[0])
        lroi_y_all.append(lroi_y[0])
        id_t_all.append(id_t[0])
        sig_t_all.append(sig_t[0])
        score_t_all.append(score_t[0])
        seq_t_all.append(seq_t[0])
    score_t_all=np.array(score_t_all)
    score_t_all[np.isnan(score_t_all)]=0.5
    score_t_all=score_t_all.tolist()
    dump({"lroi_x_all":lroi_x_all,"lroi_y_all":lroi_y_all,"id_t_all":id_t_all,"sig_t_all":sig_t_all,"score_t_all":score_t_all,"seq_t_all":seq_t_all},os.path.join(pth,'processed','bc.joblib'))


def basecall_bc_one_image(pthw,num_c,I,thresh,prominence):
    """
    Basecalling function:
    1. Basecalls bc for one tile
    2. Writes segmented rolony image per tile
    3. Returns basecall results to the calling function
    """ 
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    score_t=[]
    mask=np.zeros_like(I[0])
    for n in range(num_c):
        a=I[0][n,:,:]
        a_mask=a>thresh[n]
        a_masked=a*a_mask
        a_max= extrema.h_maxima(a_masked,prominence[n])
        
        
        mask[n,:,:]=uint16m(binary_dilation(a_max))
        # OK, SO I WILL REMOVE DILATION FROM HERE AND DILATE AFTER OVERLAP REMOVAL IN NEXT LINE
    comb_mask=clear_overlapping_rolonies(np.max(mask,axis=0))
    tfl.imwrite(os.path.join(pthw,'mask_bc.tif'),mask,photometric='minisblack')
    tfl.imwrite(os.path.join(pthw,'comb_mask_bc.tif'),comb_mask,photometric='minisblack')
    label_peaks = label(comb_mask)
    m=regionprops(label_peaks,comb_mask)
    
    [lroi_x,lroi_y,id_t,sig_t,score_t,seq]=quantify_peaks_bc(lroi_x,lroi_y,id_t,sig_t,score_t,m,I)   

    dump({"lroi_x":lroi_x,"lroi_y":lroi_y,"id_t":id_t,"sig_t":sig_t,"score_t":score_t,"seq":seq},os.path.join(pthw,'basecalls-bc.joblib'))

    return(lroi_x,lroi_y,id_t,sig_t,score_t,seq)


def clear_overlapping_rolonies(a):
    c=np.array([[0,1,1],[0,0,1],[0,0,1]])
    a=a & skimage.util.invert(binary_dilation(a,c))
    return a


def quantify_peaks_bc(lroi_x,lroi_y,id_t,sig_t,score_t,m,I):
    """
    Basecalling function:
    1. Based on the regionprops results per tile, this function creates bc basecalling output and decodes the gene
    2. Returns the basecall output to the calling function
    """ 
    sig2=[]
    lroi1_x=[]
    lroi1_y=[]
    id2=[]
    score2=[]
    gene_map=np.array(list("GTAC")) 
    
    for i,peaks in enumerate(m):
        lroi1_x.append(peaks.centroid[0])
        lroi1_y.append(peaks.centroid[1])
        score1=[]
        sig1=[]
        id1=[]
        #score=0
        for j in range(len(I)):
            intensity=I[j][:,peaks.coords[0][0],peaks.coords[0][1]]
            #score[np.isnan(score)]=0.5
            sig1.append(np.max(intensity))
            id1.append(np.argmax(intensity))
            score1.append(np.max(intensity)/np.sqrt(np.sum(np.square(intensity))))
        sig2.append(sig1)
        id2.append(id1)
        score2.append(score1)
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id2)
    sig_t.append(sig2)
    score_t.append(score2)
    seq=gene_map[id_t]   
    return(lroi_x,lroi_y,id_t,sig_t,score_t,seq)

def basecall_bc_soma_all(pth,num_ch=4,mname='dil_cell_mask_cyto3.tif',fname='alignedregn2vbcseq'):
    bc_label_all=[]
    bc_sig_all_channels_all=[]
    bc_sig_all=[]
    bc_id_all=[]
    bc_score_all=[]
    bc_seq_all=[]
    [folders,pos,x,y]=get_folders(pth)

    for folder in folders:
        [bc_label,bc_sig_all_channels,bc_sig,bc_id,bc_score,bc_seq]=basecall_bc_soma_one_image(pth,folder,num_ch,mname,fname)
        bc_sig_all_channels_all.append(bc_sig_all_channels)
        bc_sig_all.append(bc_sig)
        bc_id_all.append(bc_id)
        bc_score_all.append(bc_score)
        bc_seq_all.append(bc_seq)
        bc_label_all.append(bc_label)
        print(f'Basecalled soma-bc folder {folder}')

    dump({"bc_label":bc_label_all,"bc_sig_all_channels":bc_sig_all_channels_all,"bc_sig":bc_sig_all," bc_id": bc_id_all,"bc_score":bc_score_all,"bc_seq":bc_seq_all},os.path.join(pth,'processed','all_bccells_intensity.joblib'))


def basecall_bc_soma_one_image(pth,folder,num_ch=4,mname='dil_cell_mask_cyto3.tif', fname='alignedregn2vbcseq'):
    # NUCLEAR PROFILE BASED BACKGROUND SUBTRACTION NOT DONE
    gene_map=np.array(list("GTAC"))   
    mask=tfl.imread(os.path.join(pth,'processed',folder,'aligned',mname))
    bc_label_cycles=[]
    bc_sig_all_channels_cycles=[]
    bc_sig_cycles=[]
    bc_id_cycles=[]
    bc_score_cycles=[]
    bc_seq_cycles=[]
    if np.max(mask[:])>0:
        for i in range(num_cycles):
            bc_label=[]
            bc_sig_all_channels=[]
            bc_sig=[]
            bc_id=[]
            bc_score=[]
            bc_seq=[]
            I=np.transpose(tfl.imread(os.path.join(pth,'processed',folder,'aligned',fname+str("%0.2d"%(i+1))+'.tif'),key=range(0,num_ch,1)), axes=(1,2,0))
            cell_data=regionprops_table(mask,I,properties=('label','intensity_mean'))
            bc_label=cell_data['label']
            bc_sig_all_channels=[cell_data['intensity_mean-0'],cell_data['intensity_mean-1'],cell_data['intensity_mean-2'],cell_data['intensity_mean-3']] # [channel][cells]
            bc_sig=np.max(np.array(bc_sig_all_channels),axis=0)
            bc_id=np.argmax(np.array(bc_sig_all_channels),axis=0)
            bc_score=bc_sig/np.sqrt(np.sum(np.square(bc_sig_all_channels),axis=0))
            bc_score[np.isnan(bc_score)]=0.5
            bc_seq=gene_map[bc_id]
        
            bc_label_cycles=bc_label
            bc_sig_all_channels_cycles.append(bc_sig_all_channels)
            bc_sig_cycles.append(bc_sig)
            bc_id_cycles.append(bc_id)
            bc_score_cycles.append(bc_score)
            bc_seq_cycles.append(bc_seq)
            
    dump({"bc_label":bc_label_cycles,"bc_sig_all_channels":bc_sig_all_channels_cycles,"bc_sig":bc_sig_cycles," bc_id": bc_id_cycles,"bc_score":bc_score_cycles,"bc_seq":bc_seq_cycles},os.path.join(pth,'processed',folder,'aligned','bc-somas.joblib'))
    
    return bc_label_cycles,bc_sig_all_channels_cycles,bc_sig_cycles,bc_id_cycles,bc_score_cycles,bc_seq_cycles



if __name__ == '__main__':
    FORMAT='%(asctime)s (UTC) [ %(levelname)s ] %(filename)s:%(lineno)d %(name)s.%(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-d', '--debug', 
                        action="store_true", 
                        dest='debug', 
                        help='debug logging')

    parser.add_argument('-v', '--verbose', 
                        action="store_true", 
                        dest='verbose', 
                        help='verbose logging')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')
    
    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')

    parser.add_argument('-t','--template', 
                    metavar='template',
                    default=None,
                    required=False, 
                    type=str, 
                    help='label for this stage config')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='outfile. ')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        loglevel = 'debug'
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   
        loglevel = 'info'
    
    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    logging.debug(f'Running with config={args.config}:\n{cdict}')
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    basecall_bcseq_ski( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')
