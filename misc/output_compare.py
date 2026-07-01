#!/usr/bin/env python
#
# Compare MATLAB/Pybarseq output to Python pipeline output. 
#  
import argparse
import joblib
import json
import logging
import os
import pprint
import shutil
import sys

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

N_GENESEQ_CYCLES = 7
N_HYB_CYCLES = 1

TILENAMES = [ 
        'MAX_Pos1_000_000',
        'MAX_Pos1_000_001',
        'MAX_Pos1_001_000',
        'MAX_Pos1_001_001',
        'MAX_Pos2_000_000',
        'MAX_Pos2_000_001',
        'MAX_Pos2_001_000',
        'MAX_Pos2_001_001'
]

PBS_PREFIXES = ['original/',
                'backsub/backsub',
                'chalign/chalign',
                'bleedthrough/bleedthrough',
                'aligned/aligned',
]

BSP_PREFIXES = ['denoised',
                'background',
                'regchannels',
                'bleedthrough',
                'regcycle'
                ]


def do_compare_output(outdir1, outdir2):
    '''
    outdir1  pybarseq
    outdir2  barseq-processing
    '''
    outdir1 = os.path.abspath(outdir1)
    outdir2 = os.path.abspath(outdir2)

    identical = True
    # Handle preprocessing
    for i, pbs_prefix in enumerate(PBS_PREFIXES):
        if identical:
            bsp_prefix = BSP_PREFIXES[i]
            for gcycle in list( range(1, N_GENESEQ_CYCLES + 1)):
                fname = f'n2vgeneseq0{gcycle}.tif'
                for tilename in TILENAMES:
                    file1 = f'{outdir1}/processed/{tilename}/{pbs_prefix}{fname}'
                    rel1 = os.path.relpath(file1)
                    file2 = f'{outdir2}/{bsp_prefix}/geneseq0{gcycle}/{tilename}.tif'
                    rel2 = os.path.relpath(file2)
                    logging.info(f'comparing {file1} and {file2}')
                    ident, msg, min_sim = do_compare_images(file1, file2)
                    if ident:
                        print( f' {rel1} == {rel2}' )
                    else:
                        identical = False
                        print( f' {rel1} != {rel2} min_sim = {min_sim}' )
            
            for hcycle in list(range(1, N_HYB_CYCLES + 1)):
                fname = f'n2vhyb0{hcycle}.tif'
                for tilename in TILENAMES:
                    file1 = f'{outdir1}/processed/{tilename}/{pbs_prefix}{fname}'
                    rel1 = os.path.relpath(file1)
                    file2 = f'{outdir2}/{bsp_prefix}/hyb0{hcycle}/{tilename}.tif'
                    rel2 = os.path.relpath(file2)
                    logging.info(f'comparing {file1} and {file2}')
                    ident, msg, min_sim = do_compare_images(file1, file2)
                    if ident:
                        print( f' {rel1} == {rel2}' )
                    else:
                        identical = False
                        print( f' {rel1} != {rel2} min_sim = {min_sim}' )
        else:
            print(f'outputs have diverged. Exitting.')
            sys.exit(1)

    # Handle segmentation. 
    # processed/<tile>/aligned/cell_mask_cyto3.tif
    #   vs
    # segment/hyb/<tile>.cp_mask_cyto3.tif 
    for hcycle in list(range(1, N_HYB_CYCLES + 1)):
        for tilename in TILENAMES:
            file1 = f'{outdir1}/processed/{tilename}/aligned/cell_inp2.tif'
            rel1 = os.path.relpath(file1)
            file2 = f'{outdir2}/segment/hyb/{tilename}.cellpose_input.tif'
            rel2 = os.path.relpath(file2)
            logging.info(f'comparing {file1} and {file2}')
            ident, msg, min_sim = do_compare_images(file1, file2)
            if ident:
                print( f' {rel1} == {rel2}' )
            else:
                identical = False
                print( f' {rel1} != {rel2} min_sim = {min_sim}' )

    for hcycle in list(range(1, N_HYB_CYCLES + 1)):
        for tilename in TILENAMES:
            file1 = f'{outdir1}/processed/{tilename}/aligned/cell_mask_cyto3.tif'
            rel1 = os.path.relpath(file1)
            file2 = f'{outdir2}/segment/hyb/{tilename}.cp_mask_cyto3.tif'
            rel2 = os.path.relpath(file2)
            logging.info(f'comparing {file1} and {file2}')
            ident, msg, min_sim = do_compare_images(file1, file2)
            if ident:
                print( f' {rel1} == {rel2}' )
            else:
                identical = False
                print( f' {rel1} != {rel2} min_sim = {min_sim}' )

    # Check bardensr threshold
    file1 =  f'{outdir1}/processed/thresh_refined.txt'
    file2 =  f'{outdir2}/basecall/geneseq/bardensrparams.json'

    with open(file1, 'r') as f:
        sval = f.read()
        tval1 = float(sval)

    with open(file2, 'r') as f:
        data = json.load(f)
        tval2 = data['intensity_thresh_refined']
    
    dp = calc_proportion(tval1, tval2)
    print(f'\nbardensr threshold pbs={tval1} bpw={tval2} similarity = {dp}') 
    
    # Check bardensr geneseq spot calling.
    min_sim = 1.0 
    for tilename in TILENAMES:
        file1 = f'{outdir1}/processed/{tilename}/aligned/bardensrresult.csv'
        rel1 = os.path.relpath(file1)
        file2 = f'{outdir2}/basecall/geneseq/{tilename}.bardensrresult.csv'
        rel2 = os.path.relpath(file2)
        logging.info(f'comparing {file1} and {file2}')
        f1df = pd.read_csv(rel1)
        f2df = pd.read_csv(rel2)
        dp = calc_proportion(len(f1df),len(f2df) )
        print(f'{tilename} : pbs = {len(f1df)} bpw ={len(f2df)} similarity = {dp}')
        if dp < min_sim:
            min_sim = dp
    print(f'bardensr results. min_similarity = {min_sim}\n')
    
    # Compare hyb basecalling. 
    hyb_sim = 1.0
    for tilename in TILENAMES:
        file1 = f'{outdir1}/processed/{tilename}/aligned/mask_hyb.tif'
        rel1 = os.path.relpath(file1)
        file2 = f'{outdir2}/basecall/hyb/{tilename}.mask_hyb.tif'
        rel2 = os.path.relpath(file2)
        logging.info(f'comparing {file1} and {file2}')
        ident, msg, min_sim = do_compare_images(file1, file2)
        if ident:
            print( f' {rel1} == {rel2}' )
        else:
            identical = False
            print( f' {rel1} != {rel2} min_sim = {min_sim}' )
        if min_sim < hyb_sim:
            min_sim = hyb_sim
    print(f'basecall mask_hyb results. min_similarity = {hyb_sim}\n')

    for tilename in TILENAMES:
        file1 = f'{outdir1}/processed/{tilename}/aligned/basecall_map_hyb.tif'
        rel1 = os.path.relpath(file1)
        file2 = f'{outdir2}/basecall/hyb/{tilename}.basecall_map_hyb.tif'
        rel2 = os.path.relpath(file2)
        logging.info(f'comparing {file1} and {file2}')
        ident, msg, min_sim = do_compare_images(file1, file2)
        if ident:
            print( f' {rel1} == {rel2}' )
        else:
            identical = False
            print( f' {rel1} != {rel2} min_sim = {min_sim}' )
        if min_sim < hyb_sim:
            min_sim = hyb_sim
    print(f'basecall basecall_map_hyb results. min_similarity = {hyb_sim}\n')


    # Compare genehyb.joblib...
    print(f'Comparing genehyb.joblib entries...') 
    co1 = joblib.load(f'{outdir1}/processed/genehyb.joblib')
    co2 = joblib.load(f'{outdir2}/merge/hyb/genehyb.joblib')
    logging.info(f'comparing genehyb info...')
  
    for i, tilename in enumerate( TILENAMES ):
        for label in ['lroi_x','lroi_y','gene_id','signal']:
            t1 = co1[label][i][0]
            #for j in range(0,3):
                # t2 = co2[tilename][label][j]
                # print(f'[{tilename}][{label}]\t:\tpbs={len(t1)}\tbpw[{j}]={len(t2)}')
            t2 = co2[tilename][label]
            print(f'[{tilename}][{label}]\t:\tpbs={len(t1)}\tbpw={len(t2)}')
        print(f'\n')
    print(f'\n')


    # Compare cell_id.
    print(f'Comparing cell_id.joblib entries...') 
    co1 = joblib.load(f'{outdir1}/processed/cell_id.joblib')
    co2 = joblib.load(f'{outdir2}/merge/hyb/cell_id.joblib')
    logging.info(f'comparing cell_id info...')
    for tilename in TILENAMES:
        t1 = co1[tilename]
        t2 = co2[tilename]
        for label in ['cellid','cellidhyb']:
            c1 = t1[label]
            c2 = t2[label]
            print(f'[{tilename}][{label}]\t:\tpbs={len(c1)}\tbpw={len(c2)}')
    print(f'\n')


    # Handle final genes x cells. 
    pbscbg_file = f'{outdir1}/processed/filt_cellsbygenes.tsv'
    df1 = pd.read_csv(pbscbg_file, sep='\t', index_col=0)

    fcbg_file = f'{outdir2}/aggregated/hyb/YWT011357_4T.filt_cellsbygenes.tsv'
    df2 = pd.read_csv(fcbg_file, sep='\t', index_col=0)

    df1vals = list(df1.sum())
    df2vals = list(df2.sum())
    data = {
        'pbs' :  df1vals ,
        'bpw' :  df2vals  
    }
    df = pd.DataFrame(data)
    spearman_single = df['pbs'].corr(df['bpw'], method='spearman')
    print(f'cells found: pbs = {len(df1)} bpw = {len(df2)}')
    print(f'spearman rank correlation cells x genes: {spearman_single:.3f}')

    return identical

def get_image_info(infiles):
    '''
    
    '''
    for infile in infiles:
        logging.debug(f'handling {infile}')
        infile_image = read_image(infile)
        #logging.debug(f'{infile_image}')
        n_channels = len(infile_image)
        print(f'{infile}: {n_channels} channels.')
        for i in range(n_channels):
            chi = infile_image[i]
            shp = chi.shape
            dtp = str( chi.dtype )
            print(f'    [{i}] {shp} {dtp}')
          

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

    parser.add_argument('outdir1' ,
                        metavar='outdir1', 
                        type=str,
                        help='MATLAB/Pybarseq output tree. ')     

    parser.add_argument('outdir2' ,
                        metavar='outdir2', 
                        type=str, 
                        help='barseq-processing output tree.')

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    identical = do_compare_output(args.outdir1, args.outdir2)
    if identical:
        print('Output trees are identical. ')
    else:
        print('Output trees differ. ')