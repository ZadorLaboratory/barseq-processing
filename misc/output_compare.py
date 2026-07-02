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

POSITIONS = ['Pos1','Pos2']

POS_TILENAMES = { 'Pos1' :  ['MAX_Pos1_000_000',
                            'MAX_Pos1_000_001',
                            'MAX_Pos1_001_000',
                            'MAX_Pos1_001_001'] ,
                 'Pos2' :  ['MAX_Pos2_000_000',
                            'MAX_Pos2_000_001',
                            'MAX_Pos2_001_000',
                            'MAX_Pos2_001_001']
                }



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


    # Check all_segmentation.joblib
    print(f'\nComparing all_segmentation.joblib entry lengths/sums...') 
    as1 = joblib.load(f'{outdir1}/processed/all_segmentation.joblib')
    as2 = joblib.load(f'{outdir2}/merge/hyb/all_segmentation.joblib')    
    for tilename in TILENAMES:
        s1 = as1[tilename]
        s2 = as2[tilename]
        # dict_keys(['original_labels', 'dilated_labels', 'cell_num', 'cent_x', 'cent_y'])
        for label in ['original_labels', 'dilated_labels', 'cell_num', 'cent_x', 'cent_y']:
            c1 = s1[label]
            c2 = s2[label]
            print( f"all_segmentation[{tilename}][{label}]\t:\tpbs={len(c1)}\tbpw={len(c2)}" )
            print( f"all_segmentation[{tilename}][{label}] sum():\tpbs={c1.sum()}\tbpw={c2.sum()}" )   
    print(f'\n')    

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


    # Compare tforms_original, tforms_rescaled0p5, tforms_final
    print(f'Comparing tformsX.joblibs...') 
    for tfn in [  'tforms_original.joblib', 'tforms_rescaled0p5.joblib' ]:
        t1 = joblib.load(f'{outdir1}/processed/{tfn}')
        t2 = joblib.load(f'{outdir2}/merge/hyb/{tfn}')
        # print(f'Comparing {tfn} t1.keys={t1.keys()} t2.keys={t2.keys()}...')
        print(f'Comparing {tfn}')
        for pos in POSITIONS:
            for tilename in POS_TILENAMES[pos]:
                # print(f'Comparing pos={pos} tilename {tilename}')
                p1tn = f'{tilename}.tif'
                # print(f'p1tn = {p1tn} pos={pos} keys = {t1[pos].keys()}') 
                p1 = t1[pos][p1tn]['position'] 
                p2 = t2[pos][tilename]['position']
                print(f'[{pos}][{tilename}]\t:\tpbs={p1}\tbpw={p2}')
        print(f'\n')

    # handle tforms_final.joblib
    t1 = joblib.load(f'{outdir1}/processed/tforms_final.joblib')
    t2 = joblib.load(f'{outdir2}/merge/hyb/tforms_final.joblib')
    # print(f'Comparing tforms_final.joblib t1.keys={t1.keys()} t2.keys={t2.keys()}...')
    print(f'Comparing ')
    print(f'Comparing tforms_final.joblib...')
    for tilename in TILENAMES:
        p1 = t1[tilename] 
        p2 = t2[tilename]
        print(f'[[{tilename}].params.sum() \t:\tpbs={p1.params.sum()}\tbpw={p2.params.sum()}')
    print(f'\n')


    # Compare genehyb.joblib...
    print(f'Comparing genehyb.joblib entry lengths...') 
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
            print(f'[{tilename}][{label}] sum()\t:\tpbs={t1.sum()}\tbpw={t2.sum()}')
        print(f'\n')
    print(f'\n')


    # Compare cell_id.
    print(f'Comparing cell_id.joblib entry lengths...') 
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
            print(f'[{tilename}][{label}] sum() \t:\tpbs={c1.sum()}\tbpw={c2.sum()}')
    print(f'\n')

    # Compare processeddata.joblib
    print(f'Comparing processeddata.joblib entry lengths/sums...') 
    # ['all_data', 'filtered_data', 'expmat', 'cells', 'gene_id', 'codebook_combined'] 
    
    pd1 = joblib.load(f'{outdir1}/processed/processeddata.joblib')
    pd2 = joblib.load(f'{outdir2}/aggregated/hyb/processeddata.joblib')
    print(f'processeddata[cells]\t:\tpbs={len(pd1['cells'])}\tbpw={len(pd2['cells'])}')
    print(f'processeddata[gene_id]\t:\tpbs={len(pd1['gene_id'])}\tbpw={len(pd2['gene_id'])}')
    
    ad1 = pd1['all_data']
    ad2 = pd2['all_data']
    for label in ['cellidall' , 'sliceidall' , 'cellidall_hyb', 'sliceidall_hyb'  ]:
        print( f"processeddata[all_data][{label}]\t:\tpbs={len(ad1[label])}\tbpw={len(ad2[label])}" )
        print( f"processeddata[all_data][{label}] sum():\tpbs={ad1[label].sum()}\tbpw={ad2[label].sum()}" )


    fd1 = pd1['filtered_data']
    fd2 = pd2['filtered_data']
    for label in ['combined_gene_hyb_id', 'combined_gene_hyb_fov', 'combined_gene_hyb_cellidall', 'combined_gene_hyb_sliceidall' ]:
        print( f"processeddata[filtered_data][{label}]\t:\tpbs={len(fd1[label])}\tbpw={len(fd2[label])}" )
        print( f"processeddata[filtered_data][{label}] sum():\tpbs={fd1[label].sum()}\tbpw={fd2[label].sum()}" )

    # Compare alldata.joblib
    print(f'Comparing alldata.joblib entry lengths...') 
    ad1 = joblib.load(f'{outdir1}/processed/alldata.joblib')
    ad2 = joblib.load(f'{outdir2}/aggregated/hyb/alldata.joblib')
    logging.info(f'comparing alldata info...')
    r1 = ad1['rolonies']
    r2 = ad2['rolonies']    
    # ['id', 'pos10_x', 'pos10_y', 'pos40_x', 'pos40_y', 'slice', 'genes', 'fov', 'fov_names']
    for label in ['id', 'pos10_x', 'pos10_y', 'pos40_x', 'pos40_y', 'slice', 'fov',]:
        print( f"alldata[rolonies][{label}]\t:\tpbs={len(r1[label])}\tbpw={len(r2[label])}" )
        print( f"alldata[rolonies][{label}] sum():\tpbs={r1[label].sum()}\tbpw={r2[label].sum()}" )        

    n1 = ad1['neurons']
    n2 = ad2['neurons']
    # ['expmat', 'id', 'pos10x_x', 'pos10x_y', 'pos40x_x', 'pos40x_y', 'slice', 'genes', 'fov', 'fov_names']
    for label in ['id', 'pos10x_x', 'pos10x_y', 'pos40x_x', 'pos40x_y', 'slice', 'fov' ]:
        print( f"alldata[neurons][{label}]\t:\tpbs={len(n1[label])}\tbpw={len(n2[label])}" )
        print( f"alldata[neurons][{label}] sum():\tpbs={n1[label].sum()}\tbpw={n2[label].sum()}" )

    emo1 = ad1['neurons']['expmat']
    emo2 = ad2['neurons']['expmat']
    print(f"alldata[neurons][expmat].shape \t:\tpbs={emo1.shape}\tbpw={emo2.shape}" )
    print(f"alldata[neurons][expmat].sum() \t:\tpbs={emo1.sum()}\tbpw={emo2.sum()}" )
    print(f"alldata[neurons][expmat].getnnz() \t:\tpbs={emo1.getnnz()}\tbpw={emo2.getnnz()}" )

    print(f'\n')


    # Compares lroi10x.joblib
    print(f'\nComparing lroi10x.joblib entry lengths/sums...') 
    # ['all_data', 'filtered_data', 'expmat', 'cells', 'gene_id', 'codebook_combined'] 
    
    lx1 = joblib.load(f'{outdir1}/processed/lroi10x.joblib')
    lx2 = joblib.load(f'{outdir2}/merge/hyb/lroi10x.joblib')    
    for tilename in TILENAMES:
        t1 = lx1[tilename]
        t2 = lx2[tilename]
        for label in ['lroi10x_x', 'lroi10x_y', 'lroi10xhyb_x', 'lroi10xhyb_y', 'cellpos10x_x', 'cellpos10x_y']:
            c1 = t1[label]
            c2 = t2[label]
            print( f"lroix[{tilename}][{label}]\t:\tpbs={len(c1)}\tbpw={len(c2)}" )
            print( f"lroix[{tilename}][{label}] sum():\tpbs={c1.sum()}\tbpw={c2.sum()}" )
    print(f'\n')

    # Compare filt_neurons.joblib
    print(f'\nComparing filt_neurons.joblib entry lengths/sums...') 
    
    fn1 = joblib.load(f'{outdir1}/processed/filt_neurons.joblib')
    fn2 = joblib.load(f'{outdir2}/aggregated/hyb/filt_neurons.joblib')    
    s1 = fn1['filt_neurons']
    s2 = fn2['filt_neurons']
    # ['expmat', 'id', 'pos10x_x', 'pos10x_y', 'pos40x_x', 'pos40x_y', 'slice', 'genes', 'fov', 'fov_names']
    for label in ['id', 'pos10x_x', 'pos10x_y', 'pos40x_x', 'pos40x_y', 'slice','fov']:
        c1 = s1[label]
        c2 = s2[label]
        print( f"filt_neurons[filt_neurons][{label}]\t:\tpbs={len(c1)}\tbpw={len(c2)}" )
        print( f"filt_neurons[filt_neurons][{label}] sum():\tpbs={c1.sum()}\tbpw={c2.sum()}" )
    s1 = fn1['removecells_all']
    s2 = fn2['removecells_all']    
    print( f"filt_neurons[removecells_all]\t:\tpbs={len(s1)}\tbpw={len(s2)}" )
    print( f"filt_neurons[removecells_all] sum():\tpbs={s2.sum()}\tbpw={s2.sum()}" )    
    print(f'\n')


    # Handle final genes x cells. 
    print('\nComparing final cells x genes matrices...')
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
    print(f'Cells found: pbs = {len(df1)} bpw = {len(df2)}')
    if len(df1) != len(df2):
        identical = False
    #print(f'pbs top 15: {list( df1.sum().sort_values(ascending = False).index )[0:15]}')
    #print(f'bpw top 15: {list( df2.sum().sort_values(ascending = False).index )[0:15]}')
    print(f'pbs top 20:\n{df1.sum().sort_values(ascending = False)[:20]}')
    print(f'bpw top 20:\n{df2.sum().sort_values(ascending = False)[:20]}\n')
    print(f'Spearman rank correlation cells x genes: {spearman_single:.4f}\n')
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
        print('\nOutput trees are identical. ')
    else:
        print('\nOutput trees differ. ')