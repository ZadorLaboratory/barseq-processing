#!/usr/bin/env python
#
# Aggregate data
# 
import argparse
import datetime 
import joblib
import logging
import math
import os
import re
import pprint
import sys
import datetime as dt
from configparser import ConfigParser

import numpy as np
from natsort import natsorted as nsort

from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def make_pos_id_map( tilename_list, image_regex, position_group):
    '''
    make map (dict) from tilename to position INDEX (starting at 0)
    e.g.
    MAX_Pos1_000_000 -> 0
    MAX_Pos2_000_000 -> 1

    Retain order. 
    Tolerate non-integer position identifiers. 
    '''
    tilename_list = nsort(tilename_list)
    pos_list = []
    pos_id_map = {}
    for tilename in tilename_list:
        m = re.search(image_regex, tilename)
        if m is not None:
            pos = m.group(position_group)
            pos_list.append(pos)
        else:
            logging.error(f'unable to parse {tilename} for position!')
    unique_pos = list(dict.fromkeys(pos_list))
    index_list = []
    for i, tilename in enumerate(tilename_list):
        p = pos_list[i]
        pos_id_map[tilename] = unique_pos.index(p)
    return pos_id_map



def aggregate_data_py(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #
    # main output : processeddata.joblib
    #
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-data'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile
    # primary outfile is processeddata.joblib
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    project_id = cp.get( 'project','project_id')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    starting_slice_idx = cp.getint( stage, 'starting_slice_idx')
    starting_fov_idx = cp.getint( stage, 'starting_fov_idx')
    dummy_cell_num =  cp.getint( stage, 'dummy_cell_num')
    tilesize = cp.getint( stage, 'tilesize' ) 
    fraction_border= cp.getfloat( stage, 'fraction_border')
    image_regex = cp.get('barseq', 'file_regex')
    position_group = cp.getint('barseq', 'position_group')

    today=datetime.date.today().strftime('%d%m%Y')

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    # 'basecalls.joblib'.  'all_segmentation.joblib'   'genehyb.joblib' ...
    # return order from select function will be alphabetical by key name.  
    input_map = {   'cellid'  :  'cell_id.joblib',
                    'coord'   :  'lroi10x.joblib',
                    'gene_rol':  'basecalls.joblib',
                    'hyb_rol' :  'genehyb.joblib',
                    'seg'     :  'all_segmentation.joblib',
                    'tforms'  :  'tforms_final.joblib',
                  }
    codebook_hyb_file = os.path.join(resource_dir, 'codebook_hyb.tsv')
    codebook_hyb = pd.read_csv(codebook_hyb_file, sep='\t', index_col=0)
    codebook_geneseq_file = os.path.join(resource_dir, 'codebook_geneseq.tsv')
    codebook_geneseq = pd.read_csv(codebook_geneseq_file, sep='\t', index_col=0)
    
    (cell_id_file, coord_file, gene_rol_file, hyb_rol_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    cell_id = joblib.load(cell_id_file)
    coord = joblib.load(coord_file)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)
    tform_final =joblib.load(tforms_file)
    logging.debug(f'loaded input joblibs.')

    joblib.dump([codebook_hyb.to_numpy() ],os.path.join(outdir, 'hyb_codebook.joblib'))
    joblib.dump( [ codebook_geneseq.to_numpy() ], os.path.join(outdir, 'codebook.joblib'))
    #codebook_hyb = joblib.load(os.path.join(outdir, 'hyb_codebook.joblib'))  

    d={}
    d=data_dict_organizer(d,'initialize',fov=[], gene_rol_id=[],
                          pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],cellidall=[],sliceidall=[],
                          hyb_rol_id=[],fov_hyb=[],
                          pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],
                          cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    tilename_list = nsort( list(seg.keys()) )
    pos_id_map = make_pos_id_map( tilename_list, image_regex, position_group)

    for i, tilename in enumerate( tilename_list) :
        logging.debug(f'handling {tilename}') 
        pos_id = np.array([ pos_id_map[tilename] ])
        logging.debug(f'handling tile id: {tilename} i={i} pos_id = {pos_id} ')
        d=data_dict_organizer(d,'append',
                              fov = np.full(len(gene_rol[tilename]['gene_id']),i),
                              gene_rol_id = np.array(gene_rol[tilename]['gene_id']),
                              pos_10x_allx = coord[tilename]['lroi10x_x'],
                              pos_10x_ally = coord[tilename]['lroi10x_y'],
                              pos_40x_allx = np.array(gene_rol[tilename]['lroi_x']),
                              pos_40x_ally = np.array(gene_rol[tilename]['lroi_y']),

                              # if len(cellid[tilename]['cellid']) else np.array([0]),
                              cellidall = np.array( cell_id[tilename]['cellid'] ) + np.array(i * starting_fov_idx * dummy_cell_num), 
                              
                              # check this later,does it require -1 or not
                              sliceidall = np.full(len(gene_rol[tilename]['gene_id']) , pos_id + starting_slice_idx ),  
                              
                              hyb_rol_id = hyb_rol[tilename]['gene_id'][0][0],

                              # possible mismatch? nested list in our hyb_rol vs. notebook?
                              fov_hyb = np.full(len( hyb_rol[tilename]['gene_id'][0][0] ),i),
                              
                              pos_10x_allx_hyb = coord[tilename]['lroi10xhyb_x'],
                              pos_10x_ally_hyb = coord[tilename]['lroi10xhyb_y'],
                              pos_40x_allx_hyb = hyb_rol[tilename]['lroi_x'][0][0],
                              pos_40x_ally_hyb = hyb_rol[tilename]['lroi_y'][0][0],

                              cellidall_hyb = np.array(cell_id[tilename]['cellidhyb']) + np.array(i * starting_fov_idx * dummy_cell_num ),
                              sliceidall_hyb = np.full(len(hyb_rol[tilename]['gene_id'][0][0]), pos_id + starting_slice_idx ),
                              cell_list_all=np.array(seg[tilename]['cell_num']) + np.array(i * starting_fov_idx * dummy_cell_num ),
                              
                              cell_pos_10x_allx=coord[tilename]['cellpos10x_x'],
                              cell_pos_10x_ally=coord[tilename]['cellpos10x_y'],
                              
                              cell_pos_40x_allx=seg[tilename]['cent_x'],
                              cell_pos_40x_ally=seg[tilename]['cent_y'],
                              
                              fov_cell=np.full(len(seg[tilename]['cell_num']),i),
                              sliceidall_cell=np.full(len(seg[tilename]['cell_num']), pos_id + starting_slice_idx))

    logging.debug(f'Done appending. Concatenating tile arrays')
    d=data_dict_organizer(d,'concat', fov=[])
    d=data_dict_organizer(d,'concat', gene_rol_id=[])
    d=data_dict_organizer(d,'concat', pos_10x_allx=[])
    d=data_dict_organizer(d,'concat', pos_10x_ally=[])
    d=data_dict_organizer(d,'concat', pos_40x_allx=[])
    d=data_dict_organizer(d,'concat', pos_40x_ally=[])      
    d=data_dict_organizer(d,'concat', cellidall=[])
    d=data_dict_organizer(d,'concat', sliceidall=[])
    d=data_dict_organizer(d,'concat', hyb_rol_id=[])
    d=data_dict_organizer(d,'concat', fov_hyb=[])
    d=data_dict_organizer(d,'concat', pos_10x_allx_hyb=[])
    d=data_dict_organizer(d,'concat', pos_10x_ally_hyb=[])
    d=data_dict_organizer(d,'concat', pos_40x_allx_hyb=[])
    d=data_dict_organizer(d,'concat', pos_40x_ally_hyb=[])
    d=data_dict_organizer(d,'concat', cellidall_hyb=[])
    d=data_dict_organizer(d,'concat', sliceidall_hyb=[])
    d=data_dict_organizer(d,'concat', cell_list_all=[])
    d=data_dict_organizer(d,'concat', cell_pos_10x_allx=[])
    d=data_dict_organizer(d,'concat', cell_pos_10x_ally=[])
    d=data_dict_organizer(d,'concat', cell_pos_40x_allx=[])
    d=data_dict_organizer(d,'concat', cell_pos_40x_ally=[])
    d=data_dict_organizer(d,'concat', fov_cell=[])
    d=data_dict_organizer(d,'concat', sliceidall_cell=[])

    # Original codebook structure:
    #  codebook_geneseq   
    #.     [0] -> array len=111   
    #              -> array ['Rorb'] , ['GCTAGAG']
    #      [1] -> array len=111
    #              -> array ['Rorb'], [1,0,0,0,1, ...]  length=28 uint8
    codebook_combined = pd.concat([ codebook_geneseq, codebook_hyb], axis=0 )
    codebook_combined.reset_index(inplace=True, drop=True)

    d['hyb_rol_id1'] = d['hyb_rol_id'] + len(codebook_geneseq)

    logging.info('Merging dicts...')
    d=merge_gene_hyb_dict(d,'gene_rol_id','hyb_rol_id1','combined_gene_hyb_id')
    d=merge_gene_hyb_dict(d,'fov','fov_hyb','combined_gene_hyb_fov')
    d=merge_gene_hyb_dict(d,'pos_10x_allx','pos_10x_allx_hyb','combined_gene_hyb_pos10x_x')
    d=merge_gene_hyb_dict(d,'pos_10x_ally','pos_10x_ally_hyb','combined_gene_hyb_pos10x_y')
    d=merge_gene_hyb_dict(d,'pos_40x_allx','pos_40x_allx_hyb','combined_gene_hyb_pos40x_x')
    d=merge_gene_hyb_dict(d,'pos_40x_ally','pos_40x_ally_hyb','combined_gene_hyb_pos40x_y')
    d=merge_gene_hyb_dict(d,'cellidall','cellidall_hyb','combined_gene_hyb_cellidall')
    d=merge_gene_hyb_dict(d,'sliceidall','sliceidall_hyb','combined_gene_hyb_sliceidall') 
    
    border_size=np.round(fraction_border * tilesize)
    pos_id=d['combined_gene_hyb_id']>0 # uncalled rolonies--how does this happen? what's bardensr's code for uncalled ones
    pos_inside_border_x=(d['combined_gene_hyb_pos40x_x'] > border_size-1) & (d['combined_gene_hyb_pos40x_x'] < tilesize-border_size+1)
    pos_inside_border_y=(d['combined_gene_hyb_pos40x_y'] > border_size-1) & (d['combined_gene_hyb_pos40x_y'] < tilesize-border_size+1)
    filter_id=pos_id & pos_inside_border_x & pos_inside_border_y

    filtered_d={}
    filtered_d=data_dict_organizer(filtered_d,'initialize', combined_gene_hyb_id=[], combined_gene_hyb_fov=[], 
                                   combined_gene_hyb_pos10x_x=[], combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],
                                   combined_gene_hyb_cellidall=[],
                                   combined_gene_hyb_sliceidall=[])

    filtered_d=data_dict_organizer(filtered_d,'append',
                                   combined_gene_hyb_id=d['combined_gene_hyb_id'][filter_id],
                                   combined_gene_hyb_fov=d['combined_gene_hyb_fov'][filter_id],
                                   combined_gene_hyb_pos10x_x=d['combined_gene_hyb_pos10x_x'][filter_id],
                                   combined_gene_hyb_pos10x_y=d['combined_gene_hyb_pos10x_y'][filter_id],
                                   combined_gene_hyb_pos40x_x=d['combined_gene_hyb_pos40x_x'][filter_id],
                                   combined_gene_hyb_pos40x_y=d['combined_gene_hyb_pos40x_y'][filter_id],
                                   combined_gene_hyb_cellidall=d['combined_gene_hyb_cellidall'][filter_id],
                                   combined_gene_hyb_sliceidall=d['combined_gene_hyb_sliceidall'][filter_id])
    
    filtered_d=data_dict_organizer(filtered_d,'concat',
                                   combined_gene_hyb_id=[],combined_gene_hyb_fov=[],
                                   combined_gene_hyb_pos10x_x=[],combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],
                                   combined_gene_hyb_cellidall=[],
                                   combined_gene_hyb_sliceidall=[])

    cells=d['cell_list_all'].copy() # check if copy messed something
    genes=np.unique(d['combined_gene_hyb_id'])
    rol_id=d['combined_gene_hyb_id'].copy()
    rol_cell=d['combined_gene_hyb_cellidall'].copy()
    v=pd.crosstab(rol_cell, rol_id, rownames=['cell_index'], colnames=['genes'], dropna=False)
    v=v.reindex(index=cells, columns=genes, fill_value=0)
    exp_m=coo_matrix(v.to_numpy())
    processed_data={'all_data':d,
                    'filtered_data':filtered_d,
                    'expmat': exp_m,
                    'cells': cells,
                    'gene_id': genes,
                    'codebook_combined': codebook_combined
                    }
    logging.info(f'Writing output to {outfile}')
    joblib.dump(processed_data, outfile)

    rolonies={'id':filtered_d['combined_gene_hyb_id'],
              'pos10_x':filtered_d['combined_gene_hyb_pos10x_x'],
              'pos10_y':filtered_d['combined_gene_hyb_pos10x_y'],
              'pos40_x':filtered_d['combined_gene_hyb_pos40x_x'],
              'pos40_y':filtered_d['combined_gene_hyb_pos40x_y'],
              'slice':filtered_d['combined_gene_hyb_sliceidall'],
              'genes':codebook_combined,
              'fov':filtered_d['combined_gene_hyb_fov'],
              'fov_names':tilename_list}

    neurons={'expmat':exp_m,
             'id':d['cell_list_all'],
             'pos10x_x':d['cell_pos_10x_allx'],
             'pos10x_y':d['cell_pos_10x_ally'],
             'pos40x_x':d['cell_pos_40x_allx'],
             'pos40x_y':d['cell_pos_40x_ally'],
             'slice':d['sliceidall_cell'],
             'genes':codebook_combined,
             'fov':d['fov_cell'],
             'fov_names':tilename_list}

    alldata = {"rolonies":rolonies, "neurons":neurons} 
    joblib.dump( alldata, os.path.join(outdir, 'alldata.joblib'))
    logging.info('ALL DATA IS ORGANIZED')

    logging.info(f'Writing out data subsets...')    
    # Output subsets...
    # create individual output data files. DFs. Pandas matrix.
    #
    of = os.path.join( outdir, f'{project_id}.cellsbygenes.tsv')
    v.to_csv(of, sep='\t') 
    logging.info(f'Wrote cells X genes matrix to {of}')

    of = os.path.join(outdir, f'{project_id}.codebook_combined.tsv') 
    codebook_combined.to_csv(of, sep='\t')
    logging.info(f'Wrote combined codebook to {of}')

    logging.info(f'Done.')



def aggregate_data_py_tileindex(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #
    # main output : processeddata.joblib
    #
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-data'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile
    # primary outfile is processeddata.joblib
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    project_id = cp.get( 'project','project_id')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    starting_slice_idx = cp.getint( stage, 'starting_slice_idx')
    starting_fov_idx = cp.getint( stage, 'starting_fov_idx')
    dummy_cell_num =  cp.getint( stage, 'dummy_cell_num')
    tilesize = cp.getint( stage, 'tilesize' ) 
    fraction_border= cp.getfloat( stage, 'fraction_border')

    today=datetime.date.today().strftime('%d%m%Y')

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    #   'basecalls.joblib'.  'all_segmentation.joblib'   'genehyb.joblib' ...
    # return order from select function will be alphabetical by key name.  
    input_map = {   'cellid'  :  'cell_id.joblib',
                    'coord'   :  'lroi10x.joblib',
                    'gene_rol':  'basecalls.joblib',
                    'hyb_rol' :  'genehyb.joblib',
                    'seg'     :  'all_segmentation.joblib',
                    'tforms'  :  'tforms_final.joblib',
                  }
    codebook_hyb_file = os.path.join(resource_dir, 'codebook_hyb.tsv')
    codebook_hyb = pd.read_csv(codebook_hyb_file, sep='\t', index_col=0)
    codebook_geneseq_file = os.path.join(resource_dir, 'codebook_geneseq.tsv')
    codebook_geneseq = pd.read_csv(codebook_geneseq_file, sep='\t', index_col=0)
    
    (cell_id_file, coord_file, gene_rol_file, hyb_rol_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    cell_id = joblib.load(cell_id_file)
    coord = joblib.load(coord_file)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)
    tform_final =joblib.load(tforms_file)
    logging.debug(f'loaded input joblibs.')

    joblib.dump([codebook_hyb.to_numpy() ],os.path.join(outdir, 'hyb_codebook.joblib'))
    joblib.dump( [ codebook_geneseq.to_numpy() ], os.path.join(outdir, 'codebook.joblib'))
    #codebook_hyb = joblib.load(os.path.join(outdir, 'hyb_codebook.joblib'))  

    d={}
    d=data_dict_organizer(d,'initialize',fov=[],gene_rol_id=[],pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
                          cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    T={}
    tilename_list = nsort( list(seg.keys()) )
    for i, tilename in enumerate( tilename_list) :
        logging.debug(f'handling {tilename}') 
        pos_id = np.array([i])
        logging.debug(f'handling tile id: {tilename} i={i} pos_id = {pos_id} ')
        d = data_dict_organizer(d,'append',
                              fov=np.full(len(gene_rol[tilename]['gene_id']),i), 
                              gene_rol_id=np.array(gene_rol[tilename]['gene_id']),
                              pos_10x_allx=coord[tilename]['lroi10x_x'],
                              pos_10x_ally=coord[tilename]['lroi10x_y'],
                              pos_40x_allx=np.array(gene_rol[tilename]['lroi_x']),
                              pos_40x_ally=np.array(gene_rol[tilename]['lroi_y'][i]),
                              # if len(cellid[folders[i]]['cellid']) else np.array([0]),
                              cellidall=np.array(cell_id[tilename]['cellid']) + np.array( i*starting_fov_idx * dummy_cell_num),
                              # check this later,does it require -1 or not
                              sliceidall=np.full(len(gene_rol[tilename]['gene_id']), pos_id + starting_slice_idx), 
                              hyb_rol_id=hyb_rol[tilename]['gene_id'][0],
                              fov_hyb=np.full(len(hyb_rol[tilename]['gene_id'][0]),i),
                              pos_10x_allx_hyb=coord[tilename]['lroi10xhyb_x'],
                              pos_10x_ally_hyb=coord[tilename]['lroi10xhyb_y'],
                              pos_40x_allx_hyb=hyb_rol[tilename]['lroi_x'][0],
                              pos_40x_ally_hyb=hyb_rol[tilename]['lroi_y'][0],
                              cellidall_hyb=np.array(cell_id[tilename]['cellidhyb']) + np.array( i * starting_fov_idx * dummy_cell_num),
                              sliceidall_hyb=np.full(len(hyb_rol[tilename]['gene_id'][0]), pos_id + starting_slice_idx),
                              cell_list_all=np.array(seg[tilename]['cell_num']) + np.array(i * starting_fov_idx * dummy_cell_num),
                              cell_pos_10x_allx=coord[tilename]['cellpos10x_x'],
                              cell_pos_10x_ally=coord[tilename]['cellpos10x_y'],
                              cell_pos_40x_allx=seg[tilename]['cent_x'],
                              cell_pos_40x_ally=seg[tilename]['cent_y'],
                              fov_cell=np.full(len(seg[tilename]['cell_num']),i),
                              sliceidall_cell=np.full(len(seg[tilename]['cell_num']), pos_id + starting_slice_idx))

    #d=data_dict_organizer(d, 'concat', 
    #                      fov=[],gene_rol_id=[],
    #                      pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
    #                      cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],
    #                      pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
    #                      cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],
    #                      cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
    #                      fov_cell=[],sliceidall_cell=[])
    #
    #        d['hyb_rol_id1']=d['hyb_rol_id'] + len(codebook[0])-1
    #        codebook_comb=[codebook[0],hyb_codebook[0]]


    codebook_combined = pd.concat([ codebook_geneseq, codebook_hyb], axis=0 )
    codebook_combined.reset_index(inplace=True, drop=True)

    d['hyb_rol_id1'] = d['hyb_rol_id'] + len(codebook_geneseq)
    d=merge_gene_hyb_dict(d,'gene_rol_id','hyb_rol_id1','combined_gene_hyb_id')
    d=merge_gene_hyb_dict(d,'fov','fov_hyb','combined_gene_hyb_fov')
    d=merge_gene_hyb_dict(d,'pos_10x_allx','pos_10x_allx_hyb','combined_gene_hyb_pos10x_x')
    d=merge_gene_hyb_dict(d,'pos_10x_ally','pos_10x_ally_hyb','combined_gene_hyb_pos10x_y')
    d=merge_gene_hyb_dict(d,'pos_40x_allx','pos_40x_allx_hyb','combined_gene_hyb_pos40x_x')
    d=merge_gene_hyb_dict(d,'pos_40x_ally','pos_40x_ally_hyb','combined_gene_hyb_pos40x_y')
    d=merge_gene_hyb_dict(d,'cellidall','cellidall_hyb','combined_gene_hyb_cellidall')
    d=merge_gene_hyb_dict(d,'sliceidall','sliceidall_hyb','combined_gene_hyb_sliceidall') 
    
    border_size=np.round(fraction_border * tilesize)
    pos_id=d['combined_gene_hyb_id']>0 # uncalled rolonies--how does this happen? what's bardensr's code for uncalled ones
    pos_inside_border_x=(d['combined_gene_hyb_pos40x_x'] > border_size-1) & (d['combined_gene_hyb_pos40x_x'] < tilesize-border_size+1)
    pos_inside_border_y=(d['combined_gene_hyb_pos40x_y'] > border_size-1) & (d['combined_gene_hyb_pos40x_y'] < tilesize-border_size+1)
    filter_id=pos_id & pos_inside_border_x & pos_inside_border_y

    filtered_d={}
    filtered_d=data_dict_organizer(filtered_d,'initialize', combined_gene_hyb_id=[], combined_gene_hyb_fov=[], 
                                   combined_gene_hyb_pos10x_x=[], combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],
                                   combined_gene_hyb_cellidall=[],
                                   combined_gene_hyb_sliceidall=[])

    filtered_d=data_dict_organizer(filtered_d,'append',
                                   combined_gene_hyb_id=d['combined_gene_hyb_id'][filter_id],
                                   combined_gene_hyb_fov=d['combined_gene_hyb_fov'][filter_id],
                                   combined_gene_hyb_pos10x_x=d['combined_gene_hyb_pos10x_x'][filter_id],
                                   combined_gene_hyb_pos10x_y=d['combined_gene_hyb_pos10x_y'][filter_id],
                                   combined_gene_hyb_pos40x_x=d['combined_gene_hyb_pos40x_x'][filter_id],
                                   combined_gene_hyb_pos40x_y=d['combined_gene_hyb_pos40x_y'][filter_id],
                                   combined_gene_hyb_cellidall=d['combined_gene_hyb_cellidall'][filter_id],
                                   combined_gene_hyb_sliceidall=d['combined_gene_hyb_sliceidall'][filter_id])
    
    filtered_d=data_dict_organizer(filtered_d,'concat',
                                   combined_gene_hyb_id=[],combined_gene_hyb_fov=[],
                                   combined_gene_hyb_pos10x_x=[],combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],
                                   combined_gene_hyb_cellidall=[],
                                   combined_gene_hyb_sliceidall=[])

    cells=d['cell_list_all'].copy() # check if copy messed something
    genes=np.unique(d['combined_gene_hyb_id'])
    rol_id=d['combined_gene_hyb_id'].copy()
    rol_cell=d['combined_gene_hyb_cellidall'].copy()
    v=pd.crosstab(rol_cell, rol_id, rownames=['cell_index'], colnames=['genes'], dropna=False)
    v=v.reindex(index=cells, columns=genes, fill_value=0)
    exp_m=coo_matrix(v.to_numpy())
    processed_data={'all_data':d,
                    'filtered_data':filtered_d,
                    'expmat': exp_m,
                    'cells': cells,
                    'gene_id': genes,
                    'codebook_combined': codebook_combined}
    logging.info(f'Writing output to {outfile}')
    joblib.dump(processed_data, outfile)

    rolonies={'id':filtered_d['combined_gene_hyb_id'],
              'pos10_x':filtered_d['combined_gene_hyb_pos10x_x'],
              'pos10_y':filtered_d['combined_gene_hyb_pos10x_y'],
              'pos40_x':filtered_d['combined_gene_hyb_pos40x_x'],
              'pos40_y':filtered_d['combined_gene_hyb_pos40x_y'],
              'slice':filtered_d['combined_gene_hyb_sliceidall'],
              'genes':codebook_combined,
              'fov':filtered_d['combined_gene_hyb_fov'],
              'fov_names':tilename_list}

    neurons={'expmat':exp_m,
             'id':d['cell_list_all'],
             'pos10x_x':d['cell_pos_10x_allx'],
             'pos10x_y':d['cell_pos_10x_ally'],
             'pos40x_x':d['cell_pos_40x_allx'],
             'pos40x_y':d['cell_pos_40x_ally'],
             'slice':d['sliceidall_cell'],
             'genes':codebook_combined,
             'fov':d['fov_cell'],
             'fov_names':tilename_list}

    alldata = {"rolonies":rolonies, "neurons":neurons} 
    joblib.dump( alldata, os.path.join(outdir, 'alldata.joblib'))
    logging.info('ALL DATA IS ORGANIZED')

    logging.info(f'Writing out data subsets...')    
    # Output subsets...
    # create individual output data files. DFs. Pandas matrix.
    #
    of = os.path.join( outdir, f'{project_id}.cellsbygenes.tsv')
    v.to_csv(of, sep='\t') 
    logging.info(f'Wrote cells X genes matrix to {of}')

    of = os.path.join(outdir, f'{project_id}.codebook_combined.tsv') 
    codebook_combined.to_csv(of, sep='\t')
    logging.info(f'Wrote combined codebook to {of}')

    logging.info(f'Done.')



def data_dict_organizer(d, operation, **kwargs): 
    """
    Helper function: Organizes dictionaries
    """
    if operation=='initialize':
        d.update(kwargs)
    elif operation=='append':
        for key in kwargs:
            d[key].append(kwargs[key])
    elif operation=='concat':
        for key in kwargs:
            d[key]=np.concatenate(d[key])
    return d


def merge_gene_hyb_dict(d, key1, key2, key3):
    """
    Helper function: Combines gene and hyb data into one dictionary
    """
    ar=[d[key1], d[key2]]
    d[key3]= np.concatenate(ar)
    return d


#########################
# NOTEBOOK CODE
#########################
def organize_processed_data_notebook(pth,config_pth,is_optseq=0,hyb_codebook_name='codebookhyb.mat',starting_slice_idx=1,starting_fov_idx=1,dummy_cell_num=10000,tilesize=3200,
                           fraction_border=0.07):
    """
    Postprocessing function:
    Organizes all the data into rolonies and neurons--prepares a big dictionary
    """
    today=datetime.date.today().strftime('%d%m%Y')
    gene_rol=load(os.path.join(pth,'processed','basecalls.joblib'))
    seg=load(os.path.join(pth,'processed','all_segmentation.joblib'))
    hyb_rol=load(os.path.join(pth,'processed','genehyb.joblib'))
    Tf=load(os.path.join(pth,'processed','tforms_final.joblib'))
    coord=load(os.path.join(pth,'processed','lroi10x.joblib'))
    codebook=load(os.path.join(pth,'processed','codebook.joblib'))
    cellid=load(os.path.join(pth,'processed','cell_id.joblib'))
    hyb_codebook=scipy.io.loadmat(os.path.join(config_pth,hyb_codebook_name))['codebookhyb']
    dump([hyb_codebook],os.path.join(pth,'processed','hyb_codebook.joblib'))
    hyb_codebook=load(os.path.join(pth,'processed','hyb_codebook.joblib'))
    [folders,pos,_,_]=get_folders(pth)
    npos=nsort(np.unique(pos))
    d={}
    d=data_dict_organizer(d,'initialize',fov=[],gene_rol_id=[],pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
                          cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    for i,folder in enumerate(folders):
        
        pos_id=np.array([j for j,name in enumerate(npos) if name==pos[i]]) # search for slice/position number for this tile

        d=data_dict_organizer(d,'append',
                              fov=np.full(len(gene_rol['gene_id'][i]),i),
                              gene_rol_id=np.array(gene_rol['gene_id'][i]),
                              pos_10x_allx=coord[folders[i]]['lroi10x_x'],
                              pos_10x_ally=coord[folders[i]]['lroi10x_y'],
                              pos_40x_allx=np.array(gene_rol['lroi_x'][i]),
                              pos_40x_ally=np.array(gene_rol['lroi_y'][i]),
                              cellidall=np.array(cellid[folders[i]]['cellid'])+np.array(i*starting_fov_idx*dummy_cell_num),# if len(cellid[folders[i]]['cellid']) else np.array([0]),
                              sliceidall=np.full(len(gene_rol['gene_id'][i]),pos_id+starting_slice_idx), # check this later,does it require -1 or not
                              hyb_rol_id=hyb_rol['gene_id'][i][0],
                              fov_hyb=np.full(len(hyb_rol['gene_id'][i][0]),i),
                              pos_10x_allx_hyb=coord[folders[i]]['lroi10xhyb_x'],
                              pos_10x_ally_hyb=coord[folders[i]]['lroi10xhyb_y'],
                              pos_40x_allx_hyb=hyb_rol['lroi_x'][i][0],
                              pos_40x_ally_hyb=hyb_rol['lroi_y'][i][0],
                              cellidall_hyb=np.array(cellid[folders[i]]['cellidhyb'])+np.array(i*starting_fov_idx*dummy_cell_num),
                              sliceidall_hyb=np.full(len(hyb_rol['gene_id'][i][0]),pos_id+starting_slice_idx),
                              cell_list_all=np.array(seg[folders[i]]['cell_num'])+np.array(i*starting_fov_idx*dummy_cell_num),
                              cell_pos_10x_allx=coord[folders[i]]['cellpos10x_x'],
                              cell_pos_10x_ally=coord[folders[i]]['cellpos10x_y'],
                              cell_pos_40x_allx=seg[folders[i]]['cent_x'],
                              cell_pos_40x_ally=seg[folders[i]]['cent_y'],
                              fov_cell=np.full(len(seg[folders[i]]['cell_num']),i),
                              sliceidall_cell=np.full(len(seg[folders[i]]['cell_num']),pos_id+starting_slice_idx))

    d=data_dict_organizer(d,'concat',fov=[],gene_rol_id=[],pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
                          cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])
    if is_optseq:
        print('Has optseq')
        codebook_optseq=load(os.path.join(pth,'processed','codebook_optseq.joblib'))
        d['hyb_rol_id1']=d['hyb_rol_id']+len(codebook[0])+len(codebook_optseq[0])# -2 not needed this subtraction if passing index from above
        codebook_comb=[codebook[0],codebook_optseq[0],hyb_codebook[0]]
    else:
        d['hyb_rol_id1']=d['hyb_rol_id']+len(codebook[0]) #-1 not needed this subtraction if passing index from above
        codebook_comb=[codebook[0],hyb_codebook[0]]
        
    codebook_comb=np.concatenate(codebook_comb)
    d=merge_gene_hyb_dict(d,'gene_rol_id','hyb_rol_id1','combined_gene_hyb_id')
    d=merge_gene_hyb_dict(d,'fov','fov_hyb','combined_gene_hyb_fov')
    d=merge_gene_hyb_dict(d,'pos_10x_allx','pos_10x_allx_hyb','combined_gene_hyb_pos10x_x')
    d=merge_gene_hyb_dict(d,'pos_10x_ally','pos_10x_ally_hyb','combined_gene_hyb_pos10x_y')
    d=merge_gene_hyb_dict(d,'pos_40x_allx','pos_40x_allx_hyb','combined_gene_hyb_pos40x_x')
    d=merge_gene_hyb_dict(d,'pos_40x_ally','pos_40x_ally_hyb','combined_gene_hyb_pos40x_y')
    d=merge_gene_hyb_dict(d,'cellidall','cellidall_hyb','combined_gene_hyb_cellidall')
    d=merge_gene_hyb_dict(d,'sliceidall','sliceidall_hyb','combined_gene_hyb_sliceidall')

    
    border_size=np.round(fraction_border*tilesize)

    pos_id=d['combined_gene_hyb_id']>0 # uncalled rolonies--how does this happen? what's bardensr's code for uncalled ones
    pos_inside_border_x=(d['combined_gene_hyb_pos40x_x']>border_size-1) & (d['combined_gene_hyb_pos40x_x']<tilesize-border_size+1)
    pos_inside_border_y=(d['combined_gene_hyb_pos40x_y']>border_size-1) & (d['combined_gene_hyb_pos40x_y']<tilesize-border_size+1)
    filter_id=pos_id & pos_inside_border_x & pos_inside_border_y

    filtered_d={}
    filtered_d=data_dict_organizer(filtered_d,'initialize',combined_gene_hyb_id=[],combined_gene_hyb_fov=[],combined_gene_hyb_pos10x_x=[],combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],combined_gene_hyb_cellidall=[],combined_gene_hyb_sliceidall=[])

    filtered_d=data_dict_organizer(filtered_d,'append',
                                   combined_gene_hyb_id=d['combined_gene_hyb_id'][filter_id],
                                   combined_gene_hyb_fov=d['combined_gene_hyb_fov'][filter_id],
                                   combined_gene_hyb_pos10x_x=d['combined_gene_hyb_pos10x_x'][filter_id],
                                   combined_gene_hyb_pos10x_y=d['combined_gene_hyb_pos10x_y'][filter_id],
                                   combined_gene_hyb_pos40x_x=d['combined_gene_hyb_pos40x_x'][filter_id],
                                   combined_gene_hyb_pos40x_y=d['combined_gene_hyb_pos40x_y'][filter_id],
                                   combined_gene_hyb_cellidall=d['combined_gene_hyb_cellidall'][filter_id],
                                   combined_gene_hyb_sliceidall=d['combined_gene_hyb_sliceidall'][filter_id])
    filtered_d=data_dict_organizer(filtered_d,'concat',combined_gene_hyb_id=[],combined_gene_hyb_fov=[],combined_gene_hyb_pos10x_x=[],combined_gene_hyb_pos10x_y=[],
                                   combined_gene_hyb_pos40x_x=[],combined_gene_hyb_pos40x_y=[],combined_gene_hyb_cellidall=[],combined_gene_hyb_sliceidall=[])

    
    cells=d['cell_list_all'].copy() # check if copy messed something
    genes=np.unique(d['combined_gene_hyb_id'])
    rol_id=d['combined_gene_hyb_id'].copy()
    rol_cell=d['combined_gene_hyb_cellidall'].copy()
    v=pd.crosstab(rol_cell, rol_id, rownames=['cell_index'], colnames=['genes'],dropna=False)
    v=v.reindex(index=cells, columns=genes, fill_value=0)
    exp_m=coo_matrix(v.to_numpy())
    processed_data={'all_data':d,
                    'filtered_data':filtered_d,
                    'expmat':exp_m,
                    'cells':cells,
                    'gene_id':genes,
                    'codebook_combined':codebook_comb}
    dump(processed_data,os.path.join(pth,'processed','processeddata.joblib'))


    rolonies={'id':filtered_d['combined_gene_hyb_id'],
              'pos10_x':filtered_d['combined_gene_hyb_pos10x_x'],
              'pos10_y':filtered_d['combined_gene_hyb_pos10x_y'],
              'pos40_x':filtered_d['combined_gene_hyb_pos40x_x'],
              'pos40_y':filtered_d['combined_gene_hyb_pos40x_y'],
              'slice':filtered_d['combined_gene_hyb_sliceidall'],
              'genes':codebook_comb,
              'fov':filtered_d['combined_gene_hyb_fov'],
              'fov_names':folders}
    neurons={'expmat':exp_m,
             'id':d['cell_list_all'],
             'pos10x_x':d['cell_pos_10x_allx'],
             'pos10x_y':d['cell_pos_10x_ally'],
             'pos40x_x':d['cell_pos_40x_allx'],
             'pos40x_y':d['cell_pos_40x_ally'],
             'slice':d['sliceidall_cell'],
             'genes':codebook_comb,
             'fov':d['fov_cell'],
             'fov_names':folders}
    dated_filename='alldata'+str(today)+'.joblib'
    dump({"rolonies":rolonies,"neurons":neurons},os.path.join(pth,'processed',dated_filename))
    print('ALL DATA IS ORGANIZED')
    return dated_filename



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

    aggregate_data_py( infiles=args.infiles, 
                            outfiles=args.outfiles,
                            stage=args.stage,  
                            cp=cp )
    logging.info(f'done processing output to {args.outfiles[0]}')
