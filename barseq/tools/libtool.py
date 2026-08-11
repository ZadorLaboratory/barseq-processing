import joblib
import logging
import math
import os
import re
import pprint
import sys
import datetime as dt
from configparser import ConfigParser

from natsort import natsorted as nsort

import numpy as np
import cv2

from barseq.core import *
from barseq.tools import *
from barseq.utils import *
from barseq.imageutils import *

def aggregate_cellids_py(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #. There may be more inputs that required, so only select relevant ones...
    # E.g.
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/all_segmentation.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/genehyb.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_original.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_rescaled0p5.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/geneseq/basecalls.joblib
    # 

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-cellids'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    #   'basecalls.joblib'.  'all_segmentation.joblib'   'genehyb.joblib'
    #
    # return order will be alphabetical
    #
    input_map = { 'gene_rol' : 'basecalls.joblib',
                  'hyb_rol' :  'genehyb.joblib',
                  'seg' : 'all_segmentation.joblib'
                  }

    (gene_rol_file, hyb_rol_file, seg_file) = select_input_files(infiles, input_map)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)

    T={}
    tilename_list = nsort( list(seg.keys()) )
    for i, tilename in enumerate( tilename_list) :
        logging.debug(f'handling {tilename}') 
        t={}
        mask=seg[tilename]['dilated_labels']
        coord_xg=gene_rol[tilename]['lroi_x']
        coord_yg=gene_rol[tilename]['lroi_y']
        # coord_xh=hyb_rol[tilename]['lroi_x'][0][0]
        # coord_yh=hyb_rol[tilename]['lroi_y'][0][0]
        coord_xh=hyb_rol[tilename]['lroi_x']
        coord_yh=hyb_rol[tilename]['lroi_y']
        t['cellid']= assign_rolony_to_cell(mask, coord_xg, coord_yg)
        t['cellidhyb']= assign_rolony_to_cell(mask, coord_xh, coord_yh)
        T[tilename]=t
    joblib.dump(T,os.path.join(outfile))
    logging.info(f'Done.')

def assign_rolony_to_cell(mask, coord_x, coord_y):
    """
    Global transformation function:
    1. Calls get_cellid function if there are rolonies detected in this tile or else assigns empty cell id to this tile
    """
    #logging.debug(f'handling coord_x = {coord_x}, coord_y={coord_y}')
    if len(coord_x):
        cell_id=get_cellid(mask, coord_x, coord_y)
    else:
        cell_id=[] # earlier this was [] and was causing error later
    return cell_id

def get_cellid(mask, coord_x, coord_y):
    """
    Global transformation function:
    1. For any detected rolony-assigns it to a cell
    2. Returns the cell ids for all rolonies in this tile
    """
    #logging.debug(f'handling coord_x = {coord_x}, coord_y={coord_y}')
    coord_xl=[int(np.round(x)) for x in coord_x]
    coord_yl=[int(np.round(x)) for x in coord_y]
    cell_id=mask[coord_xl,coord_yl]
    return cell_id


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
                              
                              hyb_rol_id = hyb_rol[tilename]['gene_id'],

                              # possible mismatch? nested list in our hyb_rol vs. notebook?
                              fov_hyb = np.full(len( hyb_rol[tilename]['gene_id'] ),i),
                              
                              pos_10x_allx_hyb = coord[tilename]['lroi10xhyb_x'],
                              pos_10x_ally_hyb = coord[tilename]['lroi10xhyb_y'],
                              pos_40x_allx_hyb = hyb_rol[tilename]['lroi_x'],
                              pos_40x_ally_hyb = hyb_rol[tilename]['lroi_y'],

                              cellidall_hyb = np.array(cell_id[tilename]['cellidhyb']) + np.array(i * starting_fov_idx * dummy_cell_num ),
                              sliceidall_hyb = np.full(len(hyb_rol[tilename]['gene_id']), pos_id + starting_slice_idx ),
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

def aggregate_transform_np(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
    #.    inputs: 'basecalls.joblib'.  
    #             'all_segmentation.joblib'   
    #             'genehyb.joblib'
    #             'tforms_final.joblib'
    #
    #. There may be more inputs that required, so only select relevant ones...
    # E.g.
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/all_segmentation.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/genehyb.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_original.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/hyb/tforms_rescaled0p5.joblib 
    #   /Users/hover/project/barseq/run_barseq/BC726126.7.out/merge/geneseq/basecalls.joblib
    # 
    # lroi10x.joblib is main flag output. 

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'aggregate-transform'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We know arity is single, so we can grab the outfile
    # primary outfile is lroi10x.joblib
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    #   'basecalls.joblib'.  'all_segmentation.joblib'   'genehyb.joblib'
    #
    # return order will be alphabetical
    #
    input_map = { 'gene_rol' : 'basecalls.joblib',
                  'hyb_rol' :  'genehyb.joblib',
                  'seg' : 'all_segmentation.joblib',
                  'tforms' : 'tforms_final.joblib',
                  }

    (gene_rol_file, hyb_rol_file, seg_file, tforms_file) = select_input_files(infiles, input_map)
    gene_rol=joblib.load(gene_rol_file)
    seg=joblib.load(seg_file)
    hyb_rol=joblib.load(hyb_rol_file)
    tform_final =joblib.load(tforms_file)

    tilename_list = nsort( list(seg.keys() ))
    T={}
    for i, tilename in enumerate(tilename_list):
        logging.debug(f'handling {tilename}') 
        t={}
        tform=tform_final[tilename]        
        [x,y]=apply_transform(tform, gene_rol[tilename]['lroi_y'], gene_rol[tilename]['lroi_x'])
        t['lroi10x_x']=x
        t['lroi10x_y']=y
        [x,y]=apply_transform(tform, hyb_rol[tilename]['lroi_y'],hyb_rol[tilename]['lroi_x']) 
        t['lroi10xhyb_x']=x
        t['lroi10xhyb_y']=y
        [x,y]=apply_transform(tform, seg[tilename]['cent_y'],seg[tilename]['cent_x']) 
        t['cellpos10x_x']=x
        t['cellpos10x_y']=y
        T[tilename]=t
    
    logging.info(f'Writing output to {outfile}')
    joblib.dump(T, outfile)
    logging.info(f'Done.')
    

def apply_transform(tform, coord_x, coord_y):
    """
    Global transformation function:
    1. Transforms the local coordinates of rolonies and cells to global downsized coordinates per tile
    """
    if len(coord_x):
        if not (isinstance(coord_x,list) or isinstance(coord_x,np.ndarray)):
            coord_x=coord_x.to_list()
            coord_y=coord_y.to_list()
        q=np.zeros([len(coord_x),2])
        q[:,0]=np.reshape(coord_x,(1,-1))
        q[:,1]=np.reshape(coord_y,(1,-1))
        v=tform(q)
        x=v[:,0]
        y=v[:,1]
    else:
        x=[]
        y=[]
    return x,y


def background_cv2( infiles, outfiles, stage=None, cp=None):
    '''
    Subtracts background from select_channels of input images. 
    Remainder channels are added back unchanged. 

    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'background-geneseq'
            
    image_type = cp.get(stage,'image_type')
    radius = int(cp.get('cv2','radius'))
    output_dtype = cp.get( stage,'output_dtype')
    channel_names =  get_config_list(cp, image_type, 'channels')
    select_channels = get_config_list(cp, stage, 'channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    num_c = len(select_channels)

    logging.debug(f'output_dtype={output_dtype} radius = {radius} num_channels={num_c} select_channels = {select_channels}')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')        
        
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        I = read_image( infile)        
        I=I.copy()
        I_filtered=np.zeros_like(I)
        I_rem=I[num_c:,:,:]
        I=I[0:num_c,:,:]
        k=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (radius,radius))
        for i in range(len(I)):
            bck=cv2.morphologyEx(I[i,:,:], cv2.MORPH_OPEN, kernel = k)
            I_filtered[i,:,:] = I[i,:,:] - np.expand_dims(bck,0)
        
        I_filtered[num_c:,:,:]=I_rem    
        I_filtered=uint16m(I_filtered)

        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image( outfile, I_filtered, photometric = 'minisblack' )
        logging.debug(f'done writing {outfile}')


def basecall_geneseq_bardensr( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    load codebook, 
    run bardensr, 
    output evidence tensor dataframe to <outdir>/<mode>/<prefix>.brdnsr.tsv   
    arity is single. 
    '''

    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-geneseq'

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
    
    intensity_thresh = None
    (subdir, base, current_label, current_ext) = parse_rpath(outfile)
    param_file = os.path.join(subdir, f'bardensrparams.json')
    if os.path.exists(param_file):
        with open(param_file, 'r' ) as f:
            data = json.load(f)
            intensity_thresh = float( data['intensity_thresh_refined'] )
            noisefloor_final = float( data['noisefloor_final'])
            trim = int( data['trim'])
            cropf = float( data['cropf'])
            median_max_list = data['median_max']
            median_max_list = [ np.float64(x) for x in median_max_list ]
            median_max = np.array(median_max_list)
            logging.debug(f'type(median_max) = {type(median_max)} type(median_max[0])= {type(median_max[0])} ')
            logging.info(f'Successfully loaded intensity_thresh = {intensity_thresh} median_max = {median_max}')
    else:
        logging.warning(f'param_file={param_file} does not exist. Exitting.')
        sys.exit(1)
    logging.debug(f'noisefloor_final={noisefloor_final} trim={trim} cropf={cropf}')

    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = get_config_list(cp, stage, 'codebook_bases')
    cbfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cbfile}')
    codebook_df = load_codebook_file(cbfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook_df} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles)
    logging.info(f'Detected tilesets of {n_cycles} cycles.')
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook_df, 
                                                                        codebook_bases, 
                                                                        n_cycles=n_cycles)
    logging.info(f'R={R} C={C} J={J} codeflat.shape={codeflat.shape} len(genes)={len(genes)} pos_unused_codes={pos_unused_codes}')

    img_stack_trimmed = bd_read_images(infiles, R, C, trim=trim )
    logging.debug(f'img_stack_trimmed.shape = {img_stack_trimmed.shape} img_stack_trimmed.sum() = {img_stack_trimmed.sum()}')
    img_norm = img_stack_trimmed / median_max[ :, None, None, None]
    
    logging.debug(f'img_norm.shape={img_norm.shape} img_norm.sum()={img_norm.sum()} codeflat={codeflat} noisefloor_final={noisefloor_final}')
    et = bardensr.spot_calling.estimate_density_singleshot( img_norm, 
                                                            codeflat, 
                                                            noisefloor_final)
    logging.debug(f'estimated_density sum = {et.sum()} intensity_thresh={intensity_thresh}')
    
    spots = bardensr.spot_calling.find_peaks( et, intensity_thresh, use_tqdm_notebook=False)
    spots.loc[:,'m1'] = spots.loc[:,'m1'] + trim
    spots.loc[:,'m2'] = spots.loc[:,'m2'] + trim

    (odirpath, obase, olabel, oext) = split_path(outfile)
    logging.info(f'In {obase} found {len(spots)} spots.')            
    spots.to_csv(outfile, index=False)   
    logging.debug(f'wrote spots to outfile={outfile}')

    of = os.path.join(odirpath, f'{obase}.codeflat.joblib')
    joblib.dump(codeflat, of)


gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def basecall_hyb_ski( infiles, outfiles, stage=None, cp=None):
    '''
    Handle single tile. For current hyb, should be single infile. 
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

    # We know input is single, so we can grab the infile 
    infile = infiles[0]

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    position_regex = cp.get(stage, 'position_regex')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, file_label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')

    # Stage-specific tool params
    all_genes_ch = cp.getint(stage, 'all_genes_ch')    
    thresh_str = cp.get( stage,'thresh')    
    relaxed = cp.getboolean( stage, 'relaxed')
    no_deconv = cp.getboolean( stage, 'no_deconv')
    filter_overlap = cp.getint( stage, 'filter_overlap')
    num_c = cp.getint( stage, 'num_c')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    
    # Parameters that need evaluation
    prominence_str = cp.get( stage, 'prominence')
    logging.debug(f'params. thresh_str={thresh_str} prominence_str={prominence_str} evaluating... ')
    prominence = eval( prominence_str ) 
    thresh = eval( thresh_str )
    logging.debug(f'all_genes_ch={all_genes_ch} thresh={thresh} prominence={prominence}')

    (dirpath, base ) = os.path.split(infile)
    m = re.search(position_regex, base)
    if m is not None:
        pos_id = m.group(1)
    else:
        logging.error(f'Unable to extract position index from file base: {base}')
        sys.exit(2)
    logging.info(f'handling pos_id={pos_id}')

    #hyb_raw=tfl.imread(os.path.join(hybseq[0]), key=range(0,num_c,1))
    readchannels = list(range(0,num_c))
    hyb_2=read_image(infile, channels=readchannels)
    
    # Handle case of 1-channel image. 
    if len(hyb_2.shape) == 2:
        hyb_2 = np.expand_dims(hyb_2, axis=0)

    # zero-ing all-genes channel 3 (index 2)
    hyb_2[all_genes_ch,:,:] = 0  
    logging.debug(f'basecalling {infile} hyb_2.shape = {hyb_2.shape} all_genes_ch = {all_genes_ch} thresh={thresh} prominence={prominence}')
    
    lroi_x=[]
    lroi_y=[]
    id_t=[]
    sig_t=[]
    mask = np.zeros_like(hyb_2)
    for n in range(num_c):
        if n == all_genes_ch:
            mask[n, :, :] = 0
            continue
        else:
            a = hyb_2[n,:,:].copy()
            a_mask = a > thresh[n]
            a_masked = a * a_mask
            a_max = extrema.h_maxima(a_masked, prominence[n])
            label_peaks = label(a_max)
            m = regionprops(label_peaks, a_masked)
            # m_t = regionprops_table(label_peaks, a_masked)
            mask[n,:,:] = uint16m(binary_dilation(a_max))
            [lroi_x, lroi_y, id_t, sig_t] = quantify_peaks( lroi_x, lroi_y, id_t, sig_t, m, hyb_2)

    # Write mask_hyb image. 
    (dirpath, base ) = os.path.split(infile)
    (base,ext) = os.path.splitext(base)
    ext = ext[1:]
    of = os.path.join( outdir, f'{base}.mask_hyb.{ext}' )
    logging.debug(f'Writing mask to {of}')
    write_image(of, mask)

    gene_map = np.zeros(hyb_2.shape[1:], dtype=np.uint8)
    for ch_idx in range(len(lroi_x)):
        for k in range(len(lroi_x[ch_idx])):
            r = int(round(lroi_x[ch_idx][k]))
            c = int(round(lroi_y[ch_idx][k]))
            gene_map[r, c] = id_t[ch_idx][k] + 1  # +1 so background stays 0

    # Write out gene map. 
    of = os.path.join( outdir, f'{base}.basecall_map_hyb.{ext}' )
    logging.debug(f'Writing basecall map to {of}')
    write_image(of, gene_map)
    
    # Flatten and convert to numpy arrays...
    lroi_x = flat_np_list(lroi_x)
    lroi_y = flat_np_list(lroi_y)
    id_t = flat_np_list(id_t)
    sig_t = flat_np_list(sig_t)
    
    data_dict = {"lroi_x":  lroi_x, 
                 "lroi_y": lroi_y, 
                 "gene_id": id_t, 
                 "signal": sig_t}
    logging.debug(f'got result: lroi_x={len(lroi_x)}, lroi_y={len(lroi_y)}, id_t={len(id_t)}, sig_t={len(sig_t)} ')

    # Write out joblib
    logging.info(f'Writing results to {outfile}')
    joblib.dump(data_dict, outfile)


def flat_np_list(xss):
    flat_list = [ x for xs in xss for x in xs ]
    return np.array(flat_list)


def quantify_peaks(lroi_x, lroi_y, id_t, sig_t, m, hyb_2):
    """
    Basecalling function:
    1. Based on the regionprops results per tile, this function creates hyb basecalling output and 
       decodes the gene
    2. Returns the basecall output to the calling function
    """ 
    sig1=[]
    lroi1_x=[]
    lroi1_y=[]
    id1=[]
    ch_to_gene={0:0, 1:1, 3:2}  # skip all_genes_ch=2
    for i, peaks in enumerate(m):
        lroi1_x.append(peaks.centroid[0])
        lroi1_y.append(peaks.centroid[1])
        sig1.append(peaks.intensity_max)
        peaks_s = print_regionprop(peaks)
        # logging.debug(f'hyb_2 = {hyb_2}')
        # logging.debug(f'hyb_2.shape = {hyb_2.shape}')
        # logging.debug(f'\n [{i}] {peaks_s}')
        id1.append( ch_to_gene[ np.argmax( hyb_2[:, peaks.coords[0][0], peaks.coords[0][1]]) ])
    lroi_x.append(lroi1_x)
    lroi_y.append(lroi1_y)
    id_t.append(id1)
    sig_t.append(sig1)
    return(lroi_x, lroi_y, id_t, sig_t)

def print_regionprop(prop):
    s = ''
    s += f'\n  coords[0][0] = {prop.coords[0][0]}'
    s += f'\n  coords[0][1] = {prop.coords[0][1]}'
    s += f'\n  orientation={prop.orientation}'
    return s