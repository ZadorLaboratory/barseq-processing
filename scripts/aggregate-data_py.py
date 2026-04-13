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
    d=data_dict_organizer(d,'initialize',fov=[], gene_rol_id=[],
                          pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],cellidall=[],sliceidall=[],
                          hyb_rol_id=[],fov_hyb=[],
                          pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],
                          cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    fnames = list(tform_final.keys() )
    tilenames = nsort(  [ os.path.splitext(fn)[0] for fn in fnames ] )
    T={}
    for i, tilename in enumerate(tilenames):
        pos_id = np.array([i])
        logging.debug(f'handling tile id: {tilename} i={i} ')
        d = data_dict_organizer(d,'append',
                              fov=np.full(len(gene_rol['gene_id'][i]),i), 
                              gene_rol_id=np.array(gene_rol['gene_id'][i]),
                              pos_10x_allx=coord[tilenames[i]]['lroi10x_x'],
                              pos_10x_ally=coord[tilenames[i]]['lroi10x_y'],
                              pos_40x_allx=np.array(gene_rol['lroi_x'][i]),
                              pos_40x_ally=np.array(gene_rol['lroi_y'][i]),
                              cellidall=np.array(cell_id[tilenames[i]]['cellid']) + np.array(i*starting_fov_idx * dummy_cell_num),# if len(cellid[folders[i]]['cellid']) else np.array([0]),
                              sliceidall=np.full(len(gene_rol['gene_id'][i]), pos_id+starting_slice_idx), # check this later,does it require -1 or not
                              hyb_rol_id=hyb_rol['gene_id'][i][0],
                              fov_hyb=np.full(len(hyb_rol['gene_id'][i][0]),i),
                              pos_10x_allx_hyb=coord[tilenames[i]]['lroi10xhyb_x'],
                              pos_10x_ally_hyb=coord[tilenames[i]]['lroi10xhyb_y'],
                              pos_40x_allx_hyb=hyb_rol['lroi_x'][i][0],
                              pos_40x_ally_hyb=hyb_rol['lroi_y'][i][0],
                              cellidall_hyb=np.array(cell_id[tilenames[i]]['cellidhyb']) + np.array(i * starting_fov_idx * dummy_cell_num),
                              sliceidall_hyb=np.full(len(hyb_rol['gene_id'][i][0]), pos_id + starting_slice_idx),
                              cell_list_all=np.array(seg[tilenames[i]]['cell_num']) + np.array(i * starting_fov_idx * dummy_cell_num),
                              cell_pos_10x_allx=coord[tilenames[i]]['cellpos10x_x'],
                              cell_pos_10x_ally=coord[tilenames[i]]['cellpos10x_y'],
                              cell_pos_40x_allx=seg[tilenames[i]]['cent_x'],
                              cell_pos_40x_ally=seg[tilenames[i]]['cent_y'],
                              fov_cell=np.full(len(seg[tilenames[i]]['cell_num']),i),
                              sliceidall_cell=np.full(len(seg[tilenames[i]]['cell_num']), pos_id + starting_slice_idx))

    d=data_dict_organizer(d,'concat',fov=[],gene_rol_id=[],
                          pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
                          cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],
                          pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],
                          cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    codebook_combined = pd.concat([ codebook_geneseq, codebook_hyb], axis=0 )
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


def merge_gene_hyb_dict(d,key1,key2,key3):#
    """
    Helper function: Combines gene and hyb data into one dictionary
    """
    ar=[d[key1],d[key2]]
    d[key3]=np.concatenate(ar)
    return d


#########################
# NOTEBOOK CODE
#########################

def organize_processed_data(pth,config_pth,
                            is_optseq=0,
                            hyb_codebook_name='codebookhyb.mat',
                            starting_slice_idx=1,
                            starting_fov_idx=1,
                            dummy_cell_num=10000,
                            tilesize=3200,
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
    # pos = ['Pos1', 'Pos1',  'Pos1', ... ] 
    # folders = [['MAX_Pos1_000_000', 'MAX_Pos1_000_001', ...]
    npos=nsort(np.unique(pos))
    # npos= [np.str_('Pos1')]

    d={}
    d=data_dict_organizer(d,'initialize',fov=[],gene_rol_id=[],pos_10x_allx=[],pos_10x_ally=[],pos_40x_allx=[],pos_40x_ally=[],
                          cellidall=[],sliceidall=[],hyb_rol_id=[],fov_hyb=[],pos_10x_allx_hyb=[],pos_10x_ally_hyb=[],pos_40x_allx_hyb=[],pos_40x_ally_hyb=[],
                          cellidall_hyb=[],sliceidall_hyb=[],cell_list_all=[],cell_pos_10x_allx=[],cell_pos_10x_ally=[],cell_pos_40x_allx=[],cell_pos_40x_ally=[],
                          fov_cell=[],sliceidall_cell=[])

    for i,folder in enumerate(folders):
        # i = 0
        # folder = 'MAX_Pos1_000_000'
        pos_id=np.array([j for j,name in enumerate(npos) if name==pos[i]]) # search for slice/position number for this tile
        # array([0])

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
    
    
    # codebook. type = list.   
    # codebook[0] -> array...  np.ndarray
    # codebook[0][0] -> array([array(['Calb1'], dtype='<U5'), array(['AGTTCGG'], dtype='<U7')], dtype=object)
    # codebook[0][0][0] -> array(['Calb1'], dtype='<U5') 
    # codebook[0][0][0][0] ->. np.str_('Calb1') 
    #
    # hyb_codebook type=list
    # 
    # hyb_codebook[0] -> array
    # hyb_codebook[0][0]  array([array(['Slc17a7'], dtype='<U7'), array([[1]], dtype=uint8)], dtype=object)
    # hyb_codebook[0][0][0]   array(['Slc17a7'], dtype='<U7')
    # hyb_codebook[0][0][0][0] -> np.str_('Slc17a7')

    if is_optseq:
        print('Has optseq')
        codebook_optseq=load(os.path.join(pth,'processed','codebook_optseq.joblib'))
        d['hyb_rol_id1']=d['hyb_rol_id']+len(codebook[0])-1+len(codebook_optseq[0])-1
        codebook_comb=[codebook[0],codebook_optseq[0],hyb_codebook[0]]
    else:
        d['hyb_rol_id1']=d['hyb_rol_id']+len(codebook[0])-1
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
    
    border_size=np.round(fraction_border * tilesize)

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
    v=pd.crosstab(rol_cell, rol_id, rownames=['cell_index'], colnames=['genes'], dropna=False)
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


def data_dict_organizer_notebook(d,operation,**kwargs):#
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

def merge_gene_hyb_dict_notebook(d, key1, key2, key3):
    """
    Helper function: Combines gene and hyb data into one dictionary
    """
    ar=[d[key1],d[key2]]
    d[key3]=np.concatenate(ar)
    return d


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
