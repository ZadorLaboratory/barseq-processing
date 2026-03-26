#!/usr/bin/env python
#
# Apply transforms
# 
# 
import argparse
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

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def aggregate_transform_np(infiles, outfiles, stage=None, cp=None):
    #     cycleset map 
    #         arity=single
    #         so inputs will be (flat list of all files from first cycle)
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
    # filt_neurons.joblib is main flag output. 

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

    fnames = list(tform_final.keys() )
    tilenames = nsort(  [ os.path.splitext(fn)[0] for fn in fnames ] )
    T={}
    for i, tilename in enumerate(tilenames):
        pass
    
    logging.info(f'Writing output to {outfile}')
    joblib.dump(T, outfile)
    logging.info(f'Done.')





# NOTEBOOK CODE

from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
def filter_overlapping_cells(pth,fname='alldata06092025.joblib',rescale_factor=0.5,px=0.33,box_half_width_um=5,search_radius_um=200):
    """
    Postprocessing function:
    1. Removes overlapping neurons per tile
    2. Searching within a circle of search radius 200 um around per cell, looking for only cells from different fov and removing cells within a square of overlap width 10um
    3. Saves the final output filt_neurons
    """
    data=load(os.path.join(pth,'processed',fname))

    neurons=data['neurons']
    center_x=neurons['pos10x_x']
    center_y=neurons['pos10x_y']
    pr=px/rescale_factor
    overlap_half_width=np.round(box_half_width_um/pr)
    search_radius=np.round(search_radius_um/pr)
    xmin=center_x-overlap_half_width
    xmax=center_x+overlap_half_width
    ymin=center_y-overlap_half_width
    ymax=center_y+overlap_half_width
    c=np.column_stack((center_x,center_y))

    #max_dist=np.sqrt(2)*search_radius
    dist=cdist(c,c,'euclidean')
    dist_nearest=(dist<search_radius)
    [cell_id,nearest_neigh_id]=np.nonzero(dist_nearest)
    fov_neigh=neurons['fov'][nearest_neigh_id]
    fov_cell=neurons['fov'][cell_id]

    sel_cells_id=fov_cell!=fov_neigh
    distances_neigh=dist[cell_id[sel_cells_id],nearest_neigh_id[sel_cells_id]]
    search_cells_id=cell_id[sel_cells_id]
    search_neighbors_id=nearest_neigh_id[sel_cells_id]
    search_dist=distances_neigh
    id_overlap=((((xmin[search_cells_id]<xmin[search_neighbors_id])&(xmin[search_neighbors_id]<xmax[search_cells_id])) |
                 ((xmin[search_cells_id]<xmax[search_neighbors_id])&(xmax[search_neighbors_id]<xmax[search_cells_id]))) & 
                 (((ymin[search_cells_id]<ymin[search_neighbors_id])&(ymin[search_neighbors_id]<ymax[search_cells_id])) | 
                 ((ymin[search_cells_id]<ymax[search_neighbors_id])&(ymax[search_neighbors_id]<ymax[search_cells_id]))))
    overlap_cells_id=search_cells_id[id_overlap]
    overlap_neighbors_id=search_neighbors_id[id_overlap]
    overlap_distance=search_dist[id_overlap]
    # for i,idc in enumerate(overlap_cells_id):
    #     print(f"Cell {neurons['id'][idc]} in fov {neurons['fov_names'][neurons['fov'][idc]]} is matched to cell {neurons['id'][overlap_neighbors_id[i]]} in fov {neurons['fov_names'][neurons['fov'][overlap_neighbors_id[i]]]} with distance {overlap_distance[i]}")
    exp_mat=neurons['expmat'].todense() # I should do it in csr rather than dense--memory efficient
    total_exp_cell=np.asarray(np.sum(exp_mat,axis=1))
    [_,rev_idx]=np.unique(neurons['id'][overlap_cells_id],return_counts=False,return_inverse=True)
    df=pd.DataFrame({'cell':neurons['id'][overlap_cells_id],'neigh':neurons['id'][overlap_neighbors_id] ,'group':np.transpose(rev_idx),'neigh_exp':total_exp_cell[overlap_neighbors_id].flatten(),'cell_exp':total_exp_cell[overlap_cells_id].flatten()})
    idx_max=df.groupby('group')['neigh_exp'].idxmax()   
    # df=pd.DataFrame({'cell':neurons['id'][overlap_cells_id],'neigh':neurons['id'][overlap_neighbors_id] ,'neigh_exp':total_exp_cell[overlap_neighbors_id].flatten(),'cell_exp':total_exp_cell[overlap_cells_id].flatten()})
    # idx_max=df.groupby('cell')['neigh_exp'].idxmax()   
    df_ref=df.loc[idx_max]
    remove_cell=df_ref['neigh_exp']>df_ref['cell_exp']
    is_removed=np.zeros(center_x.shape)
    is_removed[overlap_cells_id[idx_max][remove_cell]]=1
    id_to_keep=is_removed==0
    filt_neurons={}
    expmat=coo_matrix(exp_mat[id_to_keep,:])
    filt_neurons['expmat']=expmat
    filt_neurons['id']=neurons['id'][id_to_keep]
    filt_neurons['pos10x_x']=neurons['pos10x_x'][id_to_keep]
    filt_neurons['pos10x_y']=neurons['pos10x_y'][id_to_keep]
    filt_neurons['pos40x_x']=neurons['pos40x_x'][id_to_keep]
    filt_neurons['pos40x_y']=neurons['pos40x_y'][id_to_keep]
    filt_neurons['slice']=neurons['slice'][id_to_keep]
    filt_neurons['genes']=neurons['genes']
    filt_neurons['fov']=neurons['fov'][id_to_keep]
    filt_neurons['fov_names']=neurons['fov_names']
    dump({"filt_neurons":filt_neurons,"removecells_all":id_to_keep},os.path.join(pth,'processed','filt_neurons.joblib'))
    print('OVERLAPPING CELLS REMOVED--PROCESSING FINISHED')

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

    filter_data_( infiles=args.infiles, 
                            outfiles=args.outfiles,
                            stage=args.stage,  
                            cp=cp )
    logging.info(f'done processing output to {args.outfiles[0]}')
