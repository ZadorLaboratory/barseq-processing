import tifffile as tif
import os
import numpy as np
import re
import time
import bardensr
import bardensr.plotting
import shutil
import cv2
import scipy as sp
import scipy.io
import pandas as pd
import matplotlib.pylab as plt
import random
from PIL import Image
import pathlib
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import skimage
from cellpose import models, io
import napari
import ray
import pickle
import helpers
import mapseq_helpers
import argparse
import configparser
import tkinter as tk
import datetime
from skimage import measure
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_config_file(dataset_path):
    config = configparser.ConfigParser()
    config['DATASET_SPECIFIC'] = {
            'dataset_path': dataset_path, 'animal_id': 'animal_id', 'stack_name': 'gfp...', 'stack_id': '7',
            'stack_session': '1', 'scan_session': '1', 'scan_idx': '[5, 10]',
            'slices_xy_rotation': '-30', 'xy_stack_offsets': '(0, 0)',
            'barseq_round': '1', 'nr_slices': '30', 'slice_thickness': '20', 'slice_range': '(0, 29)', 'library_used': '156',
            'bv_slices_name': 'preseq_1', 'bv_chan': '1',
            'inj_slices_name': 'preseq_1', 'inj_chan': '0',
            'somas_slices_name': 'hybseq_1', 'somas_chan': '2', 'hybseq_bv_chan': '0',
            'cropped_dimension_pixels': '5528',
            'registration_method': 'PHASE', 'allowed_reg_shift': '20', 'max_checks': '20',
            'stardist_probability_threshold': '0.35', 'basecalling_thresh': '0.55', 'noisefloor': '0.05',
            'basecalling_score_thresh': '0.85',
            'stitching_overlap': '0.15', 'stitching_buffer': '200',
            'downsample_factor': '2.5',
            'func_centroids_file': '[Stack]FieldSegmentation_StackUnit.pkl.csv', 'struct_centroids_file': '[Stack]Segmentation_Units.pkl.csv',
            'functional_data_tables': '[]',
            'positions': '[]', 'positions_to_rotate_180': '[]',
            'funseq_positions': '[]',
            'geneseq_name': 'geneseq', 'geneseq_segmentation_channel': '3', 'geneseq_soma_channels': '(0,4)',
            'bcseq_name': 'bcseq', 'bcseq_segmentation_channel': '3', 'bcseq_soma_channels': '(0,4)',
            'hybseq_name': 'hybseq', 'hybseq_dapi_channel': '3', 'hybseq_soma_channels': '(0,5)',
            'hybseq_inhib_channel': '1', 'hybseq_excite_channel': '0', 'hybseq_IT_channel': '3', 'hybseq_rolonies_channel': '2',
            'preseq_name': 'preseq', 'maxproj_name': 'maxproj', 'preseq_soma_channels': '(0,5)',
            'flip_horizontally': 'True',
            'codebook_matlab_path': '/Users/soitu/Desktop/code/bardensr/helper/codebook_163',
            'stardist_model_path': '/Users/soitu/Desktop/code/stardist/models2D/',
            'stardist_model_name': '2D_demo'
                                }
    #save config file as toml file in same folder as dataset_path (in this case, /Users/soitu/Desktop/datasets/)
    with open(config['DATASET_SPECIFIC']['dataset_path'] + 'config.toml', 'w') as configfile:
        config.write(configfile)

def parse_me_args(dataset_path):

    parser = argparse.ArgumentParser()
    # make parser function for boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    #define a type that take a string and returns a list of strings separated by commas.
    #it also has to lose brackets and quotes
    def str2list(v):
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace("'", '')
        v = v.replace('"', '')
        v = v.replace(' ', '')
        v = v.split(',')
        return v

    def list2int(v):
        #define a type that takes a list on strings separated by commas and returns a list of ints
        # lose brackets, quotes spaces
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace("'", '')
        v = v.replace('"', '')
        v = v.replace(' ', '')
        v = v.split(',')
        v = [int(i) for i in v]
        return v

    def tuple_float_type(strings):
        strings = strings.replace("(", "").replace(")", "")
        mapped_float = map(float, strings.split(","))
        return tuple(mapped_float)

    def tuple_int_type(strings):
        strings = strings.replace("(", "").replace(")", "")
        mapped_int = map(int, strings.split(","))
        return tuple(mapped_int)
    # if config file is not present in dataset_path + dataset_name, create_config_file. Else, load config file
    if not os.path.isfile(dataset_path + 'config.toml'):
        create_config_file(dataset_path)
        breakpoint()
    else:
        config = configparser.ConfigParser()
        config.read(dataset_path + 'config.toml')
        #change the dataset_path in the config file to the one that was passed using the --dataset_path argument
        config['DATASET_SPECIFIC']['dataset_path'] = dataset_path
        #save config file as toml file in same folder as dataset_path (in this case, /Users/soitu/Desktop/datasets/)
        with open(config['DATASET_SPECIFIC']['dataset_path'] + 'config.toml', 'w') as configfile:
            config.write(configfile)

    # add arguments with the same name from configfile. do not suggest names that are not in configfile

    parser.add_argument('-dataset_p', '--dataset_path', type=str, default=config['DATASET_SPECIFIC']['dataset_path'], help='path dataset to analyse')
    parser.add_argument('-animal_id', '--animal_id', type=str, default=config['DATASET_SPECIFIC']['animal_id'], help='animal_id')
    parser.add_argument('-barseq_round', '--barseq_round', type=int, default=config['DATASET_SPECIFIC']['barseq_round'], help='number of barseq round')
    parser.add_argument('-nr_slices', '--nr_slices', type=int, default=config['DATASET_SPECIFIC']['nr_slices'], help='number of slices processed')
    parser.add_argument('-slice_thick', '--slice_thickness', type=int, default=config['DATASET_SPECIFIC']['slice_thickness'], help='slice thickness in microns')
    parser.add_argument('-slice_range', '--slice_range', type=tuple, default=config['DATASET_SPECIFIC']['slice_range'], help='range of slices processed in this barseq round')
    parser.add_argument('-library', '--library_used', type=tuple, default=config['DATASET_SPECIFIC']['library_used'], help='gene library used for this')

    parser.add_argument('-bv_slices_name', '--bv_slices_name', type=str, default=config['DATASET_SPECIFIC']['bv_slices_name'], help='name of folder containing bv slices')
    parser.add_argument('-bv_chan', '--bv_chan', type=int, default=config['DATASET_SPECIFIC']['bv_chan'], help='channel of bv slices')
    parser.add_argument('-inj_slices_n', '--inj_slices_name', type=str, default=config['DATASET_SPECIFIC']['inj_slices_name'], help='in vitro stack of slices with injection and blood vessels')
    parser.add_argument('-inj_chan', '--inj_chan', type=int, default=config['DATASET_SPECIFIC']['inj_chan'], help='index of channel with injection site')
    parser.add_argument('-somas_slices_n', '--somas_slices_name', type=str, default=config['DATASET_SPECIFIC']['somas_slices_name'], help='name of folder containing soma slices')
    parser.add_argument('-somas_chan', '--somas_chan', type=int, default=config['DATASET_SPECIFIC']['somas_chan'], help='channel of soma slices')
    parser.add_argument('-stack_n', '--stack_name', type=str, default=config['DATASET_SPECIFIC']['stack_name'], help='name of in vivo stack to analyze')
    parser.add_argument('-stack_id', '--stack_id', type=int, default=config['DATASET_SPECIFIC']['stack_id'], help='id of functional stack analysed')
    parser.add_argument('-stack_sess', '--stack_session', type=int, default=config['DATASET_SPECIFIC']['stack_session'], help='stack session')
    parser.add_argument('-scanidx', '--scan_idx', type=list2int, default=config['DATASET_SPECIFIC']['scan_idx'], help='list of scans')
    parser.add_argument('-scan_sess', '--scan_session', type=int, default=config['DATASET_SPECIFIC']['scan_session'], help='scan session')
    parser.add_argument('-func_f', '--func_centroids_file', type=str, default=config['DATASET_SPECIFIC']['func_centroids_file'],
                        help='name of functional centroids file')
    parser.add_argument('-struct_f', '--struct_centroids_file', type=str, default=config['DATASET_SPECIFIC']['struct_centroids_file'],
                        help='name of functional centroids file')
    parser.add_argument('--functional_data_tables', default=config['DATASET_SPECIFIC']['functional_data_tables'], type=str2list, help='readout wieghts and meis files for each relevant scan')

    parser.add_argument('-slices_xy_rot', '--slices_xy_rotation', default=config['DATASET_SPECIFIC']['slices_xy_rotation'], type=int, help='xy macro-rotation for slices')
    parser.add_argument('-xy_st_offset', '--xy_stack_offsets', default=config['DATASET_SPECIFIC']['xy_stack_offsets'], type=tuple_float_type, help='xy offset to wrt global meso coordinates')
    parser.add_argument('-cropped_dims', '--cropped_dimension_pixels', default=config['DATASET_SPECIFIC']['cropped_dimension_pixels'], type=int, help='crop slices to what dimension')
    parser.add_argument('-reg_method', '--registration_method', default=config['DATASET_SPECIFIC']['registration_method'], type=str, help='registration method. 3 options: ECC, ORB, PHASE')
    parser.add_argument('--allowed_reg_shift', type=int, default=config['DATASET_SPECIFIC']['allowed_reg_shift'], help='maximum shift allowed between tiles of same position')
    parser.add_argument('-nr_max_checks', '--max_checks', type=int, default=config['DATASET_SPECIFIC']['max_checks'], help='max number of tiles to export for checks')
    parser.add_argument('-star_prob_thresh', '--stardist_probability_threshold', type=float, default=config['DATASET_SPECIFIC']['stardist_probability_threshold'], help='stardist probability threshold')
    parser.add_argument('-bcall_thresh', '--basecalling_thresh', type=float, default=config['DATASET_SPECIFIC']['basecalling_thresh'], help='threshold for bardensr basecalling')
    parser.add_argument('-basecall_thresh', '--basecalling_score_thresh', type=float, default=config['DATASET_SPECIFIC']['basecalling_score_thresh'], help='threshold to consider cells as infected')
    parser.add_argument('-noise_floor', '--noisefloor', type=float, default=config['DATASET_SPECIFIC']['noisefloor'], help='noisefloor for bardensr')
    parser.add_argument('-stitch_overlap', '--stitching_overlap', type=float, default=config['DATASET_SPECIFIC']['stitching_overlap'], help='default overlap for stitching')
    parser.add_argument('-stitch_buffer', '--stitching_buffer', type=int, default=config['DATASET_SPECIFIC']['stitching_buffer'], help='stitching buffer in pixels')
    parser.add_argument('-ds_factor', '--downsample_factor', type=float, default=config['DATASET_SPECIFIC']['downsample_factor'], help='factor to downsample images to visualize resulta and reduce size')
    parser.add_argument('-pos', '--positions', default=config['DATASET_SPECIFIC']['positions'], type=str2list, nargs='*', help='all positions to process')
    parser.add_argument('-pos_to_rotate', '--positions_to_rotate_180', default=config['DATASET_SPECIFIC']['positions_to_rotate_180'], type=tuple, help='all positions to rotate')
    parser.add_argument('-fun_pos', '--funseq_positions', default=config['DATASET_SPECIFIC']['funseq_positions'], type=str2list, help='all positions with valid barcodes')
    parser.add_argument('-bcseq_n', '--bcseq_name', type=str, default=config['DATASET_SPECIFIC']['bcseq_name'], help='name of bcseq folder')
    parser.add_argument('-bcseq_soma_chan', '--bcseq_soma_channels', type=tuple_int_type, default=config['DATASET_SPECIFIC']['bcseq_soma_channels'], help='channels for bcseq where somas are clear. first element:last element of tuple')
    parser.add_argument('-bcseq_seg_chan', '--bcseq_segmentation_channel', type=int, default=config['DATASET_SPECIFIC']['bcseq_segmentation_channel'], help='channel to use for bcseq segmentation. choose sth with clear soma outline and few rolonies')
    parser.add_argument('-hybseq_n', '--hybseq_name', type=str, default=config['DATASET_SPECIFIC']['hybseq_name'], help='name of hybseq folder')
    parser.add_argument('-hybseq_soma_chan', '--hybseq_soma_channels', type=tuple_int_type, default=config['DATASET_SPECIFIC']['hybseq_soma_channels'], help='channels for hybseq where somas are clear. first element:last element of tuple')
    parser.add_argument('--hybseq_dapi_channel', type=int, default=config['DATASET_SPECIFIC']['hybseq_dapi_channel'], help='channel to use for hybseq segmentation. choose sth with clear soma outline and few rolonies')
    parser.add_argument('--hybseq_inhib_channel', type=int, default=config['DATASET_SPECIFIC']['hybseq_inhib_channel'], help='channel to use for hybseq inhibitory neurons')
    parser.add_argument('--hybseq_excite_channel', type=int, default=config['DATASET_SPECIFIC']['hybseq_excite_channel'], help='channel to use for hybseq excitatory neurons')
    parser.add_argument('--hybseq_IT_channel', type=int, default=config['DATASET_SPECIFIC']['hybseq_IT_channel'], help='channel to use for hybseq IT neurons')
    parser.add_argument('--hybseq_rolonies_channel', type=int, default=config['DATASET_SPECIFIC']['hybseq_rolonies_channel'], help='channel to use to visualize all rolonies in hybseq cycles')
    parser.add_argument('--hybseq_bv_chan', type=int, default=config['DATASET_SPECIFIC']['hybseq_bv_chan'], help='channel with blood vessels in hybseq')
    parser.add_argument('-geneseq_n', '--geneseq_name', type=str, default=config['DATASET_SPECIFIC']['geneseq_name'], help='name of geneseq folder')
    parser.add_argument('-geneseq_soma_chan', '--geneseq_soma_channels', type=tuple_int_type, default=config['DATASET_SPECIFIC']['geneseq_soma_channels'], help='channels for geneseq where somas are clear. first element:last element of tuple')
    parser.add_argument('-geneseq_seg_chan', '--geneseq_segmentation_channel', type=int, default=config['DATASET_SPECIFIC']['geneseq_segmentation_channel'], help='channel to use for genseq segmentation. choose sth with clear soma outline and few rolonies')
    parser.add_argument('-preseq_n', '--preseq_name', type=str, default=config['DATASET_SPECIFIC']['preseq_name'], help='name of preseq folder')
    parser.add_argument('-preseq_soma_chan', '--preseq_soma_channels', type=tuple_int_type, default=config['DATASET_SPECIFIC']['preseq_soma_channels'], help='channels for preseq where somas are clear. first element:last element of tuple')
    parser.add_argument('-flip_bool', '--flip_horizontally', type=str2bool, default=config['DATASET_SPECIFIC']['flip_horizontally'], help='flip horizontally or not')

    # Do not try to set the following arguments. They will be initiated as a function of previous args.
    parser.add_argument('-config', '--config_file', type=str, default=dataset_path + 'config.toml', help='path to config file')
    parser.add_argument('-slices_p', '--slices_path', type=str, help='path to slices folder')
    parser.add_argument('-stack_p', '--stack_path', type=str, help='path to stack folder')
    parser.add_argument('-stack_landmarks_p', '--stack_landmarks_path', type=str, help='path to stack/landmarks folder')
    parser.add_argument('-scan_p', '--scan_path', type=str, help='path to scan folder')
    parser.add_argument('-mapseq_p', '--mapseq_path', type=str, help='path to mapseq folder')
    parser.add_argument('-proc_p', '--proc_path', type=str, help='path to processed')

    parser.add_argument('-matching_p', '--matching_path', type=str, help='path to match')
    parser.add_argument('-matching_centroids_p', '--matching_centroids_path', type=str, help='path to matching/centroids')
    parser.add_argument('-matching_matched_p', '--matching_matched_path', type=str, help='path to matching/matched')
    parser.add_argument('-matching_images_p', '--matching_images_path', type=str, help='path to matching/images')
    parser.add_argument('-matching_tables_p', '--matching_tables_path', type=str, help='path to matching/tables')
    parser.add_argument('--matching_MEIs_path', type=str, help='path to matching/MEIs')
    parser.add_argument('--matching_weights_path', type=str, help='path to matching/weights - these are readout weights')
    parser.add_argument('--matching_properties_path', type=str, help='path to matching/properties - these are the rest of the functional properties')
    parser.add_argument('--history_path', type=str, help='path to history to access backups')

    parser.add_argument('--analysis_path', type=str, help='path to analysis')
    parser.add_argument('--analysis_funseq_path', type=str, help='path to analysis/funseq')
    parser.add_argument('--analysis_funseq_tables_path', type=str, help='path to analysis/funseq/tables')
    parser.add_argument('--analysis_funseq_plots_path', type=str, help='path to analysis/funseq/plots')
    parser.add_argument('--analysis_geneseq_path', type=str, help='path to analysis/geneseq')
    parser.add_argument('--analysis_geneseq_tables_path', type=str, help='path to analysis/geneseq/tables')
    parser.add_argument('--analysis_geneseq_plots_path', type=str, help='path to analysis/geneseq/plots')

    parser.add_argument('-proc_transf_p', '--proc_transf_path', type=str, help='path to processed/transformations')
    parser.add_argument('-proc_transf_macro_p', '--proc_transf_macro_path', type=str, help='path to processed/transformations/macro')
    parser.add_argument('-proc_transf_align_p', '--proc_transf_align_path', type=str, help='path to processed/transformations/align')
    parser.add_argument('-proc_transf_align_geneseq_p', '--proc_transf_align_geneseq_path', type=str, help='path to processed/transformations/align/geneseq')
    parser.add_argument('-proc_transf_align_bcseq_p', '--proc_transf_align_bcseq_path', type=str, help='path to processed/transformations/align/bcseq')
    parser.add_argument('-proc_transf_align_hybseq_p', '--proc_transf_align_hybseq_path', type=str, help='path to processed/transformations/align/hybseq')
    parser.add_argument('-proc_transf_align_preseq_p', '--proc_transf_align_preseq_path', type=str, help='path to processed/transformations/align/preseq')
    parser.add_argument('-proc_samples_p', '--proc_samples_path', type=str, help='path to processed/samples')
    parser.add_argument('-proc_slices_p', '--proc_slices_path', type=str, help='path to processed/slices')
    parser.add_argument('-proc_stack_p', '--proc_stack_path', type=str, help='path to processed/stack')
    parser.add_argument('-proc_scan_p', '--proc_scan_path', type=str, help='path to processed/scan')
    parser.add_argument('-proc_coordinates_p', '--proc_coordinates_path', type=str, help='path to processed/coordinates')
    parser.add_argument('-proc_summary_p', '--proc_summary_path', type=str, help='path to processed/summary')
    parser.add_argument('-proc_illumination_p', '--proc_illumination_path', type=str, help='path to processed/illumination')
    parser.add_argument('-proc_illumination_geneseq_p', '--proc_illumination_geneseq_path', type=str, help='path to processed/illumination/geneseq')
    parser.add_argument('-proc_illumination_bcseq_p', '--proc_illumination_bcseq_path', type=str, help='path to processed/illumination/bcseq')
    parser.add_argument('-proc_illumination_hybseq_p', '--proc_illumination_hybseq_path', type=str, help='path to processed/illumination/hybseq')
    parser.add_argument('-proc_illumination_preseq_p', '--proc_illumination_preseq_path', type=str, help='path to processed/illumination/preseq')
    parser.add_argument('-proc_orig_p', '--proc_original_path', type=str, help='path to processed/original')
    parser.add_argument('-proc_orig_geneseq_p', '--proc_original_geneseq_path', type=str, help='path to processed/original/geneseq/')
    parser.add_argument('-proc_orig_bcseq_p', '--proc_original_bcseq_path', type=str, help='path to processed/original/bcseq/')
    parser.add_argument('-proc_orig_hybseq_p', '--proc_original_hybseq_path', type=str, help='path to processed/original/hybseq/')
    parser.add_argument('-proc_orig_preseq_p', '--proc_original_preseq_path', type=str, help='path to processed/original/preseq/')
    parser.add_argument('-proc_checks_p', '--proc_checks_path', type=str, help='path to processed/checks')
    parser.add_argument('-proc_checks_scan_p', '--proc_checks_scan_path', type=str, help='path to processed/checks/scan')
    parser.add_argument('-proc_checks_align_p', '--proc_checks_alignment_path', type=str, help='path to processed/checks/alignment')
    parser.add_argument('-proc_checks_align_stitch_p', '--proc_checks_alignment_stitch_path', type=str, help='path to processed/checks/alignment/stitch')
    parser.add_argument('-proc_checks_align_raw_p', '--proc_checks_alignment_raw_path', type=str, help='path to processed/checks/alignment/raw')
    parser.add_argument('-proc_checks_basecall_p', '--proc_checks_basecalling_path', type=str, help='path to processed/checks/basecalling')
    parser.add_argument('-proc_checks_illumination_p', '--proc_checks_illumination_path', type=str, help='path to processed/checks/illumination')
    parser.add_argument('-proc_checks_illumination_geneseq_p', '--proc_checks_illumination_geneseq_path', type=str, help='path to processed/checks/illumination/geneseq')
    parser.add_argument('-proc_checks_illumination_bcseq_p', '--proc_checks_illumination_bcseq_path', type=str, help='path to processed/checks/illumination/bcseq')
    parser.add_argument('-proc_checks_illumination_hybseq_p', '--proc_checks_illumination_hybseq_path', type=str, help='path to processed/checks/illumination/hyseq')
    parser.add_argument('-proc_checks_illumination_preseq_p', '--proc_checks_illumination_preseq_path', type=str, help='path to processed/checks/illumination/preseq')
    parser.add_argument('-proc_checks_segmentation_p', '--proc_checks_segmentation_path', type=str, help='path to processed/checks/segmentation')
    parser.add_argument('-proc_checks_segmentation_geneseq_p', '--proc_checks_segmentation_geneseq_path', type=str, help='path to processed/checks/segmentation/geneseq')
    parser.add_argument('-proc_checks_segmentation_bcseq_p', '--proc_checks_segmentation_bcseq_path', type=str, help='path to processed/checks/segmentation/bcseq')
    parser.add_argument('-proc_checks_segmentation_hybseq_p', '--proc_checks_segmentation_hybseq_path', type=str, help='path to processed/checks/segmentation/hybseq')
    parser.add_argument('-proc_checks_segmentation_final_p', '--proc_checks_segmentation_final_path', type=str, help='path to processed/checks/segmentation/final')
    parser.add_argument('-proc_checks_color_corr_p', '--proc_checks_color_correction_path', type=str, help='path to processed/checks/color_correction')
    parser.add_argument('-proc_aligned_p', '--proc_aligned_path', type=str, help='path to processed/aligned')
    parser.add_argument('-proc_aligned_geneseq_p', '--proc_aligned_geneseq_path', type=str, help='path to processed/aligned/geneseq')
    parser.add_argument('-proc_aligned_bcseq_p', '--proc_aligned_bcseq_path', type=str, help='path to processed/aligned/bcseq')
    parser.add_argument('-proc_aligned_hybseq_p', '--proc_aligned_hybseq_path', type=str, help='path to processed/aligned/hybseq')
    parser.add_argument('-proc_aligned_preseq_p', '--proc_aligned_preseq_path', type=str, help='path to processed/aligned/preseq')
    parser.add_argument('-proc_seg_p', '--proc_segmented_path', type=str, help='path to processed/segmented')
    parser.add_argument('-proc_seg_orig_p', '--proc_segmented_original_path', type=str, help='path to processed/segmented/original')
    parser.add_argument('-proc_seg_barc_p', '--proc_segmented_barcoded_path', type=str, help='path to processed/segmented/barcoded')
    parser.add_argument('-proc_seg_somas_p', '--proc_segmented_somas_path', type=str, help='path to processed/segmented/somas')
    parser.add_argument('-proc_seg_orig_geneseq_p', '--proc_segmented_original_geneseq_path', type=str, help='path to processed/segmented/original/geneseq/')
    parser.add_argument('-proc_seg_orig_bcseq_p', '--proc_segmented_original_bcseq_path', type=str, help='path to processed/segmented/original/bcseq/')
    parser.add_argument('-proc_seg_orig_hybseq_p', '--proc_segmented_original_hybseq_path', type=str, help='path to processed/segmented/original/hybseq/')
    parser.add_argument('-proc_seg_aligned_p', '--proc_segmented_aligned_path', type=str, help='path to processed/segmented/aligned')
    parser.add_argument('-proc_seg_aligned_geneseq_p', '--proc_segmented_aligned_geneseq_path', type=str, help='path to processed/segmented/aligned/geneseq')
    parser.add_argument('-proc_seg_aligned_bcseq_p', '--proc_segmented_aligned_bcseq_path', type=str, help='path to processed/segmented/aligned/bcseq')
    parser.add_argument('-proc_seg_aligned_hybseq_p', '--proc_segmented_aligned_hybseq_path', type=str, help='path to processed/segmented/aligned/hybseq')
    parser.add_argument('-proc_seg_final_p', '--proc_segmented_final_path', type=str, help='path to processed/segmented/final')
    parser.add_argument('-proc_seg_final_geneseq_p', '--proc_segmented_final_geneseq_path', type=str, help='path to processed/segmented/final/geneseq')
    parser.add_argument('-proc_seg_final_bcseq_p', '--proc_segmented_final_bcseq_path', type=str, help='path to processed/segmented/final/bcseq')
    parser.add_argument('-proc_seg_final_hybseq_p', '--proc_segmented_final_hybseq_path', type=str, help='path to processed/segmented/final/hybseq')
    parser.add_argument('-proc_stitched_p', '--proc_stitched_path', type=str, help='path to processed/stitched')
    parser.add_argument('-proc_stitched_geneseq_p', '--proc_stitched_geneseq_path', type=str, help='path to processed/stitched/geneseq')
    parser.add_argument('-proc_stitched_bcseq_p', '--proc_stitched_bcseq_path', type=str, help='path to processed/stitched/bcseq')
    parser.add_argument('-proc_stitched_hybseq_p', '--proc_stitched_hybseq_path', type=str, help='path to processed/stitched/hybseq')
    parser.add_argument('-proc_stitched_preseq_p', '--proc_stitched_preseq_path', type=str, help='path to processed/stitched/preseq')
    parser.add_argument('-proc_stitched_funseq_p', '--proc_stitched_funseq_path', type=str, help='path to processed/stitched/funseq')
    parser.add_argument('-proc_stitched_funseq_bv_p', '--proc_stitched_funseq_bv_path', type=str, help='path to processed/stitched/funseq/bv')
    parser.add_argument('-proc_stitched_funseq_somas_p', '--proc_stitched_funseq_somas_path', type=str, help='path to processed/stitched/funseq/somas')
    parser.add_argument('-proc_slice_stats_p', '--proc_slice_stats_path', type=str, help='path to processed/slice_stats')
    parser.add_argument('-proc_disp_rol_p', '--proc_display_rolonies_path', type=str, help='path to processed/display_rolonies')
    parser.add_argument('-proc_genes_p', '--proc_genes_path', type=str, help='path to processed/genes')
    parser.add_argument('-proc_centroids_p', '--proc_centroids_path', type=str, help='path to processed/centroids')
    parser.add_argument('-proc_transformed_p', '--proc_transformed_path', type=str, help='path to processed/transformed')
    parser.add_argument('-proc_transformed_geneseq_p', '--proc_transformed_geneseq_path', type=str, help='path to processed/transformed/geneseq')
    parser.add_argument('-proc_transformed_bcseq_p', '--proc_transformed_bcseq_path', type=str, help='path to processed/transformed/bcseq')
    parser.add_argument('-proc_transformed_hybseq_p', '--proc_transformed_hybseq_path', type=str, help='path to processed/transformed/hybseq')
    parser.add_argument('-proc_transformed_preseq_p', '--proc_transformed_preseq_path', type=str, help='path to processed/transformed/preseq')

    parser.add_argument('-proc_bv_slices_n', '--proc_bv_slices_name', type=str, default='blood_vessels.tif', help='name of processed blood vessel stack')
    parser.add_argument('-proc_inj_slices_n', '--proc_inj_slices_name', type=str, default='injection.tif', help='name of processed injection stack')
    parser.add_argument('-proc_somas_slices_n', '--proc_somas_slices_name', type=str, default='somas.tif', help='name of processed somas stack')

    parser.add_argument('-func_cent_n', '--func_centroids_name', default='func_centroids', type=str, help='name of functional centroids file')
    parser.add_argument('-struct_cent_n', '--struct_centroids_name', default='struct_centroids', type=str, help='name of structural centroids file')
    parser.add_argument('-rot_func_cent_n', '--rotated_func_centroids_name', default='rotated_func_centroids', type=str,
                        help='name of rotated functional centroids file')
    parser.add_argument('-rot_struct_cent_n', '--rotated_struct_centroids_name', default='rotated_struct_centroids', type=str,
                        help='name of rotated structural centroids file')

    parser.add_argument('-gfp_stack_n', '--gfp_stack_name', default='gfp.tif', type=str, help='name of gfp stack after deinterleaving')
    parser.add_argument('-bv_stack_n', '--bv_stack_name', default='blood_vessels.tif', type=str, help='name of bv stack after deinterleaving')
    parser.add_argument('-proc_bv_stack_n', '--proc_bv_stack_name', default='rotated_blood_vessels_stack.tif', type=str, help='name of processed bv stack')
    parser.add_argument('-proc_gfp_stack_n', '--proc_gfp_stack_name', default='rotated_gfp_stack.tif', type=str, help='name of processed gfp stack')

    parser.add_argument('-codebook_path', '--codebook_matlab_path', type=str, default=config['DATASET_SPECIFIC']['codebook_matlab_path'], help='path to codebook')
    parser.add_argument('-cellpose_model_p', '--cellpose_model_path', type=str, default='/Users/soitu/Desktop/cellpose_models/CS_ISS_13Dec', help='path to cellpose model')
    parser.add_argument('-star_model_p', '--stardist_model_path', type=str, default='/Users/soitu/Desktop/code/stardist/models2D/', help='path to stardist model')
    parser.add_argument('-star_model_n', '--stardist_model_name', type=str, default='trained_on_100x_clampFISH', help='name of stardist model')
    parser.add_argument('-nr_cpus', '--nr_cpus', type=int, default=4, help='nr cpus to paralellize job')


    args = parser.parse_args()

    args.slices_path = helpers.quick_dir(args.dataset_path, 'slices')
    args.stack_path = helpers.quick_dir(args.dataset_path, 'stack')
    args.stack_landmarks_path = helpers.quick_dir(args.stack_path, 'landmarks')
    args.scan_path = helpers.quick_dir(args.dataset_path, 'scan')

    args.mapseq_path = helpers.quick_dir(args.dataset_path, 'mapseq')
    args.history_path = helpers.quick_dir(args.dataset_path, 'history')
    args.matching_path = helpers.quick_dir(args.dataset_path, 'matching')
    args.matching_centroids_path = helpers.quick_dir(args.matching_path, 'centroids')
    args.matching_images_path = helpers.quick_dir(args.matching_path, 'images')
    args.matching_matched_path = helpers.quick_dir(args.matching_path, 'matched')
    args.matching_tables_path = helpers.quick_dir(args.matching_path, 'tables')
    args.matching_MEIs_path = helpers.quick_dir(args.matching_path, 'MEIs')
    args.matching_weights_path = helpers.quick_dir(args.matching_path, 'weights')
    args.matching_properties_path = helpers.quick_dir(args.matching_path, 'properties')

    args.analysis_path = helpers.quick_dir(args.dataset_path, 'analysis')
    args.analysis_funseq_path = helpers.quick_dir(args.analysis_path, 'funseq')
    args.analysis_funseq_tables_path = helpers.quick_dir(args.analysis_funseq_path, 'tables')
    args.analysis_funseq_plots_path = helpers.quick_dir(args.analysis_funseq_path, 'plots')
    args.analysis_geneseq_path = helpers.quick_dir(args.analysis_path, 'geneseq')
    args.analysis_geneseq_tables_path = helpers.quick_dir(args.analysis_geneseq_path, 'tables')
    args.analysis_geneseq_plots_path = helpers.quick_dir(args.analysis_funseq_tables_path, 'plots')

    args.proc_path = helpers.quick_dir(args.dataset_path, 'processed')
    args.proc_transf_path = helpers.quick_dir(args.proc_path, 'transformations')
    args.proc_transf_macro_path = helpers.quick_dir(args.proc_transf_path, 'macro')
    args.proc_transf_align_path = helpers.quick_dir(args.proc_transf_path, 'align')
    args.proc_transf_align_geneseq_path = helpers.quick_dir(args.proc_transf_align_path, 'geneseq')
    args.proc_transf_align_bcseq_path = helpers.quick_dir(args.proc_transf_align_path, 'bcseq')
    args.proc_transf_align_hybseq_path = helpers.quick_dir(args.proc_transf_align_path, 'hybseq')
    args.proc_transf_align_preseq_path = helpers.quick_dir(args.proc_transf_align_path, 'preseq')
    args.proc_slices_path = helpers.quick_dir(args.proc_path, 'slices')
    args.proc_stack_path = helpers.quick_dir(args.proc_path, 'stack')
    args.proc_scan_path = helpers.quick_dir(args.proc_path, 'scan')
    args.proc_coordinates_path = helpers.quick_dir(args.proc_path, 'coordinates')
    args.proc_samples_path = helpers.quick_dir(args.proc_path, 'samples')
    args.proc_summary_path = helpers.quick_dir(args.proc_path, 'summary')
    args.proc_illumination_path = helpers.quick_dir(args.proc_path, 'illumination')
    args.proc_illumination_geneseq_path = helpers.quick_dir(args.proc_illumination_path, 'geneseq')
    args.proc_illumination_bcseq_path = helpers.quick_dir(args.proc_illumination_path, 'bcseq')
    args.proc_illumination_hybseq_path = helpers.quick_dir(args.proc_illumination_path, 'hybseq')
    args.proc_illumination_preseq_path = helpers.quick_dir(args.proc_illumination_path, 'preseq')
    args.proc_original_path = helpers.quick_dir(args.proc_path, 'original')
    args.proc_original_geneseq_path = helpers.quick_dir(args.proc_original_path, args.geneseq_name)
    args.proc_original_bcseq_path = helpers.quick_dir(args.proc_original_path, args.bcseq_name)
    args.proc_original_hybseq_path = helpers.quick_dir(args.proc_original_path, args.hybseq_name)
    args.proc_original_preseq_path = helpers.quick_dir(args.proc_original_path, args.preseq_name)
    args.proc_aligned_path = helpers.quick_dir(args.proc_path, 'aligned')
    args.proc_aligned_geneseq_path = helpers.quick_dir(args.proc_aligned_path, args.geneseq_name)
    args.proc_aligned_bcseq_path = helpers.quick_dir(args.proc_aligned_path, args.bcseq_name)
    args.proc_aligned_hybseq_path = helpers.quick_dir(args.proc_aligned_path, args.hybseq_name)
    args.proc_aligned_preseq_path = helpers.quick_dir(args.proc_aligned_path, args.preseq_name)
    args.proc_checks_path = helpers.quick_dir(args.proc_path, 'checks')
    args.proc_checks_alignment_path = helpers.quick_dir(args.proc_checks_path, 'alignment')
    args.proc_checks_scan_path = helpers.quick_dir(args.proc_checks_path, 'scan')
    args.proc_checks_alignment_raw_path = helpers.quick_dir(args.proc_checks_alignment_path, 'raw')
    args.proc_checks_alignment_stitch_path = helpers.quick_dir(args.proc_checks_alignment_path, 'stitch')
    args.proc_checks_basecalling_path = helpers.quick_dir(args.proc_checks_path, 'basecalling')
    args.proc_checks_illumination_path = helpers.quick_dir(args.proc_checks_path, 'illumination')
    args.proc_checks_illumination_geneseq_path = helpers.quick_dir(args.proc_checks_illumination_path, 'geneseq')
    args.proc_checks_illumination_bcseq_path = helpers.quick_dir(args.proc_checks_illumination_path, 'bcseq')
    args.proc_checks_illumination_hybseq_path = helpers.quick_dir(args.proc_checks_illumination_path, 'hybseq')
    args.proc_checks_illumination_preseq_path = helpers.quick_dir(args.proc_checks_illumination_path, 'preseq')
    args.proc_checks_segmentation_path = helpers.quick_dir(args.proc_checks_path, 'segmentation')
    args.proc_checks_segmentation_geneseq_path = helpers.quick_dir(args.proc_checks_segmentation_path, args.geneseq_name)
    args.proc_checks_segmentation_bcseq_path = helpers.quick_dir(args.proc_checks_segmentation_path, args.bcseq_name)
    args.proc_checks_segmentation_hybseq_path = helpers.quick_dir(args.proc_checks_segmentation_path, args.hybseq_name)
    args.proc_checks_segmentation_final_path = helpers.quick_dir(args.proc_checks_segmentation_path, 'final')
    args.proc_checks_color_correction_path = helpers.quick_dir(args.proc_checks_path, 'color_correction')
    args.proc_segmented_path = helpers.quick_dir(args.proc_path, 'segmented')
    args.proc_segmented_original_path = helpers.quick_dir(args.proc_segmented_path, 'original')
    args.proc_segmented_original_geneseq_path = helpers.quick_dir(args.proc_segmented_original_path, args.geneseq_name)
    args.proc_segmented_original_bcseq_path = helpers.quick_dir(args.proc_segmented_original_path, args.bcseq_name)
    args.proc_segmented_original_hybseq_path = helpers.quick_dir(args.proc_segmented_original_path, args.hybseq_name)
    args.proc_segmented_aligned_path = helpers.quick_dir(args.proc_segmented_path, 'aligned')
    args.proc_segmented_aligned_geneseq_path = helpers.quick_dir(args.proc_segmented_aligned_path, args.geneseq_name)
    args.proc_segmented_aligned_bcseq_path = helpers.quick_dir(args.proc_segmented_aligned_path, args.bcseq_name)
    args.proc_segmented_aligned_hybseq_path = helpers.quick_dir(args.proc_segmented_aligned_path, args.hybseq_name)
    args.proc_segmented_final_path = helpers.quick_dir(args.proc_segmented_path, 'final')
    args.proc_segmented_barcoded_path = helpers.quick_dir(args.proc_segmented_path, 'barcoded')
    args.proc_segmented_somas_path = helpers.quick_dir(args.proc_segmented_path, 'somas')

    args.proc_segmented_final_geneseq_path = helpers.quick_dir(args.proc_segmented_final_path, args.geneseq_name)
    args.proc_segmented_final_bcseq_path = helpers.quick_dir(args.proc_segmented_final_path, args.bcseq_name)
    args.proc_segmented_final_hybseq_path = helpers.quick_dir(args.proc_segmented_final_path, args.hybseq_name)
    args.proc_stitched_path = helpers.quick_dir(args.proc_path, 'stitched')
    args.proc_stitched_geneseq_path = helpers.quick_dir(args.proc_stitched_path, args.geneseq_name)
    args.proc_stitched_bcseq_path = helpers.quick_dir(args.proc_stitched_path, args.bcseq_name)
    args.proc_stitched_hybseq_path = helpers.quick_dir(args.proc_stitched_path, args.hybseq_name)
    args.proc_stitched_preseq_path = helpers.quick_dir(args.proc_stitched_path, args.preseq_name)
    args.proc_stitched_funseq_path = helpers.quick_dir(args.proc_stitched_path, 'funseq')
    args.proc_stitched_funseq_bv_path = helpers.quick_dir(args.proc_stitched_funseq_path, 'bv')
    args.proc_stitched_funseq_somas_path = helpers.quick_dir(args.proc_stitched_funseq_path, 'somas')
    args.proc_slice_stats_path = helpers.quick_dir(args.proc_path, 'slice_stats')
    args.proc_display_rolonies_path = helpers.quick_dir(args.proc_path, 'display_rolonies')
    args.proc_genes_path = helpers.quick_dir(args.proc_path, 'genes')
    args.proc_centroids_path = helpers.quick_dir(args.proc_path, 'centroids')
    args.proc_transformed_path = helpers.quick_dir(args.proc_path, 'transformed')
    args.proc_transformed_geneseq_path = helpers.quick_dir(args.proc_transformed_path, args.geneseq_name)
    args.proc_transformed_bcseq_path = helpers.quick_dir(args.proc_transformed_path, args.bcseq_name)
    args.proc_transformed_hybseq_path = helpers.quick_dir(args.proc_transformed_path, args.hybseq_name)
    args.proc_transformed_preseq_path = helpers.quick_dir(args.proc_transformed_path, args.preseq_name)

    return args


def preprocess_slices(args):
    '''

    Preprocessing steps for slices. Images come in tiles so they have to be stiched, then flipped horizontally.
    User has to indicate blood vessel channel and injection channel
    :param slices_path:
    :param stack_name:
    :param blood_vessel_chan:
    :param injection_chan:
    :return:
    '''

    # stitch and flip images if this has not been done yet, i.e. check to see if stitched folder exists.
    #if not os.path.exists(args.slices_path + 'stitched_' + args.inj_slices_name + '_' + str(args.inj_chan)):
    folders_to_stitch = [args.inj_slices_name, args.bv_slices_name, args.somas_slices_name]
    channels_to_stitch = [args.inj_chan, args.bv_chan, args.somas_chan]
    output_folders_names = ['stitched_inj', 'stitched_bv', 'stitched_somas']
    stitched_paths = []

    for i in range(len(folders_to_stitch)):
        if not os.path.exists(args.slices_path + output_folders_names[i]):
            folder_to_stitch_path = helpers.quick_dir(args.slices_path, folders_to_stitch[i])
            folder_to_stitch_path = helpers.quick_dir(folder_to_stitch_path, 'maxproj')
            stitched_path = stitch_images_imperfectly_folder(folder_to_stitch_path, specific_chan=channels_to_stitch[i], output_folder_name='../' + output_folders_names[i])
            stitched_path = flip_horizontally(stitched_path, overwrite_images=True)
            stitched_paths.append(stitched_path)
        else:
            stitched_paths.append(helpers.quick_dir(args.slices_path, output_folders_names[i]))

    # extract dimensions of files from a sample slice
    slices_names = helpers.list_files(stitched_paths[0])
    #remove .tif or .tiff ending from file names if present
    slices_names = helpers.human_sort(slices_names)
    positions = [s.replace('.tiff', '') for s in slices_names]
    positions = [s.replace('.tif', '') for s in positions]
    sample_slice = tif.imread(stitched_paths[0] + slices_names[0])
    sample_slice = np.squeeze(sample_slice)
    center_x, center_y = sample_slice.shape[0] / 2, sample_slice.shape[1] / 2

    # initialize arrays for injection, blood vessel and somas stacks and load images
    preseq_inj_slices_stack = np.ndarray((len(slices_names), sample_slice.shape[0], sample_slice.shape[1]), dtype=np.int16)
    preseq_bv_slices_stack = np.zeros_like(preseq_inj_slices_stack, dtype=np.int16)
    somas_slices_stack = np.zeros_like(preseq_inj_slices_stack, dtype=np.int16)

    for slice_id, slice in enumerate(slices_names):
        preseq_inj_slices_stack[slice_id] = tif.imread(stitched_paths[0] + slice)
        preseq_bv_slices_stack[slice_id] = tif.imread(stitched_paths[1] + slice)
        somas_slices_stack[slice_id] = tif.imread(stitched_paths[2] + slice)

    # napari mark injection site to coarsely align slices to each other
    size_of_points = 0.018 * sample_slice.shape[1]
    slice_viewer = napari.view_image(preseq_inj_slices_stack, title='In vitro stack - mark injection site -> 1 point/slice then press s. mark slices to be rotated with r', name='invitro_stack')
    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    slice_viewer.add_points(np.empty((0, 3)), face_color="green", size=size_of_points, name='slice_saved_points', ndim=3)
    slice_viewer.add_points(face_color="red", size=size_of_points, name='slice_points', ndim=3)
    slice_viewer.layers['slice_points'].mode = 'add'

    # create function - by pressing -s- coordinates are saved. s from shift
    @slice_viewer.bind_key('s')
    def move_on(slice_viewer):
        slice_points = slice_viewer.layers['slice_points'].data[-1:]
        slice_saved_points = slice_viewer.layers['slice_saved_points'].data
        slice_points = np.rint(slice_points)
        slice_saved_points = np.rint(slice_saved_points)
        slice_saved_points = np.vstack((slice_saved_points, slice_points))
        slice_viewer.layers['slice_points'].data = []
        slice_viewer.layers['slice_saved_points'].data = slice_saved_points

        slice_id = int(slice_points[0, 0])
        print('Slice ', slice_id, 'marked')
        position = positions[slice_id]
        slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, position)
        shifts = np.array([slice_points[0, 1] - center_x, slice_points[0, 2] - center_y])
        np.savetxt(slice_transformations_path + 'xy_shift.txt', shifts, fmt='%i')
        np.save(slice_transformations_path + 'xy_shift.npy', shifts)

    # napari mark slices that need to be rotated 180 degrees
    @slice_viewer.bind_key('r')
    def move_on(slice_viewer):
        slice_points = slice_viewer.layers['slice_points'].data[-1:]
        slice_saved_points = slice_viewer.layers['slice_saved_points'].data
        slice_points = np.rint(slice_points)
        slice_saved_points = np.rint(slice_saved_points)
        slice_saved_points = np.vstack((slice_saved_points, slice_points))
        slice_viewer.layers['slice_points'].data = []
        slice_viewer.layers['slice_saved_points'].data = slice_saved_points

        slice_id = int(slice_points[0, 0])
        print('Slice ', slice_id, 'marked for rotation')
        position = positions[slice_id]
        slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, position)
        rotation = np.array([1])
        np.savetxt(slice_transformations_path + 'rotation_180.txt', rotation, fmt='%i')
        np.save(slice_transformations_path + 'rotation_180.npy', rotation)

    napari.run()


    # Apply xy shifts to align slices and crop. we don't need the entire slice, so crop around center
    # Chosen a conservative dimension, should probably take this out and make it into an argument at some point
    cropped_dim_x, cropped_dim_y = args.cropped_dimension_pixels, args.cropped_dimension_pixels
    cropped_bv_slices_stack = np.zeros(shape=(len(slices_names), cropped_dim_x, cropped_dim_y), dtype=preseq_inj_slices_stack.dtype)
    cropped_inj_slices_stack = np.zeros(shape=(len(slices_names), cropped_dim_x, cropped_dim_y), dtype=preseq_inj_slices_stack.dtype)
    cropped_somas_slices_stack = np.zeros(shape=(len(slices_names), cropped_dim_x, cropped_dim_y), dtype=preseq_inj_slices_stack.dtype)

    # check to see if there is a folder args.slice_transformations_path/slice. create a list with slices that have been marked. Load config file and add list of marked slices to config file and then save it.
    marked_slices = []
    for slice in positions:
        if os.path.exists(args.proc_transf_macro_path + slice):
            marked_slices.append(slice)

    config = configparser.ConfigParser()
    config.read(args.config_file)
    config['DATASET_SPECIFIC']['positions'] = str(marked_slices)
    with open(args.config_file, 'w') as configfile:
        config.write(configfile)
    args.positions = config['DATASET_SPECIFIC']['positions']

    print('Applying xy shifts to align slices and crop')
    for slice_id, slice in enumerate(positions):
        if slice in marked_slices:
            slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, slice)
            shifts = np.load(slice_transformations_path + 'xy_shift.npy')
            #print('Shifting and cropping slice', slice)
            transformation_matrix = np.array([[1, 0, shifts[1]], [0, 1, shifts[0]]])
            preseq_inj_slices_stack[slice_id] = cv2.warpAffine(preseq_inj_slices_stack[slice_id], transformation_matrix,
                                                               (sample_slice.shape[0], sample_slice.shape[1]),
                                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            preseq_bv_slices_stack[slice_id] = cv2.warpAffine(preseq_bv_slices_stack[slice_id], transformation_matrix,
                                                              (sample_slice.shape[0], sample_slice.shape[1]),
                                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            somas_slices_stack[slice_id] = cv2.warpAffine(somas_slices_stack[slice_id], transformation_matrix, (sample_slice.shape[0], sample_slice.shape[1]),
                                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        cropped_inj_slices_stack[slice_id] = helpers.crop_center_image(preseq_inj_slices_stack[slice_id], cropped_dim_x, cropped_dim_y)
        cropped_bv_slices_stack[slice_id] = helpers.crop_center_image(preseq_bv_slices_stack[slice_id], cropped_dim_x, cropped_dim_y)
        cropped_somas_slices_stack[slice_id] = helpers.crop_center_image(somas_slices_stack[slice_id], cropped_dim_x, cropped_dim_y)

    del preseq_inj_slices_stack, preseq_bv_slices_stack, somas_slices_stack

    # apply rotation if needed. for some datasets there's 180 degree rotations needed every 4 slices.
    print('Applying 180 degree rotations')
    for slice_id, slice in enumerate(positions):
        if slice in marked_slices:
            #if there is a folder rotation_180.txt in transformations folder, apply rotation
            slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, slice)
            if os.path.exists(slice_transformations_path + 'rotation_180.txt'):
                #print('Rotating image ', slice)
                cropped_inj_slices_stack[slice_id] = cv2.rotate(cropped_inj_slices_stack[slice_id], cv2.ROTATE_180)
                cropped_bv_slices_stack[slice_id] = cv2.rotate(cropped_bv_slices_stack[slice_id], cv2.ROTATE_180)
                cropped_somas_slices_stack[slice_id] = cv2.rotate(cropped_somas_slices_stack[slice_id], cv2.ROTATE_180)

    # rotate to match xy rotation based on manual input. sorry
    print('Rotating to match xy rotation based on manual input')
    if np.array([args.slices_xy_rotation]) != 0:
        for slice_id, slice in enumerate(positions):
            if slice in marked_slices:
                slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, slice)
                np.savetxt(slice_transformations_path + 'macro_rotation.txt', np.array([args.slices_xy_rotation]), fmt='%i')
                np.save(slice_transformations_path + 'macro_rotation.npy', args.slices_xy_rotation)
                print('Rotating slice', slice)
                cropped_bv_slices_stack[slice_id] = scipy.ndimage.rotate(cropped_bv_slices_stack[slice_id], angle=args.slices_xy_rotation, reshape=False)
                cropped_inj_slices_stack[slice_id] = scipy.ndimage.rotate(cropped_inj_slices_stack[slice_id], angle=args.slices_xy_rotation, reshape=False)
                cropped_somas_slices_stack[slice_id] = scipy.ndimage.rotate(cropped_somas_slices_stack[slice_id], angle=args.slices_xy_rotation, reshape=False)

    tif.imwrite(args.proc_slices_path + args.proc_bv_slices_name, cropped_bv_slices_stack)
    tif.imwrite(args.proc_slices_path + args.proc_inj_slices_name, cropped_inj_slices_stack)
    tif.imwrite(args.proc_slices_path + args.proc_somas_slices_name, cropped_somas_slices_stack)

    generate_metadata(args)

def preprocess_stack(args):
    '''
    Takes structural stack and divides it into GFP and RFP - somas and blood vessels channels. Then napari window will ask you to select 3 common points
    to resolve macro rotation problem between slices and stack.
    '''
    size_of_points = 18
    save_last_n_points = 3

    # load raw stack. if odd number of planes lose the last plane
    raw_stack = tif.imread(args.stack_path + args.stack_name)
    if raw_stack.shape[0] % 2 == 1:
        raw_stack = raw_stack[:-1]
    no_planes = raw_stack.shape[0]
    sacrificial_array = np.zeros_like(raw_stack)
    gfp_stack, blood_vessels_stack = np.split(sacrificial_array, 2)

    # save blood vessels and somas stacks
    for plane_id in range(int(no_planes / 2)):
        gfp_stack[plane_id] = raw_stack[plane_id * 2]
        blood_vessels_stack[plane_id] = raw_stack[plane_id * 2 + 1]
    tif.imwrite(args.stack_path + args.gfp_stack_name, gfp_stack)
    tif.imwrite(args.stack_path + args.bv_stack_name, blood_vessels_stack)

    # load previosuly created slices
    somas_slices = tif.imread(args.proc_slices_path + args.proc_somas_slices_name)
    bv_slices = tif.imread(args.proc_slices_path + args.proc_bv_slices_name)

    # select 3 common points between slices and stack and then press q
    slice_size_of_points = 0.018 * somas_slices.shape[-1]
    slice_viewer = napari.view_image(bv_slices, title='In vitro slices - select 3 common points and press q', name='somas_stack')
    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    slice_viewer.add_points(np.empty((0, 3)), face_color='green', size=slice_size_of_points, name='slice_saved_points', ndim=3)
    slice_viewer.add_points(face_color="blue", size=slice_size_of_points, name='slice_points', ndim=3)
    slice_viewer.layers['slice_points'].mode = 'add'

    stack_viewer = napari.view_image(blood_vessels_stack, title='In vivo stack - select 3 common points and press q', name='invivo_stack')
    stack_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    stack_viewer.add_points(np.empty((0, 3)), face_color="green", size=size_of_points, name='stack_saved_points', ndim=3)
    stack_viewer.add_points(face_color="blue", size=size_of_points, name='stack_points', ndim=3)
    stack_viewer.layers['stack_points'].mode = 'add'

    # upon pressing q common points are saved

    @stack_viewer.bind_key('q')
    def move_on(stack_viewer):
        slice_points = slice_viewer.layers['slice_points'].data[-save_last_n_points:]
        slice_saved_points = slice_viewer.layers['slice_saved_points'].data
        slice_points = np.rint(slice_points)
        slice_saved_points = np.rint(slice_saved_points)
        slice_saved_points = np.vstack((slice_saved_points, slice_points))
        slice_viewer.layers['slice_points'].data = []
        slice_viewer.layers['slice_saved_points'].data = slice_saved_points

        stack_points = stack_viewer.layers['stack_points'].data[-save_last_n_points:]
        stack_saved_points = stack_viewer.layers['stack_saved_points'].data
        stack_points = np.rint(stack_points)
        stack_saved_points = np.rint(stack_saved_points)
        stack_saved_points = np.vstack((stack_saved_points, stack_points))
        stack_viewer.layers['stack_points'].data = []
        stack_viewer.layers['stack_saved_points'].data = stack_saved_points

        # point coordinates are xyz, whereas in 3d image it's zxy
        slice_points[:, [0, 2]] = slice_points[:, [2, 0]]
        stack_points[:, [0, 2]] = stack_points[:, [2, 0]]

        np.savetxt(args.stack_landmarks_path + 'slice_points.txt', slice_points, fmt='%i')
        np.save(args.stack_landmarks_path + 'slice_points.npy', slice_points)
        np.savetxt(args.stack_landmarks_path + 'stack_points.txt', stack_points, fmt='%i')
        np.save(args.stack_landmarks_path + 'stack_points.npy', stack_points)
    # napari gui does not show up
    stack_viewer.show(block=True)
    slice_viewer.show(block=True)
    #napari.run(force=True)

    slice_points = np.load(args.stack_landmarks_path + 'slice_points.npy')
    stack_points = np.load(args.stack_landmarks_path + 'stack_points.npy')

    # compute rotation between stack and slices using the 3 pairs of points
    x_rot, y_rot, z_rot = helpers.find_stack_rotation(stack_points, slice_points)
    print('x y z rotations:', x_rot, y_rot, z_rot)

    # apply rotation to both blood vessel and somas stack
    rotated_gfp_stack_x = scipy.ndimage.interpolation.rotate(gfp_stack, angle=-x_rot, axes=(0, 1), reshape=True)
    rotated_gfp_stack_xy = scipy.ndimage.interpolation.rotate(rotated_gfp_stack_x, angle=y_rot, axes=(0, 2), reshape=True)
    tif.imwrite(args.matching_images_path + args.proc_gfp_stack_name, rotated_gfp_stack_xy)
    del rotated_gfp_stack_x, rotated_gfp_stack_xy, gfp_stack

    rotated_blood_vessels_stack_x = scipy.ndimage.interpolation.rotate(blood_vessels_stack, angle=-x_rot, axes=(0, 1), reshape=True)
    rotated_blood_vessels_stack_xy = scipy.ndimage.interpolation.rotate(rotated_blood_vessels_stack_x, angle=y_rot, axes=(0, 2), reshape=True)
    tif.imwrite(args.matching_images_path + args.proc_bv_stack_name, rotated_blood_vessels_stack_xy)

    # load rotated stack back into napari to check
    slice_viewer = napari.view_image(bv_slices, title='In vitro slices - check', name='bv_stack')
    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)

    stack_viewer = napari.view_image(rotated_blood_vessels_stack_xy, title='In vivo stack - check', name='invivo_stack')
    stack_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)

    napari.run()

def preprocess_scan(args):
    '''
    Preprocess point clouds that are the centroids for functional and structural stack

    '''

    print('Preprocessing scan...')

    func_centroids_pd = pd.read_csv(args.scan_path + args.func_centroids_file)
    struct_centroids_pd = pd.read_csv(args.scan_path + args.struct_centroids_file)
    print('no func and struct centroids before selection', len(func_centroids_pd), len(struct_centroids_pd))
    struct_centroids_pd = struct_centroids_pd[struct_centroids_pd['stack_idx'] == args.stack_id]
    func_centroids_pd = func_centroids_pd[func_centroids_pd['stack_idx'] == args.stack_id]
    #reshuffle such that duplicates on 'sunit_id' are next to each other
    func_centroids_pd = func_centroids_pd.sort_values(by=['sunit_id'])
    #func_centroids_pd = func_centroids_pd.drop_duplicates(subset=['sunit_id'])
    print('no func and struct centroids after selection', len(func_centroids_pd), len(struct_centroids_pd))

    gfp_stack = tif.imread(args.stack_path + args.gfp_stack_name)
    rotated_gfp_stack = tif.imread(args.proc_stack_path + args.proc_gfp_stack_name)

    planes, height, width = gfp_stack.shape

    offset_um = [args.xy_stack_offsets[0] - 0.5 * width, args.xy_stack_offsets[1] - 0.5 * height]

    func_centroids_pd['is_fun'] = 1

    struct_centroids_pd = pd.merge(struct_centroids_pd, func_centroids_pd[['sunit_id', 'is_fun']], on='sunit_id', how='left')
    print('no func and struct centroids after merge', len(func_centroids_pd), len(struct_centroids_pd))

    struct_centroids_pd['is_fun'] = struct_centroids_pd['is_fun'].fillna(0)

    struct_centroids_pd = struct_centroids_pd[['sunit_x', 'sunit_y', 'sunit_z', 'sunit_id', 'is_fun']]
    struct_centroids = struct_centroids_pd.to_numpy()
    struct_centroids[:, 2] *= 0.5
    struct_centroids[:, 0:2] = struct_centroids[:, 0:2] - offset_um
    struct_centroids = np.rint(struct_centroids)

    func_centroids_pd = func_centroids_pd[['sunit_x', 'sunit_y', 'sunit_z', 'sunit_id', 'field', 'depth', 'is_fun', 'scan_idx']]
    func_centroids = func_centroids_pd.to_numpy()
    func_centroids[:, 2] *= 0.5
    func_centroids[:, 5] *= 0.25
    func_centroids[:, 0:2] = func_centroids[:, 0:2] - offset_um
    func_centroids = np.rint(func_centroids)

    np.savetxt(args.scan_path + args.func_centroids_name + '.txt', func_centroids, fmt='%i')
    np.save(args.scan_path + args.func_centroids_name + '.npy', func_centroids)
    np.savetxt(args.scan_path + args.struct_centroids_name + '.txt', struct_centroids, fmt='%i')
    np.savetxt(args.scan_path + args.struct_centroids_name + '.csv', struct_centroids)
    np.save(args.scan_path + args.struct_centroids_name + '.npy', struct_centroids)

    slice_points = np.load(args.stack_landmarks_path + 'slice_points.npy')
    stack_points = np.load(args.stack_landmarks_path + 'stack_points.npy')

    struct_ids = struct_centroids[:, 3]
    struct_ids = struct_ids.reshape((-1, 1))
    func_ids = func_centroids[:, 3]
    func_ids = func_ids.reshape((-1, 1))
    scan_idx = func_centroids[:, 7]
    scan_idx = scan_idx.reshape((-1, 1))

    func_centroids = func_centroids[:, 0:3]
    struct_centroids = struct_centroids[:, 0:3]

    # apply rotation of the previosuly identified pairs of points
    x_rot, y_rot, z_rot = helpers.find_stack_rotation(stack_points, slice_points)
    print('x y z rotations:', x_rot, y_rot, z_rot)
    rotation_matrix = scipy.spatial.transform.Rotation.from_euler('XY', (x_rot, y_rot), degrees=True)

    og_center = np.array(gfp_stack.shape) * 0.5
    og_center[0], og_center[2] = og_center[2], og_center[0]
    new_center = np.array(rotated_gfp_stack.shape) * 0.5
    new_center[0], new_center[2] = new_center[2], new_center[0]

    del rotated_gfp_stack

    rotated_func_centroids = func_centroids[:] - og_center
    rotated_func_centroids = rotation_matrix.apply(rotated_func_centroids)
    rotated_func_centroids = rotated_func_centroids[:] + new_center
    rotated_struct_centroids = struct_centroids[:] - og_center
    rotated_struct_centroids = rotation_matrix.apply(rotated_struct_centroids)
    rotated_struct_centroids = rotated_struct_centroids[:] + new_center

    rotated_struct_centroids = np.hstack((rotated_struct_centroids, struct_ids))
    rotated_func_centroids = np.hstack((rotated_func_centroids, func_ids, scan_idx))

    print('no rotated func and struct centroids', len(rotated_func_centroids), len(rotated_struct_centroids))

    np.save(args.proc_scan_path + args.rotated_func_centroids_name + '.npy', rotated_func_centroids)
    np.savetxt(args.proc_scan_path + args.rotated_func_centroids_name + '.txt', rotated_func_centroids, fmt='%i')
    np.save(args.proc_scan_path + args.rotated_struct_centroids_name + '.npy', rotated_struct_centroids)
    np.savetxt(args.proc_scan_path + args.rotated_struct_centroids_name + '.txt', rotated_struct_centroids, fmt='%i')

    padded_struct_centroids_stack = np.empty(shape=(0, 4))
    for centroid in rotated_struct_centroids:
        zpos = centroid[2]
        for z in range(-2, 3):
            to_add = np.array([zpos + z, centroid[0], centroid[1], centroid[3]])
            padded_struct_centroids_stack = np.vstack((padded_struct_centroids_stack, to_add))

    np.savetxt(args.proc_scan_path + 'padded_struct_centroids.txt', padded_struct_centroids_stack, fmt='%i')
    np.save(args.proc_scan_path + 'padded_struct_centroids.npy', padded_struct_centroids_stack)
    np.savetxt(args.matching_centroids_path + 'padded_struct_centroids.txt', padded_struct_centroids_stack, fmt='%i')
    np.save(args.matching_centroids_path + 'padded_struct_centroids.npy', padded_struct_centroids_stack)

    padded_func_centroids_stack = np.empty(shape=(0, 5))
    for centroid in rotated_func_centroids:
        zpos = centroid[2]
        for z in range(-2, 3):
            to_add = np.array([zpos + z, centroid[0], centroid[1], centroid[3], centroid[4]])
            padded_func_centroids_stack = np.vstack((padded_func_centroids_stack, to_add))

    np.savetxt(args.proc_scan_path + 'padded_func_centroids.txt', padded_func_centroids_stack, fmt='%i')
    np.save(args.proc_scan_path + 'padded_func_centroids.npy', padded_func_centroids_stack)
    np.savetxt(args.matching_centroids_path + 'padded_func_centroids.txt', padded_func_centroids_stack, fmt='%i')
    np.save(args.matching_centroids_path + 'padded_func_centroids.npy', padded_func_centroids_stack)

    print('nr padded func and struct centroids', len(padded_func_centroids_stack), len(padded_struct_centroids_stack))

    # mark location of those centroids to see if functional and structural points overlap
    color = 50;
    thickness = 1;
    size = 12;
    for index, point in enumerate(func_centroids):
        plane = round(point[2])
        center = (round(point[0]), round(point[1]))
        for z in range(-5, 5):
            try:
                cv2.circle(gfp_stack[plane - 1 + z], center, size - abs(z), color, thickness)
            except IndexError:
                pass

    marked_gfp_stack = gfp_stack

    color = 70;
    thickness = 1;
    size = 10;
    for index, point in enumerate(struct_centroids):
        plane = round(point[2])
        center = (round(point[0]), round(point[1]))
        for z in range(-5, 5):
            try:
                cv2.circle(gfp_stack[plane - 1 + z], center, size - abs(z), color, thickness)
            except IndexError:
                pass

    gfp_stack = gfp_stack.astype(np.int16)
    tif.imwrite(args.proc_checks_scan_path + 'check_fun_struct_match_' + args.proc_gfp_stack_name, gfp_stack)

    rotated_marked_x = scipy.ndimage.interpolation.rotate(marked_gfp_stack, angle=-x_rot, axes=(0, 1), reshape=True)
    rotated_marked_xy = scipy.ndimage.interpolation.rotate(rotated_marked_x, angle=y_rot, axes=(0, 2), reshape=True)

    # mark and see if rotating the point clouds separately matches the rotation of the 3D stack
    color = 50;
    thickness = 1;
    size = 18;
    for id, point in enumerate(rotated_func_centroids):
        plane = round(point[2])
        center = (round(point[0]), round(point[1]))
        for z in range(-5, 5):
            try:
                cv2.circle(rotated_marked_xy[plane - 1 + z], center, size - abs(z), color, thickness)
            except IndexError:
                pass
    rotated_marked_xy = rotated_marked_xy.astype(np.int16)
    tif.imwrite(args.proc_checks_scan_path + 'check_rot_' + args.proc_gfp_stack_name, rotated_marked_xy)

def mark_barcodes_areas(args):

    bcseq_cycles = helpers.list_files(args.slices_path)
    bcseq_cycles = [bcseq_cycle for bcseq_cycle in bcseq_cycles if args.bcseq_name in bcseq_cycle]
    bcseq_cycle = helpers.human_sort(bcseq_cycles)[-1]
    print('Marking infected neurons on bcseq cycle', bcseq_cycle)

    #check to see if bcseq has been stitched
    if not os.path.exists(args.slices_path + 'stitched_barcodes'):
        bcseq_path = helpers.quick_dir(args.slices_path, bcseq_cycle)
        bcseq_path = helpers.quick_dir(bcseq_path, 'maxproj')
        stitched_bcseq_path = stitch_images_imperfectly_folder(bcseq_path, output_folder_name='../stitched_barcodes')
        stitched_bcseq_path = flip_horizontally(stitched_bcseq_path, overwrite_images=True)

    else:
        stitched_bcseq_path = helpers.quick_dir(args.slices_path, 'stitched_barcodes')

    if len(args.positions) == 0:
        positions = helpers.list_files(stitched_bcseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    # perform all the transformations like on the rest of the slices
    sample_slice = tif.imread(stitched_bcseq_path + positions[0] + '.tif')
    cropped_dim_x, cropped_dim_y = args.cropped_dimension_pixels, args.cropped_dimension_pixels
    barcodes_stack = np.ndarray((len(positions), sample_slice.shape[1], sample_slice.shape[2]), dtype=np.int16)
    cropped_barcodes_stack = np.zeros(shape=(len(positions), cropped_dim_x, cropped_dim_y), dtype=np.int16)

    for slice_id, slice in enumerate(positions):
        slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, slice)
        shifts = np.load(slice_transformations_path + 'xy_shift.npy')
        #print('Shifting and cropping slice', slice)
        transformation_matrix = np.array([[1, 0, shifts[1]], [0, 1, shifts[0]]])
        barcodes_image = tif.imread(stitched_bcseq_path + slice + '.tif')
        barcodes_image = np.max(barcodes_image[0:4], axis=0)
        barcodes_image = cv2.warpAffine(barcodes_image, transformation_matrix,(sample_slice.shape[1], sample_slice.shape[2]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        cropped_barcodes_stack[slice_id] = helpers.crop_center_image(barcodes_image, cropped_dim_x, cropped_dim_y)

    print('Applying 180 degree rotations')
    for slice_id, slice in enumerate(positions):
        #if there is a folder rotation_180.txt in transformations folder, apply rotation
        slice_transformations_path = helpers.quick_dir(args.proc_transf_macro_path, slice)
        if os.path.exists(slice_transformations_path + 'rotation_180.txt'):
            #print('Rotating image ', slice)
            cropped_barcodes_stack[slice_id] = cv2.rotate(cropped_barcodes_stack[slice_id], cv2.ROTATE_180)

    # rotate to match xy rotation based on manual input. sorry
    print('Rotating to match xy rotation based on manual input')
    if np.array([args.slices_xy_rotation]) != 0:
        for slice_id, slice in enumerate(positions):
            print('Rotating slice', slice)
            cropped_barcodes_stack[slice_id] = scipy.ndimage.rotate(cropped_barcodes_stack[slice_id], angle=args.slices_xy_rotation, reshape=False)
    tif.imwrite(args.proc_slices_path + 'barcodes.tif', cropped_barcodes_stack)


    for positions_id, position in enumerate(positions):
        print(position)
        coordinates_position_path = helpers.quick_dir(args.proc_coordinates_path, position)

        #bcseq_image = tif.imread(stitched_bcseq_path + position + '.tif')
        #bcseq_image = np.max(bcseq_image[0:4], axis=0)

        slice_viewer = napari.view_image(cropped_barcodes_stack[positions_id], title='In vitro slice - mark barcoded area and press q')

        slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
        slice_viewer.add_shapes(shape_type='rectangle', name='rectangle', edge_width=5, edge_color='coral', face_color='royalblue')
        @slice_viewer.bind_key('q')
        def move_on(slice_viewer):
            corners = slice_viewer.layers['rectangle'].data
            corners = np.array(corners[0])
            np.savetxt(coordinates_position_path + 'barcodes_area.txt', corners, fmt='%i')
        slice_viewer.show(block=True)

    marked_slices = []
    for position in positions:
        coordinates_position_path = helpers.quick_dir(args.proc_coordinates_path, position)
        if os.path.exists(coordinates_position_path + 'barcodes_area.txt'):
            marked_slices.append(position)

    config = configparser.ConfigParser()
    config.read(args.config_file)
    config['DATASET_SPECIFIC']['funseq_positions'] = str(marked_slices)
    with open(args.config_file, 'w') as configfile:
        config.write(configfile)

def generate_metadata(args):
    '''
    Create a function that generates metadata from the config file. This metadata should be a pandas file, where each argument
    in the config file is a column header in the pandas file.
    First read in the config file, then create a pandas file with the same name as the config file, but with a .csv extension.

    '''

    config = configparser.ConfigParser()
    config.read(args.config_file)
    config_dict = {}
    for section in config.sections():
        for key, value in config.items(section):
            config_dict[key] = value

    metadata = pd.DataFrame(config_dict, index=[0])
    metadata.to_csv(args.dataset_path + 'metadata.csv', index=False)


def arrange_by_pos(args):
    '''
    Arranges images by position. This means for each position all cycles will be in the same folder
    :param local_bool:
    :param local_path:
    :param channels_order:
    :param remote_path:
    :param server:
    :param username:
    :param password:
    :return:
    '''

    print('Arranging images in local folder by position')

    #Go through all sets of sequencing - geneseq, barcode seq and hyb seq
    seq_types = [args.geneseq_name, args.bcseq_name, args.hybseq_name, args.preseq_name]
    #seq_types = [args.hybseq_name]
    seq_types = [i for i in seq_types if i is not None]

    for seq_type in seq_types:
        seq_folders = helpers.list_files(args.slices_path)
        seq_type_folders = helpers.remove_outstringers(seq_folders, seq_type)
        #keep folder names that contain substring seq
        seq_type_folders = sorted(seq_type_folders, key=lambda x: x[-1])


        if len(seq_type_folders) > 0:
            all_tiles = helpers.list_files(args.slices_path + seq_type_folders[0] + '/maxproj/')
            positions_dict = helpers.sort_position_folders(all_tiles)
            original_output_path = helpers.quick_dir(args.proc_original_path, seq_type)
            cycle_path = args.slices_path + seq_type_folders[0] + '/maxproj/'
            print('Checking if darkfield and flatfield images are already created for ' + seq_type)
            illumination_path = helpers.quick_dir(args.proc_illumination_path, seq_type)
            if not os.path.exists(illumination_path + 'darkfield.tif'):
                print('Creating darkfield and flatfield images for ' + seq_type)
                flatfield, darkfield = helpers.create_illumination_profiles(cycle_path, nr_images=100)
                tif.imwrite(illumination_path + 'darkfield.tif', darkfield)
                tif.imwrite(illumination_path + 'flatfield.tif', flatfield)


            if len(args.positions) == 0:
                positions = list(positions_dict.keys())
            else:
                positions = args.positions
            print('Positions dict: ', positions_dict.keys())
            print('Positions to be processed: ', positions)

        #move all files to 'original' folder and for each position add all cycles
            for cycle in seq_type_folders:
                cycle_path = args.slices_path + cycle + '/maxproj/'
                new_cycle_path = helpers.quick_dir(original_output_path, cycle)
                for position in positions:
                    new_pos_path = helpers.quick_dir(new_cycle_path, position)
                    for tile in positions_dict[position]:
                        shutil.copy(cycle_path + tile, new_pos_path + tile)

def preprocess_images(args):

    sequencing_folder_paths = [args.proc_original_geneseq_path, args.proc_original_bcseq_path, args.proc_original_hybseq_path, args.proc_original_preseq_path]
    checks_illumination_paths = [args.proc_checks_illumination_geneseq_path, args.proc_checks_illumination_bcseq_path, args.proc_checks_illumination_hybseq_path, args.proc_checks_illumination_preseq_path]
    soma_channels = [args.geneseq_soma_channels, args.bcseq_soma_channels, args.hybseq_soma_channels, args.preseq_soma_channels]
    color_bleeding_bools = [True, True, False, False]
#    sequencing_folder_paths = [args.proc_original_hybseq_path]
#    checks_illumination_paths = [args.proc_checks_illumination_hybseq_path]
#    soma_channels = [args.hybseq_soma_channels]
#    color_bleeding_bools = [False]


    #get a list of all positions to randomly select a few for checks
    cycles = helpers.list_files(sequencing_folder_paths[0])
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_folder_paths[0] + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    check_positions = random.sample(positions, min(args.max_checks, len(positions)))
    print('Correcting for uneven illumination and colorbleed for all positions')
    for (sequencing_folder_path, checks_illumination_path, soma_chan, color_bleeding_bool) \
            in zip(sequencing_folder_paths, checks_illumination_paths, soma_channels, color_bleeding_bools):
        cycles = helpers.list_files(sequencing_folder_path)
        cycles = helpers.human_sort(cycles)
        if len(cycles) > 0:
            print('Preprocessing sequencing folder ', sequencing_folder_path)

            # load darkfield and flatfield images for illumination correction
            illumination_path = helpers.quick_dir(args.proc_illumination_path, sequencing_folder_path.split('/')[-2])
            darkfield = tif.imread(illumination_path + 'darkfield.tif')
            flatfield = tif.imread(illumination_path + 'flatfield.tif')
            darkfield = darkfield.astype(np.int16)


            no_channels = soma_chan[1]

            for cycle_id, cycle in enumerate(cycles):
                print('Preprocessing cycle', cycle)
                pos_tic = time.perf_counter()
                cycle_path = helpers.quick_dir(sequencing_folder_path, cycle)

                for position in positions:
                    position_path = helpers.quick_dir(cycle_path, position)

                    tiles = helpers.list_files(position_path)

                    if position in check_positions:
                        color_corr_checks_position_path = helpers.quick_dir(args.proc_checks_color_correction_path, position)
                        export_checks_bool = True
                    export_checks_bool = False

                    zero_array = np.zeros_like(darkfield[0])
                    for tile in tiles:
                        tile_image = tif.imread(position_path + tile)
                        tile_image = tile_image[:no_channels]
                        tile_image = tile_image.astype(np.int16)

                        for chan in range(tile_image.shape[0]):
                            tile_image[chan] = tile_image[chan] - darkfield[chan]
                            tile_image[chan] = np.maximum(tile_image[chan], zero_array)
                            tile_image[chan] = tile_image[chan] / flatfield[chan]

                        #color bleed correction not needed for hybseq and preseq
                        if color_bleeding_bool is True:
                            tile_image = np.expand_dims(tile_image, axis=(0,2))
                            colormixing_matrix = np.array([
                                [1, 0.02, 0, 0],
                                [0.9, 1, 0, 0],
                                [0, 0, 1, 0.99],
                                [0, 0, 0, 1]])

                            fix = np.linalg.inv(colormixing_matrix)
                            tile_image_ColorCorr = np.clip(np.einsum('rcxyz,cd->rdxyz', tile_image, fix), 0, None)


                            if export_checks_bool is True:
                                bardensr.preprocessing.colorbleed_plot(tile_image_ColorCorr[0, 0], tile_image_ColorCorr[0, 1])
                                plt.savefig(color_corr_checks_position_path + cycle + tile +'_after_colorCorr0_1.jpg')
                                plt.clf()
                                bardensr.preprocessing.colorbleed_plot(tile_image[0, 0], tile_image[0, 1])
                                plt.savefig(color_corr_checks_position_path + cycle + tile + '_before_colorCorr0_1.jpg')

                                plt.clf()
                                bardensr.preprocessing.colorbleed_plot(tile_image_ColorCorr[0, 2], tile_image_ColorCorr[0, 3])
                                plt.savefig(color_corr_checks_position_path + cycle + tile + '_after_colorCorr2_3.jpg')
                                plt.clf()
                                bardensr.preprocessing.colorbleed_plot(tile_image[0, 2], tile_image[0, 3])
                                plt.savefig(color_corr_checks_position_path + cycle + tile + '_before_colorCorr2_3.jpg')

                            tile_image_ColorCorr = tile_image_ColorCorr.astype(np.int16)
                            #tif.imwrite(position_output_path + tile, tile_image_ColorCorr)
                            tile_image_ColorCorr = np.squeeze(tile_image_ColorCorr)
                            tile_image = tile_image_ColorCorr
                        tif.imwrite(position_path + tile, tile_image)
                if cycle_id == 0:
                    stitched_path = stitch_images_imperfectly_folder(position_path, output_path=checks_illumination_path)

                pos_toc = time.perf_counter()
                print('Cycle preprocessed in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def manual_align_hybseq_cycle(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 5
    seq_path = args.proc_original_hybseq_path
    aligned_path = args.proc_aligned_hybseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif',
                      '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)
    # cycles = ['bcseq_10']

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    # Align bcseq cycles
    print('Scaling hybseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        # first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_hybseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for relevant_tile in relevant_tiles:
                    reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    # reference = tif.imread(args.proc_aligned_preseq_path + 'preseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    # reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_10' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    reference = np.max(reference[0:2], axis=0)
                    # reference = reference[0]
                    # reference = np.median(reference[2:4], axis=0)
                    # reference = np.min(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    to_align_mp = np.max(to_align[1:4], axis=0)
                    # to_align_mp = to_align[args.somas_chan]
                    # to_align_mp = to_align[3]
                    # to_align_mp = np.median(to_align[2:4], axis=0)
                    # to_align_mp = np.min(to_align[0:4], axis=0)

                    overlap = helpers.check_images_overlap(to_align_mp, reference, save_output=False)

                    slice_viewer = napari.view_image(overlap, title='mark common features')
                    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
                    slice_viewer.add_points(face_color="blue", size=30, name='slice_points', ndim=3)
                    slice_viewer.layers['slice_points'].mode = 'add'

                    @slice_viewer.bind_key('q')
                    def move_on(slice_viewer):

                        slice_points = slice_viewer.layers['slice_points'].data[-2:]
                        slice_points = np.rint(slice_points)

                        yshift = int(slice_points[1, 1] - slice_points[0, 1])
                        xshift = int(slice_points[1, 2] - slice_points[0, 2])
                        print('xshift', xshift, 'yshift', yshift)
                        transformation_matrix = np.array([[1, 0, -xshift], [0, 1, -yshift]])

                        np.savetxt(check_alignment_path + 'TM.txt', transformation_matrix)

                    slice_viewer.show(block=True)

                    transformation_matrix = np.loadtxt(check_alignment_path + 'TM.txt')
                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    overlap = helpers.check_images_overlap(aligned, reference, save_output=False)
                    tif.imwrite(check_alignment_path + 'zoverlap.tif', overlap)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0, 0] = 1
                        transformation_matrix[0, 1] = 0
                        transformation_matrix[1, 0] = 0
                        transformation_matrix[1, 1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        # aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        # tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    # to_align_mp = np.max(to_align[0:4], axis=0)
                    # _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    # transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    # transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    # transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    # transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    # print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop

            else:
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    # to_align_mp = np.max(to_align[0:4], axis=0)
                    # _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    # transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    # transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    # transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    # print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def scale_align_geneseq_cycles(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 4
    seq_path = args.proc_original_geneseq_path
    aligned_path = args.proc_aligned_geneseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif', '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Scaling geneseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        #first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_geneseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for tile_id, relevant_tile in enumerate(relevant_tiles):
                    reference = tif.imread(seq_path + 'geneseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    reference = np.max(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    to_align_mp = np.max(to_align[0:4], axis=0)

                    reference = helpers.scale_to_8bit(reference)
                    to_align_mp = helpers.scale_to_8bit(to_align_mp)

                    _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.ECC_reg(to_align_mp, reference)
                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Attempt ' + str(tile_id + 1) + ' of ' + str(len(relevant_tiles)) + '. Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0,0] = 1
                        transformation_matrix[0,1] = 0
                        transformation_matrix[1,0] = 0
                        transformation_matrix[1,1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        #aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        #tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    #transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def scale_align_hybseq_cycle(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 5
    seq_path = args.proc_original_hybseq_path
    aligned_path = args.proc_aligned_hybseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif', '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)
    #cycles = ['bcseq_10']

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Scaling hybseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        #first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_hybseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for relevant_tile in relevant_tiles:
                    reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = tif.imread(args.proc_aligned_preseq_path + 'preseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_10' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    reference = np.max(reference[0:2], axis=0)
                    #reference = reference[0]
                    #reference = np.median(reference[2:4], axis=0)
                    #reference = np.min(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    to_align_mp = np.max(to_align[1:4], axis=0)
                    #to_align_mp = to_align[args.somas_chan]
                    #to_align_mp = to_align[3]
                    #to_align_mp = np.median(to_align[2:4], axis=0)
                    #to_align_mp = np.min(to_align[0:4], axis=0)

                    reference = helpers.scale_to_8bit(reference)
                    to_align_mp = helpers.scale_to_8bit(to_align_mp)

                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    _, transformation_matrix = helpers.PhaseCorr_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.ECC_reg(to_align_mp, reference)
                    # transformation_matrix =  [[ 1.04202662e+00  5.69321065e-04 -7.08105834e+00]
                    #  [-5.69321065e-04  1.04202662e+00 -2.31072058e+01]]
                    #transformation_matrix = np.array([[1.04202662e+00, 5.69321065e-04, -7.08105834e+00],
                    #                                 [-5.69321065e-04, 1.04202662e+00, -2.31072058e+01]])
                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    overlap = helpers.check_images_overlap(aligned, reference, save_output=False)
                    tif.imwrite(check_alignment_path + 'zoverlap.tif', overlap)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0,0] = 1
                        transformation_matrix[0,1] = 0
                        transformation_matrix[1,0] = 0
                        transformation_matrix[1,1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        #aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        #tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    #transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop

            else:
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def scale_align_preseq_cycle(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 4
    seq_path = args.proc_original_preseq_path
    aligned_path = args.proc_aligned_preseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif', '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)
    #cycles = ['bcseq_10']

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Scaling geneseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        #first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_preseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for relevant_tile in relevant_tiles:
                    reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_10' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = np.max(reference[0:4], axis=0)
                    reference = reference[2]
                    #reference = np.median(reference[2:4], axis=0)
                    #reference = np.min(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    to_align_mp = to_align[0]
                    #to_align_mp = np.median(to_align[2:4], axis=0)
                    #to_align_mp = np.min(to_align[0:4], axis=0)

                    reference = helpers.scale_to_8bit(reference)
                    to_align_mp = helpers.scale_to_8bit(to_align_mp)

                    _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.PhaseCorr_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.ECC_reg(to_align_mp, reference)
                    #transformation_matrix = np.array([[1, 0, 10], [0, 1, -10]], dtype=np.float32)
                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0,0] = 1
                        transformation_matrix[0,1] = 0
                        transformation_matrix[1,0] = 0
                        transformation_matrix[1,1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        #aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        #tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    #transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def scale_align_bcseq_cycle(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 4
    seq_path = args.proc_original_bcseq_path
    aligned_path = args.proc_aligned_bcseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif', '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)
    cycles = ['bcseq_10']

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Scaling geneseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        #first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for relevant_tile in relevant_tiles:
                    reference = tif.imread(args.proc_aligned_preseq_path + 'preseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_10' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    #reference = np.max(reference[0:4], axis=0)
                    reference = reference[0]
                    #reference = np.median(reference[0:4], axis=0)
                    #reference = np.min(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    to_align_mp = to_align[2]
                    #to_align_mp = np.median(to_align[0:4], axis=0)
                    #to_align_mp = np.min(to_align[0:4], axis=0)

                    reference = helpers.scale_to_8bit(reference)
                    to_align_mp = helpers.scale_to_8bit(to_align_mp)

                    _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.ECC_reg(to_align_mp, reference)
                    #_, transformation_matrix = helpers.PhaseCorr_reg(to_align_mp, reference)
                    #transformation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0,0] = 1
                        transformation_matrix[0,1] = 0
                        transformation_matrix[1,0] = 0
                        transformation_matrix[1,1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        #aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        #tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])
                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    #transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def scale_align_all_bcseq_cycles(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    import prompter
    import sys
    print('Starting alignment of cycles')
    no_channels = 4
    seq_path = args.proc_original_bcseq_path
    aligned_path = args.proc_aligned_bcseq_path
    relevant_tiles = ['_001_001.tif', '_001_002.tif', '_001_000.tif', '_002_000.tif', '_000_002.tif', '_000_001.tif', '_002_001.tif', '_002_002.tif', '_000_000.tif']
    cycles = helpers.list_files(seq_path)
    cycles = helpers.human_sort(cycles)

    check_alignment_path = helpers.quick_dir(args.proc_path, 'check_alignment')

    if len(args.positions) == 0:
        positions = helpers.list_files(seq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Scaling geneseq cycles')
    for position in positions:
        print('Scaling position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

        #first copy all images to aligned folders and then overwrite those

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            cycle_path = helpers.quick_dir(seq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_geneseq_path, cycle)

            tiles = helpers.list_files(geneseq_position_path)
            if not os.path.exists(transformations_cycle_path + position + '/' + tiles[0] + '.txt'):
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                happy_with_alignment = False
                for relevant_tile in relevant_tiles:
                    #reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_10' + '/' + position + '/' + 'MAX_' + position + relevant_tile)
                    reference = np.max(reference[0:4], axis=0)

                    to_align = tif.imread(geneseq_position_path + '/' + 'MAX_' + position + relevant_tile)
                    to_align_mp = np.max(to_align[0:4], axis=0)

                    reference = helpers.scale_to_8bit(reference)
                    to_align_mp = helpers.scale_to_8bit(to_align_mp)

                    _, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    print('Transformation matrix for cycle, pos, tile', cycle, position, relevant_tile, 'is', transformation_matrix)
                    aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                    tif.imwrite(check_alignment_path + 'reference.tif', reference)
                    tif.imwrite(check_alignment_path + 'to_align.tif', to_align_mp)

                    # trigger a prompt where user has to select yes or no to change the value of bool happy_with_alignment
                    # if happy_with_alignment is True, then break out of the loop
                    # if happy_with_alignment is False, then continue with the loop
                    prompt_response = prompter.prompt('Are you happy with the alignment?')
                    if prompt_response == 'y':
                        happy_with_alignment = True
                        break
                    elif prompt_response == 'r':
                        happy_with_alignment = True
                        transformation_matrix[0,0] = 1
                        transformation_matrix[0,1] = 0
                        transformation_matrix[1,0] = 0
                        transformation_matrix[1,1] = 1
                        print('changing to rigid transform with matrix', transformation_matrix)
                        #aligned = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[1], to_align_mp.shape[0]),
                        #                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        #tif.imwrite(check_alignment_path + 'aligned.tif', aligned)
                        break

                    elif prompt_response == 'n':
                        continue

                if happy_with_alignment is False:
                    print('You are not happy with the alignment. Please check the images in the folder', check_alignment_path)
                    sys.exit()

                for tile_id, tile in enumerate(tiles):
                    print('Scaling tile', tile)

                    aligned_cycle_path = helpers.quick_dir(aligned_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)


                    #to_align_mp = np.max(to_align[0:4], axis=0)
                    #_, transformation_matrix = helpers.ORB_reg(to_align_mp, reference)
                    #transformation_matrix = [[ 1.04450721e+00 -2.54915848e-05 -1.42233741e+01] [ 2.54915848e-05  1.04450721e+00 -1.22292699e+01]]
                    #transformation_matrix = np.array([[1.04450721e+00, -2.54915848e-05, -1.42233741e+01], [2.54915848e-05, 1.04450721e+00, -1.22292699e+01]])

                    #transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    to_align = tif.imread(geneseq_position_path + '/' + tile)

                    #transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    #print('Transformation matrix for tile ' + tile + ' is ' + str(transformation_matrix))

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
                for tile in tiles:
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix)
                # break from for loop
        pos_toc = time.perf_counter()
        print('Position scaled in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def align_geneseq_cycles_parallel(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)
    #cycles = ['geneseq_6']


    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Aligning geneseq cycles')
    @ray.remote
    def align_position(position, cycles, no_channels, args):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_geneseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

            tiles = helpers.list_files(geneseq_position_path)
            for tile in tiles:
                shutil.copy(geneseq_position_path + tile, aligned_geneseq_position_path + tile)

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_geneseq_path, cycle)
            if not os.path.exists(transformations_cycle_path + position):
                xshifts = []
                yshifts = []

                # run a consensus strategy. go through all the tiles of the same position and take the most likely offsets.
                for tile in tiles:
                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
                    reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + tile)
                    reference = np.max(reference[0:4], axis=0)

                    to_align = tif.imread(aligned_geneseq_position_path + '/' + tile)
                    to_align_mp = np.max(to_align[0:4], axis=0)
                    # to_align_mp = np.mean(to_align[0:4], axis=0)

                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                    yshifts.append(transformation_matrix[0, 2])
                    xshifts.append(transformation_matrix[1, 2])

                print('shifts for cycle and position', cycle, position, xshifts, yshifts)
                x_shift = np.percentile(xshifts, 50)
                y_shift = np.percentile(yshifts, 50)
                print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)
                print('Aligning all tiles')
                for tile_id, tile in enumerate(tiles):
                    #print('Aligning tile', tile)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_geneseq_position_path + '/' + tile)

                    transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                    if abs(transformation_matrix[0, 2] - y_shift) > args.allowed_reg_shift:
                        print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                        transformation_matrix[0, 2] = y_shift

                    if abs(transformation_matrix[1, 2] - x_shift) > args.allowed_reg_shift:
                        print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                        transformation_matrix[1, 2] = x_shift

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
            else:
                print('Aligning all tiles')
                for tile_id, tile in enumerate(tiles):
                    #print('Aligning tile', tile)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
                    aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_geneseq_position_path + '/' + tile)

                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position, cycles, no_channels, args) for position in positions])
    ray.shutdown()

def align_hybseq_to_geneseq_parallel(args):
    print('Starting alignment of hybseq cycle')
    no_channels = 5
    #no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    @ray.remote
    def align_position(position):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'hybseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_hybseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
        hybseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_hybseq_path, cycle)

        tiles = helpers.list_files(hybseq_position_path)

        for tile in tiles:
            shutil.copy(hybseq_position_path + tile, aligned_hybseq_position_path + tile)

        # check if shifts have been calculated before. if not, do phase correlation now.
        # otherwise, load previously calculated transformation matrices
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
            xshifts = []
            yshifts = []
            for tile in tiles:
                reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + tile)
                reference = np.max(reference[0:4], axis=0)
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[0:4], axis=0)
                #to_align_mp = np.mean(to_align[0:4], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)
            print('Aligning all tiles')

            for tile_id, tile in enumerate(tiles):

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > args.allowed_reg_shift:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > args.allowed_reg_shift:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_hybseq_position_path + '/' + tile, aligned)

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[1:4])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')
        else:
            print('Aligning all tiles')

            for tile_id, tile in enumerate(tiles):

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_hybseq_position_path + '/' + tile, aligned)

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[1:4])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

    #use n cores for parallel processing
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position) for position in positions])
    ray.shutdown()

def align_preseq_to_hybseq_parallel(args):
    '''

    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align geneseq cycles
    print('Aligning preseq cycles')

    @ray.remote
    def align_position(position):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'preseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_preseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
        preseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_preseq_path, cycle)
        tiles = helpers.list_files(preseq_position_path)

        for tile in tiles:
            shutil.copy(preseq_position_path + tile, aligned_preseq_position_path + tile)

        #check to see if transformations already exist
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
            xshifts = []
            yshifts = []

            for tile in tiles:
                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = reference[args.hybseq_bv_chan]
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)
                #to_align_mp = np.max(to_align[0:4], axis=0)
                to_align_mp = to_align[args.bv_chan]
                #to_align_mp = np.mean(to_align[0:4], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
    #        x_shift, y_shift = -133.0, -73.0
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)
            print('std shifts for cycle', cycle, 'are', np.std(xshifts), np.std(yshifts))
            print('Aligning all tiles')

            for tile_id, tile in enumerate(tiles):

                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = reference[args.hybseq_bv_chan]

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > args.allowed_reg_shift:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > args.allowed_reg_shift:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                #save transformation matrix
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_preseq_position_path + '/' + tile, aligned[:2])

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

                bv_overlap = helpers.check_images_overlap(aligned[args.bv_chan], reference, save_output=False)
                resized = np.zeros((bv_overlap.shape[0], int(bv_overlap.shape[1] / args.downsample_factor),
                                    int(bv_overlap.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(bv_overlap.shape[0]):
                    resized[i] = cv2.resize(bv_overlap[i], (resized.shape[1], resized.shape[2]))
                tif.imwrite(aligned_tile_path + 'zoverlap_bv.tif', resized)
        else:
            print('Transformations already exist for', cycle, position)
            print('Aligning all tiles')

            for tile in tiles:
                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                # reference = np.max(reference[0:4], axis=0)
                reference = reference[args.hybseq_bv_chan]

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # save transformation matrix
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_preseq_position_path + '/' + tile, aligned[:2])

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

                bv_overlap = helpers.check_images_overlap(aligned[args.bv_chan], reference, save_output=False)

                resized = np.zeros((bv_overlap.shape[0], int(bv_overlap.shape[1] / args.downsample_factor),
                                    int(bv_overlap.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(bv_overlap.shape[0]):
                    resized[i] = cv2.resize(bv_overlap[i], (resized.shape[1], resized.shape[2]))
                tif.imwrite(aligned_tile_path + 'zoverlap_bv.tif', resized)

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

    #use n cores for parallel processing
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position) for position in positions])
    ray.shutdown()

def align_first_bcseq_to_geneseq_parallel(args):

    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align geneseq cycles
    print('Aligning first bcseq cycles')

    @ray.remote
    def align_position(position):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'bcseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
        bcseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)

        tiles = helpers.list_files(bcseq_position_path)

        for tile in tiles:
            shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        #check to see if transformations already exist
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

            xshifts = []
            yshifts = []
            #run a consensus strategy. go through all the tiles of the same position and take the most likely offsets.
            for tile in tiles:
                reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = np.median(reference[0:4], axis=0)
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                to_align_mp = np.median(to_align[0:4], axis=0)
                #to_align_mp = np.mean(to_align[0:4], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)

            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)

                cycle = 'bcseq_1'
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > args.allowed_reg_shift:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > args.allowed_reg_shift:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        else:
            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                cycle = 'bcseq_1'
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

    #use n cores for parallel processing
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position) for position in positions])
    ray.shutdown()

def align_bcseq_cycles_parallel(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    cycles.remove('bcseq_1')
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Aligning bcseq cycles')
    @ray.remote
    def align_position(position, cycles, no_channels, args):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
            bcseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

            tiles = helpers.list_files(bcseq_position_path)
            for tile in tiles:
                shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)
            if not os.path.exists(transformations_cycle_path + position):

                xshifts = []
                yshifts = []

                # run a consensus strategy. go through all the tiles of the same position and take the most likely offsets.
                for tile in tiles:
                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
                    reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_1' + '/' + position + '/' + tile)
                    reference = np.max(reference[0:4], axis=0)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                    to_align_mp = np.max(to_align[0:4], axis=0)
                    # to_align_mp = np.mean(to_align[0:4], axis=0)

                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                    yshifts.append(transformation_matrix[0, 2])
                    xshifts.append(transformation_matrix[1, 2])

                print('shifts for cycle and position', cycle, position, xshifts, yshifts)
                x_shift = np.percentile(xshifts, 50)
                y_shift = np.percentile(yshifts, 50)
                print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)

                for tile_id, tile in enumerate(tiles):
                    print('Aligning tile', tile)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                    transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                    if abs(transformation_matrix[0, 2] - y_shift) > args.allowed_reg_shift:
                        print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                        transformation_matrix[0, 2] = y_shift

                    if abs(transformation_matrix[1, 2] - x_shift) > args.allowed_reg_shift:
                        print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                        transformation_matrix[1, 2] = x_shift

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
            else:
                for tile_id, tile in enumerate(tiles):
                    print('Aligning tile', tile)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)
                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position, cycles, no_channels, args) for position in positions])
    ray.shutdown()

def align_bcseq_cycles_parallel_bcseq(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    #cycles.remove('bcseq_1')
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions
    #cycles = ['bcseq_16']
    #positions = ['Pos0', 'Pos1', 'Pos2']
    #Align bcseq cycles
    print('Aligning bcseq cycles')
    @ray.remote
    def align_position(position, cycles, no_channels, args):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
            bcseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

            tiles = helpers.list_files(bcseq_position_path)
            for tile in tiles:
                shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        for cycle_id, cycle in enumerate(cycles):
            # check to see if transformations already exist
            transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)
            if not os.path.exists(transformations_cycle_path + position):

                xshifts = []
                yshifts = []

                # run a consensus strategy. go through all the tiles of the same position and take the most likely offsets.
                for tile in tiles:
                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
                    reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_1' + '/' + position + '/' + tile)
                    reference = np.max(reference[0:4], axis=0)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                    to_align_mp = np.max(to_align[0:4], axis=0)
                    # to_align_mp = np.mean(to_align[0:4], axis=0)

                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                    yshifts.append(transformation_matrix[0, 2])
                    xshifts.append(transformation_matrix[1, 2])

                print('shifts for cycle and position', cycle, position, xshifts, yshifts)
                x_shift = np.percentile(xshifts, 50)
                y_shift = np.percentile(yshifts, 50)
                print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)

                #x_shift = -22
                #y_shift = -16
                for tile_id, tile in enumerate(tiles):
                    print('Aligning tile', tile)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                    transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                    if abs(transformation_matrix[0, 2] - y_shift) > 20:
                        print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                        transformation_matrix[0, 2] = y_shift

                    if abs(transformation_matrix[1, 2] - x_shift) > 20:
                        print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                        transformation_matrix[1, 2] = x_shift

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')
            else:
                for tile_id, tile in enumerate(tiles):
                    print('Aligning tile', tile)
                    transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                    aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                    aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                    to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                    # apply transformation matrix on raw image for each channel
                    to_align = to_align[:no_channels]
                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)
                    resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                        int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                    for i in range(aligned.shape[0]):
                        resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                    aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                    aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                    aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position, cycles, no_channels, args) for position in positions])
    ray.shutdown()

def stitch_tiles_middle(args):
    '''
    Stitches images taken in tiles sequence by microscope. Really messy way of doing it, but couldn't be bothered to find a better one.
    Input:
        :param filepath: Path to where files are
        :param input_overlap: known overlap between images
        :param specific_chan: integer; if specific channel is wanted only, specify here
    Returns:
        nothing - it creates a folder 'stitched' where it adds all images
    '''

    print('Stitching images..')

    sequencing_folders = [args.proc_aligned_geneseq_path, args.proc_aligned_bcseq_path, args.proc_aligned_hybseq_path, args.proc_aligned_preseq_path]
    stitched_sequencing_folders = [args.proc_stitched_geneseq_path, args.proc_stitched_bcseq_path, args.proc_stitched_hybseq_path, args.proc_stitched_preseq_path]

#    sequencing_folders = [args.proc_aligned_hybseq_path]
#    stitched_sequencing_folders = [args.proc_stitched_hybseq_path]

#    sequencing_folders = [args.proc_aligned_preseq_path]
#    stitched_sequencing_folders = [args.proc_stitched_preseq_path]

#    sequencing_folders = [args.proc_aligned_bcseq_path]
#    stitched_sequencing_folders = [args.proc_stitched_bcseq_path]

    cycles = helpers.list_files(sequencing_folders[0])
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_folders[0] + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions


    all_tiles = helpers.list_files(sequencing_folders[0] + cycles[0] + '/' + positions[0])


    # Sometimes images are not max projections, so their naming scheme is different. Namely, it's not 'MAX_Pos1_1_1', but just 'Pos1_1_1'.
    # Distinguish between the two cases by getting the starting segment before 'Pos'.
    segment = re.search('(.*)Pos(.*)', all_tiles[0])
    starting_segment = segment.group(1)

    sample_image = tif.imread(sequencing_folders[0] + cycles[0] + '/' + positions[0] + '/' + all_tiles[0])

    if sample_image.ndim == 2:
        no_channels = 1
        pixel_dim = sample_image.shape[1]
    elif sample_image.ndim == 3:
        no_channels = min(sample_image.shape)
        pixel_dim = sample_image.shape[1]
    else:
        print('I only prepared this function for 2 or 3 channel images')

    x_max = 0
    y_max = 0
    for pos in all_tiles:
        segment = re.search(starting_segment + '(.*)_(.*)_(.*).tif', pos)
        x_max = max(x_max, int(segment.group(3)))
        y_max = max(y_max, int(segment.group(2)))


    reduced_pixel_dim = int((1 - args.stitching_overlap) * sample_image.shape[1])
    x_max += 1
    y_max += 1
    x_pixels = (x_max - 1) * reduced_pixel_dim + sample_image.shape[1]
    y_pixels = (y_max - 1) * reduced_pixel_dim + sample_image.shape[1]

    #reduced_pixel_dim += 100
    buffer = args.stitching_buffer
    pixel_dim -= buffer


    for (sequencing_folder, stitched_sequencing_folder) in zip(sequencing_folders, stitched_sequencing_folders):

        # for each position, stitch images. start by stitching images into individual columns, and the stitch columns.
        # The maths is messy, but it works
        cycles = helpers.list_files(sequencing_folder)
        cycles = helpers.human_sort(cycles)
        
        if len(cycles) > 0:

            sample_image = tif.imread(sequencing_folder + cycles[0] + '/' + positions[0] + '/' + all_tiles[0])
            if sample_image.ndim == 2:
                no_channels = 1
            elif sample_image.ndim == 3:
                no_channels = min(sample_image.shape)

            for position in positions:
                tic = time.perf_counter()
                stitched_position = np.empty((no_channels * len(cycles), x_pixels, y_pixels), dtype=np.int16)
                stitched_position_path = helpers.quick_dir(stitched_sequencing_folder, position)
                for cycled_id, cycle in enumerate(cycles):
                    cycle_path = helpers.quick_dir(sequencing_folder, cycle)
                    position_path = helpers.quick_dir(cycle_path, position)

                    stitched = np.empty((no_channels, x_pixels, y_pixels), dtype=np.float16)
                    stitched_cols = np.empty((no_channels, y_max, x_pixels, sample_image.shape[1]), dtype=np.int16)

                    for column in range(y_max):
                        for row in range(x_max):
                            image_name = starting_segment + position + '_00' + str(y_max - column - 1) + '_00' + str(x_max - row - 1) + '.tif'

                            image = tif.imread(position_path + '/' + image_name)

                            if image.shape[2] == no_channels:
                                image = np.transpose(image, (2, 0, 1))

                            if image.ndim == 2: image = np.expand_dims(image, axis=0)

                            if row == 0:
                                stitched_cols[:, column, :buffer + reduced_pixel_dim, :] = image[:, 0:buffer + reduced_pixel_dim, :]
                            elif row != (x_max - 1):
                                stitched_cols[:, column, buffer + row * reduced_pixel_dim: buffer + (row + 1) * reduced_pixel_dim, :] = image[:, buffer:buffer + reduced_pixel_dim, :]
                            else:
                                stitched_cols[:, column, buffer + row * reduced_pixel_dim:, :] = image[:, buffer:, :]

                    for column in range(y_max):
                        if column == 0:
                            stitched[:, :, :buffer + reduced_pixel_dim] = stitched_cols[:, column, :, 0:buffer + reduced_pixel_dim]
                        elif column != (y_max - 1):
                            stitched[:, :, buffer + column * reduced_pixel_dim:buffer + (column + 1) * reduced_pixel_dim] = stitched_cols[:, column, :, buffer:buffer + reduced_pixel_dim]
                        else:
                            stitched[:, :, buffer + column * reduced_pixel_dim:buffer + (column * reduced_pixel_dim + pixel_dim)] = stitched_cols[:, column, :, buffer:]

                    stitched = stitched.astype(np.uint16)

                    if args.flip_horizontally is True:
                        if stitched.ndim == 2:
                            stitched = np.flip(stitched, axis=1)
                        elif stitched.ndim == 3:
                            stitched = np.flip(stitched, axis=2)
    #                if sequencing_folder == args.proc_aligned_hybseq_path:
    #                    tif.imwrite(args.proc_stitched_funseq_bv_path + position + '.tif', stitched[args.blood_vessels_chan])
    #                    tif.imwrite(args.proc_stitched_funseq_somas_path + position + '.tif', stitched[args.somas_chan])
                    stitched_position[cycled_id * no_channels: (cycled_id + 1) * no_channels] = stitched

                tif.imwrite(stitched_position_path + position + '.tif', stitched_position)
                resized = np.zeros((stitched_position.shape[0], int(stitched_position.shape[1] / args.downsample_factor),
                                    int(stitched_position.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(stitched_position.shape[0]):
                    resized[i] = cv2.resize(stitched_position[i], (resized.shape[1], resized.shape[2]))

                tif.imwrite(args.proc_checks_alignment_stitch_path + position + '.tif', resized)

                toc = time.perf_counter()
                print('stitching of ' + position + sequencing_folder + ' finished in ' + f'{toc - tic:0.4f}' + ' seconds')

def apply_slice_transformations(args):

    #sequencing_folders = [args.proc_stitched_geneseq_path, args.proc_stitched_hybseq_path]
    #transformed_folders = [args.proc_transformed_geneseq_path, args.proc_transformed_hybseq_path]


#    sequencing_folders = [args.proc_stitched_bcseq_path]
#    transformed_folders = [args.proc_transformed_bcseq_path]

    sequencing_folders = [args.proc_stitched_geneseq_path, args.proc_stitched_bcseq_path, args.proc_stitched_hybseq_path, args.proc_stitched_preseq_path]
    transformed_folders = [args.proc_transformed_geneseq_path, args.proc_transformed_bcseq_path, args.proc_transformed_hybseq_path, args.proc_transformed_preseq_path]

#    sequencing_folders = [args.proc_stitched_hybseq_path]
#    transformed_folders = [args.proc_transformed_hybseq_path]

#    sequencing_folders = [args.proc_stitched_preseq_path]
#    transformed_folders = [args.proc_transformed_preseq_path]


    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_folders[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions


    for (sequencing_folder, transformed_folder) in zip(sequencing_folders, transformed_folders):
        print('Applying transformations for folder', sequencing_folder)

        @ray.remote
        def transform_position(position):
            print('Applying transformations for position', position)
            tic = time.perf_counter()

            position_path = helpers.quick_dir(sequencing_folder, position)
            transformed_position_path = helpers.quick_dir(transformed_folder, position)
            transformations_position_path = helpers.quick_dir(args.proc_transf_macro_path, position)

            position_stack = tif.imread(position_path + position + '.tif')
            position_stack = position_stack.astype(np.int16)
            xy_shift = np.loadtxt(transformations_position_path + 'xy_shift.txt')
            transformation_matrix = np.array([[1, 0, xy_shift[1]], [0, 1, xy_shift[0]]])
            for chan_id in range(position_stack.shape[0]):
                position_stack[chan_id] = cv2.warpAffine(position_stack[chan_id], transformation_matrix,
                                                               (position_stack.shape[1], position_stack.shape[2]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            macro_rotation = np.loadtxt(transformations_position_path + 'macro_rotation.txt')
            for chan_id in range(position_stack.shape[0]):
                position_stack[chan_id] = scipy.ndimage.rotate(position_stack[chan_id], angle=macro_rotation, reshape=False)
            if os.path.isfile(transformations_position_path + 'rotation_180.txt'):
                for chan_id in range(position_stack.shape[0]):
                   position_stack[chan_id] = cv2.rotate(position_stack[chan_id], cv2.ROTATE_180)

            tif.imwrite(transformed_position_path + position + '.tif', position_stack)
            #os.remove(position_path + position + '.tif')
            if sequencing_folder == args.proc_stitched_bcseq_path:
                samples_position_path = helpers.quick_dir(args.proc_samples_path, position)
                tif.imwrite(samples_position_path + position + '.tif', np.max(position_stack[:4], axis=0))

            toc = time.perf_counter()
            print('Position ' + position + ' finished in ' + f'{toc - tic:0.4f}' + ' seconds')

        # use n cores for parallel processing
        ray.init(num_cpus=args.nr_cpus)
        ray.get([transform_position.remote(position) for position in positions])
        ray.shutdown()

def create_funseq_stacks(args):
    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_hybseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    last_pos = helpers.get_trailing_number(positions[-1]) + 1
    print('last pos is', last_pos)

    sample_image = tif.imread(args.proc_transformed_hybseq_path + positions[0] + '/' + positions[0] + '.tif')
    #bv_stack = np.zeros(shape=(last_pos, int(sample_image.shape[1]/2), int(sample_image.shape[2]/2)), dtype=np.int16)
    bv_stack = np.zeros(shape=(last_pos, sample_image.shape[1], sample_image.shape[2]), dtype=np.int16)
    somas_stack = np.zeros_like(bv_stack)
    bv_somas_stack = np.zeros_like(bv_stack)
    gfp_bv_somas_stack = np.zeros_like(bv_stack)
    barcoded_stack = np.zeros_like(bv_stack)
    geneseq_stack = np.zeros_like(bv_stack)


    for pos_id, position in enumerate(positions):
        id = helpers.get_trailing_number(position)
        position_image = tif.imread(args.proc_transformed_hybseq_path + position + '/' + position + '.tif')
        somas_stack[id] = position_image[args.hybseq_dapi_channel]
        geneseq_stack[id] = position_image[2]
        #somas_stack[id] = cv2.resize(position_image[args.hybseq_dapi_channel], (bv_stack.shape[1], bv_stack.shape[2]))

        position_image = tif.imread(args.proc_transformed_preseq_path + position + '/' + position + '.tif')
        bv_stack[id] = position_image[0]
        #bv_stack[id] = cv2.resize(position_image[1], (bv_stack.shape[1], bv_stack.shape[2]))
        gfp_somas = position_image[1]
        #gfp_somas = cv2.resize(position_image[0], (bv_stack.shape[1], bv_stack.shape[2]))
        #bv_stack[id] = position_image[0]
        #somas_stack[id] = position_image[0]
        #bv_stack[id] = skimage.exposure.equalize_hist(position_image[0])
        #bv_somas_duo = np.stack((bv_stack[id]*15, somas_stack[id]), axis=0)
        #bv_somas_duo = np.stack((bv_stack[id]*30, somas_stack[id]), axis=0)
        bv_somas_duo = np.stack((bv_stack[id]*15, somas_stack[id]), axis=0)
        bv_somas_stack[id] = np.max(bv_somas_duo, axis=0)
        bv_genes_duo = np.stack((bv_stack[id]*1, geneseq_stack[id]), axis=0)
        geneseq_stack[id] = np.max(bv_genes_duo, axis=0)
        gfp_somas_bv_duo = np.stack((bv_stack[id]*20, gfp_somas), axis=0)
        gfp_bv_somas_stack[id] = np.max(gfp_somas_bv_duo, axis=0)
        if os.path.exists(args.proc_samples_path + position):
            samples_position_path = helpers.quick_dir(args.proc_samples_path, position)
            barcoded_stack[id] = tif.imread(samples_position_path + position + '.tif')
            bv_barcoded_duo = np.stack((bv_stack[id]*2, barcoded_stack[id]), axis=0)
            barcoded_stack[id] = np.max(bv_barcoded_duo, axis=0)

    tif.imwrite(args.matching_images_path + 'somas.tif', somas_stack)
    tif.imwrite(args.matching_images_path + 'blood_vessels.tif', bv_stack)
    tif.imwrite(args.matching_images_path + 'somas_blood_vessels.tif', bv_somas_stack)
    tif.imwrite(args.matching_images_path + 'gfp_somas_blood_vessels.tif', gfp_bv_somas_stack)
    tif.imwrite(args.matching_images_path + 'barcoded.tif', barcoded_stack)
    tif.imwrite(args.matching_images_path + 'genes.tif', geneseq_stack)

def bardensr_gene_basecalling(args):
    '''
    Bardensr is used for gene basecalling. genebook has to be loaded and then some preprocessing steps such as background subtraction
    and color correction.
    :param local_path:
    :param helper_files_path:
    :param no_channels:
    :param basecalling_thresh:
    :param max_checks:
    :param flat_result:
    :return:
    '''
    print('Gene basecalling...')

    if len(args.positions) == 0:
        #positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.list_files(args.proc_stitched_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    sample_image = tif.imread(args.proc_transformed_geneseq_path + positions[0] + '/' + positions[0] + '.tif')
    #sample_image = tif.imread(args.proc_stitched_geneseq_path + positions[0] + '/' + positions[0] + '.tif')
    sample_image = sample_image.squeeze()

    codeflat, genebook, unused_bc_ids = prepare_codebook(args, no_channels=sample_image.shape[0])
    del sample_image

    @ray.remote
    def basecall_position(position):
        pos_tic = time.perf_counter()

        position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        #position_path = helpers.quick_dir(args.proc_stitched_geneseq_path, position)

        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        checks_basecalling_position_path = helpers.quick_dir(args.proc_checks_basecalling_path, position)

        Xflat = tif.imread(position_path + position + '.tif')
        #Xflat = Xflat[:, 2700:3100, 2700:3100]
        Xflat = np.expand_dims(Xflat, axis=1)
        Xflat = bardensr.preprocessing.minmax(Xflat)
        Xflat = bardensr.preprocessing.background_subtraction(Xflat, [0, 10, 10])
        Xflat = bardensr.preprocessing.minmax(Xflat)

        evidence_tensor = bardensr.spot_calling.estimate_density_singleshot(Xflat, codeflat, noisefloor=args.noisefloor)

        result = bardensr.spot_calling.find_peaks(evidence_tensor, args.basecalling_thresh)
        result = result.rename(columns={"m0": "rol_Z", "m1": "rol_Y", "m2": "rol_X"})
        result['rol_ID'] = range(0, len(result))
        genes_result = result.merge(genebook, left_on='j', right_on='geneID')
        del genes_result['j']
        genes_result.to_csv(genes_position_path + 'rolonies.csv')

        centers = genes_result[['rol_Y', 'rol_X', 'geneID']].to_numpy(dtype=np.int)

        first_cycle = np.squeeze(Xflat[0:4])
        first_cycle_mp = np.max(first_cycle, axis=0)
        rol_image = np.zeros_like(first_cycle_mp)

        for rol_id in range(len(centers)):
            center = tuple([int(centers[rol_id, 1]), int(centers[rol_id, 0])])
            color = 250
            cv2.circle(rol_image, center, radius=2, color=color, thickness=-1)


        combined = np.stack((first_cycle_mp, rol_image), axis=0)
        tif.imwrite(checks_basecalling_position_path + 'gene_basecalling.tif', combined)

        no_unused_barcodes = 0
        for unused_id in unused_bc_ids:
            try:
                no_unused_barcodes += result['j'].value_counts()[unused_id]
            except KeyError:
                no_unused_barcodes += 0
        try:
            error_rate = no_unused_barcodes / result.shape[0]
        except ZeroDivisionError:
            error_rate = 0
        no_rol = result.shape[0]
        no_genes = len(pd.unique(result['j']))

        pos_toc = time.perf_counter()
        print('Gene basecalling for position', position, 'finished in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
        print('Nr. of rolonies identified:', no_rol)
        print('Nr. of unused barcodes identified:', no_unused_barcodes, ', error rate ', error_rate)
        print("Nr. of genes identified:", no_genes)
        # generate a file with the results
        #title should be gene_basecalling_summary.txt + args.basecalling_thresh
        summary_title = str(args.basecalling_thresh) + '_gene_basecalling_summary.txt'
        with open(genes_position_path + summary_title, 'w') as f:
            f.write('Gene basecalling for position ' + position + ' finished in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
            f.write('\n')
            f.write('Nr. of rolonies identified: ' + str(no_rol))
            f.write('\n')
            f.write('Nr. of unused barcodes identified: ' + str(no_unused_barcodes) + ', error rate ' + str(error_rate))
            f.write('\n')
            f.write('Nr. of genes identified: ' + str(no_genes))


    #use n cores for parallel processing
    ray.init(num_cpus=args.nr_cpus)
    ray.get([basecall_position.remote(position) for position in positions])
    ray.shutdown()

def cellpose_segment_barcoded(args):

    model = models.Cellpose(model_type='cyto2')

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_bcseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #keep only positions that are also in args.funseq_positions
    positions = [pos for pos in positions if pos in args.funseq_positions]

    @ray.remote
    def segment_position(position):
        print('Segmenting position', position)
        pos_tic = time.perf_counter()
        coordinates_position_path = helpers.quick_dir(args.proc_coordinates_path, position)

        bcseq_position_path = helpers.quick_dir(args.proc_transformed_bcseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_barcoded_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        bcseq_image = tif.imread(bcseq_position_path + position + '.tif')

        corners = np.loadtxt(coordinates_position_path + 'barcodes_area.txt')
        x_min, x_max = int(corners[0, 1]), int(corners[2, 1])
        y_min, y_max = int(corners[0, 0]), int(corners[2, 0])

        bcseq_image = np.max(bcseq_image, axis=0)
        reduced_image = bcseq_image[y_min:y_max, x_min:x_max]

        channels = [0, 0]
        reduced_mask, flows, styles, diams = model.eval(reduced_image, diameter=None, channels=channels, flow_threshold=0.4, cellprob_threshold=0)

        diams = np.round(diams, 2)
        np.savetxt(coordinates_position_path + 'barcoded_segmentation_diameter.txt', np.array([diams]))

        mask = np.zeros_like(bcseq_image, dtype=reduced_mask.dtype)
        mask[y_min:y_max, x_min:x_max] = reduced_mask
        tif.imwrite(position_output_path + position + '.tif', mask)

        labels_image = bcseq_image * (mask > 0)
        cycle_image_check = helpers.check_images_overlap(bcseq_image, labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_segmentation.tif', cycle_image_check)
        contours, centroids = helpers.find_cellpose_contours(mask)

        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds, with', len(contours), 'cells found')

    ray.init(num_cpus=args.nr_cpus)
    ray.get([segment_position.remote(position) for position in positions])
    ray.shutdown()


def stardist_segment_hybseq_rolonies(args):
    model = StarDist2D(None, name=args.stardist_model_name, basedir=args.stardist_model_path)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_hybseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    segmentation_channels = [args.hybseq_inhib_channel, args.hybseq_excite_channel, args.hybseq_IT_channel]
    channel_names = ['inhib', 'excite', 'IT']

    rol_id_offset = 0

    for position in positions:
        print('Segmenting position', position)
        pos_tic = time.perf_counter()

        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')

        for segmentation_channel, channel_name in zip(segmentation_channels, channel_names):
            image = hybseq_image[segmentation_channel]
            axis_norm = (0, 1)
            norm_image = normalize(image, 1, 99.8, axis=axis_norm)

            # Labels is an array with the masks and details contains the centroids.
            mask, details = model.predict_instances(norm_image, prob_thresh=args.stardist_probability_threshold)

            labels_image = image * (mask > 0)
            cycle_image_check = helpers.check_images_overlap(image, labels_image, save_output=False)
            tif.imwrite(checks_segmentation_position_path + channel_name + '_segmentation.tif', cycle_image_check)

            contours = np.array(details['coord'], dtype=np.int32)
            centroids = np.array(details['points'])

            # create dataframe with cell_id, centroid.x, centroid.y
            centroids_df = pd.DataFrame(centroids, columns=['rol_Y', 'rol_X'])
            centroids_df['rol_ID'] = centroids_df.index + rol_id_offset
            centroids_df['slice'] = position
            rol_id_offset += len(contours)

            centroids_df.to_csv(centroids_position_path + channel_name + '_rolonies.csv')
            # np.savetxt(centroids_position_path + channel_name + '_centroids.txt', centroids, fmt='%i')
            print('Channel', channel_name, ' segmented with', len(contours), 'cells found')

        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def allocate_rolonies_to_barcoded(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions = [pos for pos in positions if pos in args.funseq_positions]

    cell_id_offset = 0

    for position in positions:
        print('Processing position', position)
        pos_tic = time.perf_counter()
        position_index = helpers.get_trailing_number(position)

        geneseq_position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        bcseq_position_path = helpers.quick_dir(args.proc_transformed_bcseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_barcoded_path, position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        checks_basecalling_position_path = helpers.quick_dir(args.proc_checks_basecalling_path, position)

        bcseq_image = tif.imread(bcseq_position_path + position + '.tif')
        max_bcseq_image = np.max(bcseq_image[:4], axis=0)

        geneseq_image = tif.imread(geneseq_position_path + position + '.tif')
        geneseq_image = np.max(geneseq_image, axis=0)

        mask = tif.imread(position_output_path + position + '.tif')
        labels_image = max_bcseq_image * (mask > 0)

        contours, centroids = helpers.find_cellpose_contours(mask)

        cells_df = pd.DataFrame(columns=['cell_ID', 'cell_X', 'cell_Y', 'cell_Z', 'slice'])
        contours_df = pd.DataFrame(columns=['contour'])
        contours_df['contour'] = contours_df['contour'].astype('object')

        cells_df['genes'] = ''
        cells_df['gene_IDs'] = ''

        rolonies_df = pd.read_csv(genes_position_path + 'rolonies.csv', index_col=[0])
        rolonies = rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        genes_cells = rolonies_df.copy()

        excite_rolonies_df = pd.read_csv(centroids_position_path + 'excite_rolonies.csv', index_col=[0])
        inhib_rolonies_df = pd.read_csv(centroids_position_path + 'inhib_rolonies.csv', index_col=[0])
        IT_rolonies_df = pd.read_csv(centroids_position_path + 'IT_rolonies.csv', index_col=[0])

        excite_rolonies = excite_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        inhib_rolonies = inhib_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        IT_rolonies = IT_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)

        all_contours_image = np.zeros_like(geneseq_image, dtype=np.int8)
        all_hybseq_rol_image = np.zeros_like(geneseq_image, dtype=np.int8)

        cells_intensities = np.empty(shape=(0, bcseq_image.shape[0]))

        printcounter = 0
        barcoded_cells = []
        for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
            cells_df.at[cell_id, 'cell_ID'] = cell_id + cell_id_offset
            cells_df.at[cell_id, 'cell_X'] = centroid[0]
            cells_df.at[cell_id, 'cell_Y'] = centroid[1]
            cells_df.at[cell_id, 'cell_Z'] = position_index * 20
            cells_df.at[cell_id, 'genes'] = ''
            cells_df.at[cell_id, 'gene_IDs'] = ''
            cells_df.at[cell_id, 'gene_counts'] = ''
            cells_df.at[cell_id, 'slice'] = position
            cells_df.at[cell_id, 'excite_count'] = ''
            cells_df.at[cell_id, 'inhib_count'] = ''
            cells_df.at[cell_id, 'IT_count'] = ''
            cells_df.at[cell_id, 'excite_IDs'] = ''
            cells_df.at[cell_id, 'inhib_IDs'] = ''
            cells_df.at[cell_id, 'IT_IDs'] = ''
            contours_df.at[cell_id, 'contour'] = contour

            contours_image = labels_image
            centroids_image = np.zeros_like(labels_image, dtype=np.int8)
            cv2.drawContours(image=contours_image, contours=[contour], contourIdx=-1, color=120, thickness=1)
            center = tuple([int(centroid[1]), int(centroid[0])])
            cv2.circle(centroids_image, center, 15, 200, -1)

            if printcounter == 100:
                print('cell id is', cell_id, 'out of', len(centroids))
                printcounter = 0
            printcounter += 1

            cell_color = np.random.randint(255)
            cv2.drawContours(image=all_contours_image, contours=[contour], contourIdx=-1, color=cell_color, thickness=1)
            cv2.drawContours(image=all_hybseq_rol_image, contours=[contour], contourIdx=-1, color=cell_color, thickness=1)

            relevant_rows = np.where((rolonies[:, 1] < centroid[0] + 30) & (rolonies[:, 1] > centroid[0] - 30))
            relevant_rolonies = rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = genes_cells[genes_cells['rol_ID'] == real_rol_ID].index[0]
                        genes_cells.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        genes_cells.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        genes_cells.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        genes_cells.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        genes_cells.at[row_index_rol_ID, 'cell_alloc'] = 1
                        genes_cells.at[row_index_rol_ID, 'slice'] = position
                        if cells_df.at[cell_id, 'genes'] == '':
                            cells_df.at[cell_id, 'genes'] = rolonies_df.at[real_rol_ID, 'gene']
                            cells_df.at[cell_id, 'gene_IDs'] = rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                            cells_df.at[cell_id, 'gene_counts'] = '1'
                        else:
                            # pull cells_df.at[cell_id, 'genes'], cells_df.at[cell_id, 'gene_IDs'] and cells_df.at[cell_id, 'gene_counts']into lists.
                            # check whether rolonies_df.at[real_rol_ID, 'gene'] and rolonies_df.at[real_rol_ID, 'geneID'].astype(str) exist in lists.
                            # if they do, increment corresponding element of gene_counts
                            # if they don't, append to lists
                            genes = cells_df.at[cell_id, 'genes'].split(',')
                            gene_IDs = cells_df.at[cell_id, 'gene_IDs'].split(',')
                            gene_counts = cells_df.at[cell_id, 'gene_counts'].split(',')

                            if rolonies_df.at[real_rol_ID, 'gene'] in genes:
                                gene_index = genes.index(rolonies_df.at[real_rol_ID, 'gene'])
                                gene_counts[gene_index] = str(int(gene_counts[gene_index]) + 1)
                            else:
                                genes.append(rolonies_df.at[real_rol_ID, 'gene'])
                                gene_IDs.append(str(rolonies_df.at[real_rol_ID, 'geneID']))
                                gene_counts.append('1')
                            cells_df.at[cell_id, 'genes'] = ','.join(genes)
                            cells_df.at[cell_id, 'gene_IDs'] = ','.join(gene_IDs)
                            cells_df.at[cell_id, 'gene_counts'] = ','.join(gene_counts)#cells_df.at[cell_id, 'gene_IDs'] = cells_df.at[cell_id, 'gene_IDs'] + ',' + rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                        cv2.circle(all_contours_image, rolony_centroid, 2, color=cell_color, thickness=-1)
            except IndexError:
                continue

            # do the same for excitatory cells
            relevant_rows = np.where((excite_rolonies[:, 1] < centroid[0] + 20) & (excite_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = excite_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]

                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        # find index of real_rol_ID in excite_rolonies_df, column rol_ID
                        row_index_rol_ID = excite_rolonies_df[excite_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1

                        if cells_df.at[cell_id, 'excite_count'] == '':
                            cells_df.at[cell_id, 'excite_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'excite_count'] = '1'
                        else:
                            excite_IDs = cells_df.at[cell_id, 'excite_IDs'].split(',')
                            excite_count = cells_df.at[cell_id, 'excite_count']

                            # add the next excite ID and increment excite_count
                            excite_IDs.append(str(real_rol_ID))
                            excite_count = str(int(excite_count) + 1)

                            cells_df.at[cell_id, 'excite_IDs'] = ','.join(excite_IDs)
                            cells_df.at[cell_id, 'excite_count'] = excite_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, radius=4, color=250, thickness=-1)

            except IndexError:
                continue

            # do the same for inhibitory cells
            relevant_rows = np.where((inhib_rolonies[:, 1] < centroid[0] + 20) & (inhib_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = inhib_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]

                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = inhib_rolonies_df[inhib_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1
                        if cells_df.at[cell_id, 'inhib_count'] == '':
                            cells_df.at[cell_id, 'inhib_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'inhib_count'] = '1'
                        else:
                            inhib_IDs = cells_df.at[cell_id, 'inhib_IDs'].split(',')
                            inhib_count = cells_df.at[cell_id, 'inhib_count']

                            # add the next excite ID and increment excite_count
                            inhib_IDs.append(str(real_rol_ID))
                            inhib_count = str(int(inhib_count) + 1)

                            cells_df.at[cell_id, 'inhib_IDs'] = ','.join(inhib_IDs)
                            cells_df.at[cell_id, 'inhib_count'] = inhib_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, 4, color=100, thickness=2)

            except IndexError:
                continue

            # do the same for IT cells
            relevant_rows = np.where((IT_rolonies[:, 1] < centroid[0] + 20) & (IT_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = IT_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = IT_rolonies_df[IT_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1
                        if cells_df.at[cell_id, 'IT_count'] == '':
                            cells_df.at[cell_id, 'IT_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'IT_count'] = '1'
                        else:
                            IT_IDs = cells_df.at[cell_id, 'IT_IDs'].split(',')
                            IT_count = cells_df.at[cell_id, 'IT_count']

                            # add the next excite ID and increment excite_count
                            IT_IDs.append(str(real_rol_ID))
                            IT_count = str(int(IT_count) + 1)

                            cells_df.at[cell_id, 'IT_IDs'] = ','.join(IT_IDs)
                            cells_df.at[cell_id, 'IT_count'] = IT_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, 2, color=175, thickness=-1)

            except IndexError:
                continue


#        for cell_id, cell in enumerate(cells):
            barcoded_cells.append(cell_id + cell_id_offset)
            cell_mask = np.zeros_like(bcseq_image[0])
            cv2.drawContours(image=cell_mask, contours=[contour], contourIdx=-1, color=255, thickness=-1)
            pts = np.where(cell_mask == 255)
            chan_intensities = np.zeros(shape=(1, bcseq_image.shape[0]))
            for chan_id in range(bcseq_image.shape[0]):
                chan_intensities[0, chan_id] = int(np.mean(bcseq_image[chan_id][pts[0], pts[1]]))
            cells_intensities = np.vstack((cells_intensities, chan_intensities))
        for chan_id in range(cells_intensities.shape[1]):
            chan_mean = np.mean(cells_intensities[:, chan_id])
            cells_intensities[:, chan_id] = cells_intensities[:, chan_id] / chan_mean
        np.savetxt(centroids_position_path + 'intensities.txt', cells_intensities, fmt='%f2')

        no_channels = 4
        cycles = int(bcseq_image.shape[0] / no_channels)
        cells_seqs = np.zeros((2, cells_intensities.shape[0], cycles))

        for cell_id, cell in enumerate(barcoded_cells):
            for cycle_id in range(cycles):
                max_int = np.max(cells_intensities[cell_id, cycle_id * no_channels: (cycle_id + 1) * no_channels])
                max_id = np.argmax(cells_intensities[cell_id, cycle_id * no_channels: (cycle_id + 1) * no_channels])
                sum_quare_int = 0
                for i in range(4):
                    sum_quare_int += cells_intensities[cell_id, cycle_id * no_channels + i] ** 2
                cells_seqs[0, cell_id, cycle_id] = max_id
                cells_seqs[1, cell_id, cycle_id] = max_int / np.sqrt(sum_quare_int)

        first_n_bp = 15
        for cell_id, real_cell_id in enumerate(barcoded_cells):
            contour = [contours[cell_id]]
            for chan_id in range(bcseq_image.shape[0]):
                cv2.drawContours(image=bcseq_image[chan_id], contours=contour, contourIdx=-1, color=200, thickness=2)
            if np.mean(cells_seqs[1, cell_id, :first_n_bp] > args.basecalling_score_thresh):
                row_index = cells_df.index[cells_df['cell_ID'] == real_cell_id].tolist()
                cells_df.at[row_index[0], 'barcoded'] = 1
                cells_df.at[row_index[0], 'bc_seq'] = str(cells_seqs[0, cell_id])


                for chan_id in range(bcseq_image.shape[0]):
                    cv2.drawContours(image=bcseq_image[chan_id], contours=contour, contourIdx=-1, color=100, thickness=2)
                    if chan_id % 4 == cells_seqs[0, cell_id, int(chan_id / 4)]:
                        centroid = centroids[cell_id]
                        cell_centroid = tuple([int(centroid[0]), int(centroid[1])])
                        cv2.circle(bcseq_image[chan_id], center=cell_centroid, radius=30, color=350, thickness=3)

        resized = np.zeros((bcseq_image.shape[0], int(bcseq_image.shape[1]/args.downsample_factor), int(bcseq_image.shape[2]/args.downsample_factor)), dtype=np.int16)
        for i in range(bcseq_image.shape[0]):
            resized[i] = cv2.resize(bcseq_image[i], (resized.shape[1], resized.shape[2]))
        tif.imwrite(checks_basecalling_position_path + 'check_soma_basecalling.tif', resized)

        try:
            barcoded_cells_df = cells_df.loc[cells_df['barcoded'] == 1.0]

            print('nr barcoded cells is', len(barcoded_cells_df))
        except KeyError:
            print('no barcodes cells found')
            barcoded_cells_df = cells_df.iloc[:0, :].copy()
        np.savetxt(centroids_position_path + 'basecalling_scores.txt', cells_seqs[1], fmt='%f2')
        barcoded_cells_df.to_csv(genes_position_path + 'barcoded_cells.csv')
        geneseq_image = helpers.scale_to_8bit(geneseq_image, unsigned=False)

        contours_image_check = helpers.check_images_overlap(geneseq_image, all_contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_rolony_allocation.tif', contours_image_check)

        genes_cells.to_csv(genes_position_path + 'funseq_rolonies.csv')

        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')

        inhib_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_inhib_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'inhib_rolony_allocation_funseq.tif', inhib_image_check)

        excite_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_excite_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'excite_rolony_allocation_funseq.tif', excite_image_check)

        IT_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_IT_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'IT_rolony_allocation_funseq.tif', IT_image_check)

        excite_rolonies_df.to_csv(centroids_position_path + 'excite_funseq.csv')
        inhib_rolonies_df.to_csv(centroids_position_path + 'inhib_funseq.csv')
        IT_rolonies_df.to_csv(centroids_position_path + 'IT_funseq.csv')

        contours_df.to_csv(centroids_position_path + 'contours.csv')

        cell_id_offset += len(barcoded_cells_df)
        alloc_rol_ids = genes_cells.index[genes_cells['cell_alloc'] == 1].tolist()
        print('nr rolonies allocated is', len(alloc_rol_ids))
        pos_toc = time.perf_counter()
        print('Rolony allocation for position', position, 'done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def allocate_projection_areas(args):

    #barcodes_df = pd.read_csv(args.mapseq_path + 'barcodes_summary.csv', index_col=[0])
    barcodes_df = pd.read_csv(args.mapseq_path + 'barcodes_summary.csv')
    all_barcode_sequences = barcodes_df['sequence']

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_genes_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions
    positions = [pos for pos in positions if pos in args.funseq_positions]

    for position in positions:
        print('Identifying matches for position', position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        barcoded_cells_df = pd.read_csv(genes_position_path + 'barcoded_cells.csv', index_col=[0])
        #barcoded_cells_df = pd.read_csv(genes_position_path + 'barcoded_cells.csv')
        barcoded_cells_ids = barcoded_cells_df.index[barcoded_cells_df['barcoded'] == 1].tolist()
        print('looking at', len(barcoded_cells_ids), 'barcodes')
        counter = 0
        for row_id in barcoded_cells_ids:
            cell_sequence = barcoded_cells_df.at[row_id, 'bc_seq']
            cell_sequence = cell_sequence.replace(".", "")
            for bc_id, barcode_sequence in enumerate(all_barcode_sequences):
                if helpers.hamming_distance(cell_sequence, barcode_sequence) <= 1:
                    #pd_index = barcodes_df.at[bc_id, 'target_areas']
                    barcoded_cells_df.at[row_id, 'target_areas'] = barcodes_df.at[bc_id, 'target_areas']
                    barcoded_cells_df.at[row_id, 'target_areas_IDs'] = barcodes_df.at[bc_id, 'target_areas_IDs']
                    barcoded_cells_df.at[row_id, 'target_areas_counts'] = barcodes_df.at[bc_id, 'target_areas_counts']
                    barcoded_cells_df.at[row_id, 'bc_id'] = bc_id
                    barcoded_cells_df.at[row_id, 'projection'] = 1
                    counter += 1
        try:
            funseq_cells_df = barcoded_cells_df.loc[barcoded_cells_df['projection'] == 1.0]
            print('nr matches found is', len(funseq_cells_df))
            print(funseq_cells_df['target_areas'].tolist())
            characters_to_remove = ['[', ']', "'", ' ']
            for column in ['target_areas', 'target_areas_IDs', 'target_areas_counts']:
                for character in characters_to_remove:
                    funseq_cells_df[column] = funseq_cells_df[column].str.replace(character, '')
            funseq_cells_df = funseq_cells_df.drop(columns=['barcoded', 'projection', 'bc_seq'])

        except KeyError:
            print('no matches cells')
            funseq_cells_df = barcoded_cells_df.iloc[:0, :].copy()
            #add columns 'target_areas', 'target_areas_IDs', 'target_areas_counts'
            funseq_cells_df['target_areas'] = np.nan
            funseq_cells_df['target_areas_IDs'] = np.nan
            funseq_cells_df['target_areas_counts'] = np.nan
            funseq_cells_df = funseq_cells_df.drop(columns=['barcoded'])

        funseq_cells_df.to_csv(genes_position_path + 'funseq_cells.csv')

def cellpose_segment_cells(args):

    diam_model = models.Cellpose(model_type='cyto2')
    model = models.CellposeModel(pretrained_model=args.cellpose_model_path)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    presegment_path = helpers.quick_dir(args.proc_path, 'presegment')

    @ray.remote
    def segment_position(position):
        print('Segmenting position', position)
        pos_tic = time.perf_counter()

        geneseq_position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_somas_path, position)
        coordinates_position_path = helpers.quick_dir(args.proc_coordinates_path, position)

        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        geneseq_image = tif.imread(geneseq_position_path + position + '.tif')
        geneseq_image = np.max(geneseq_image, axis=0)

        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')
        hybseq_image = hybseq_image[args.hybseq_dapi_channel]
        #geneseq_image = skimage.exposure.match_histograms(geneseq_image, hybseq_image)
        geneseq_image = skimage.exposure.equalize_hist(geneseq_image)

        position_image = helpers.check_images_overlap(geneseq_image, hybseq_image, save_output=False)
        tif.imwrite(presegment_path + position + '_overlap.tif', position_image)
        #position_image = position_image[:, 3000:3500, 2000:2500]


        mask, flows, styles, nuclei_diam = diam_model.eval(hybseq_image, diameter=None, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0)
        tif.imwrite(position_output_path + position + '_nuclei.tif', mask)

        labels_image = hybseq_image * (mask > 0)
        rol_image_check = helpers.check_images_overlap(hybseq_image, labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_nuclei_segmentation.tif', rol_image_check)
        contours, centroids = helpers.find_cellpose_contours(mask)
        print('Nuclei found for position', position, ':', len(contours))

        #breakpoint()

        channels = [1, 2]
        _, _, _, slice_diam = diam_model.eval(position_image, diameter=None, channels=channels, flow_threshold=0.4, cellprob_threshold=0)
        mask, flows, styles = model.eval(position_image, diameter=slice_diam, channels=channels, flow_threshold=0.4, cellprob_threshold=0)
        #mask, flows, styles, diams = model.eval(position_image, diameter=24.12, channels=channels, flow_threshold=0.4, cellprob_threshold=0)

        tif.imwrite(position_output_path + position + '.tif', mask)
        #save slice diam to args.centroids_path + position + 'segemntation_diameter.txt'
        slice_diam = np.round(slice_diam, 2)
        np.savetxt(coordinates_position_path + 'segmentation_diameter.txt', np.array([slice_diam]))

        labels_image = position_image[0] * (mask > 0)
        rol_image_check = helpers.check_images_overlap(position_image[0], labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_geneseq_segmentation.tif', rol_image_check)
        rol_image_check = helpers.check_images_overlap(position_image[1], labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_dapi_segmentation.tif', rol_image_check)
        contours, centroids = helpers.find_cellpose_contours(mask)
        print('Cells found for position', position, ':', len(contours))
        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

    ray.init(num_cpus=args.nr_cpus)
    ray.get([segment_position.remote(position) for position in positions])
    ray.shutdown()

def allocate_rolonies_geneseq(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    cell_id_offset = 0
    for position in positions:
        print('Allocating geneseq rolonies for position', position)
        pos_tic = time.perf_counter()
        position_index = helpers.get_trailing_number(position)

        geneseq_position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_somas_path, position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        coordinates_position_path = helpers.quick_dir(args.proc_coordinates_path, position)
        checks_basecalling_position_path = helpers.quick_dir(args.proc_checks_basecalling_path, position)


        geneseq_image = tif.imread(geneseq_position_path + position + '.tif')
        geneseq_image = np.max(geneseq_image, axis=0)
        #geneseq_image = geneseq_image[3000:3500, 2000:2500]

        mask_cyto = tif.imread(position_output_path + position + '.tif')
        mask_nuclei = tif.imread(position_output_path + position + '_nuclei.tif')
        contours, centroids, mask = helpers.find_cellpose_contours_overlapping_masks(mask_nuclei, mask_cyto)

        #mask = tif.imread(position_output_path + position + '.tif')
        #contours, centroids= helpers.find_cellpose_contours(mask)

        labels_image = geneseq_image * (mask > 0)
        overlap_image = helpers.check_images_overlap(geneseq_image, labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'cyto_nuclei.tif', overlap_image)
        tif.imwrite(position_output_path + position + '_combined.tif', mask)



        cells_df = pd.DataFrame(columns=['cell_ID', 'cell_X', 'cell_Y', 'cell_Z', 'slice'])
        contours_df = pd.DataFrame(columns=['contour'])
        cells_df['genes'] = ''
        cells_df['gene_IDs'] = ''

        rolonies_df = pd.read_csv(genes_position_path + 'rolonies.csv', index_col=[0])
        rolonies = rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        genes_cells = rolonies_df.copy()
        excite_rolonies_df = pd.read_csv(centroids_position_path + 'excite_rolonies.csv', index_col=[0])
        inhib_rolonies_df = pd.read_csv(centroids_position_path + 'inhib_rolonies.csv', index_col=[0])
        IT_rolonies_df = pd.read_csv(centroids_position_path + 'IT_rolonies.csv', index_col=[0])

        excite_rolonies = excite_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        inhib_rolonies = inhib_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        IT_rolonies = IT_rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)

        all_contours_image = np.zeros_like(geneseq_image, dtype=np.int8)
        all_hybseq_rol_image = np.zeros_like(geneseq_image, dtype=np.int8)

        printcounter = 0

        for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
            cells_df.at[cell_id, 'cell_ID'] = cell_id + cell_id_offset
            cells_df.at[cell_id, 'cell_X'] = centroid[0]
            cells_df.at[cell_id, 'cell_Y'] = centroid[1]
            cells_df.at[cell_id, 'cell_Z'] = position_index * 20
            cells_df.at[cell_id, 'genes'] = ''
            cells_df.at[cell_id, 'gene_IDs'] = ''
            cells_df.at[cell_id, 'gene_counts'] = ''
            cells_df.at[cell_id, 'slice'] = position
            cells_df.at[cell_id, 'excite_count'] = ''
            cells_df.at[cell_id, 'inhib_count'] = ''
            cells_df.at[cell_id, 'IT_count'] = ''
            cells_df.at[cell_id, 'excite_IDs'] = ''
            cells_df.at[cell_id, 'inhib_IDs'] = ''
            cells_df.at[cell_id, 'IT_IDs'] = ''
            contours_df.at[cell_id, 'contour'] = contour

            contours_image = labels_image
            centroids_image = np.zeros_like(labels_image, dtype=np.int8)
            cv2.drawContours(image=contours_image, contours=[contour], contourIdx=-1, color=120, thickness=1)
            center = tuple([int(centroid[1]), int(centroid[0])])
            cv2.circle(centroids_image, center, 15, 200, -1)

            if printcounter == 10000:
                print('cell id is', cell_id, 'out of', len(centroids))
                printcounter = 0
            printcounter += 1

            cell_color = np.random.randint(255)
            cv2.drawContours(image=all_contours_image, contours=[contour], contourIdx=-1, color=cell_color, thickness=1)
            cv2.drawContours(image=all_hybseq_rol_image, contours=[contour], contourIdx=-1, color=cell_color, thickness=1)



            relevant_rows = np.where((rolonies[:, 1] < centroid[0] + 20) & (rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = genes_cells[genes_cells['rol_ID'] == real_rol_ID].index[0]
                        genes_cells.at[real_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        genes_cells.at[real_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        genes_cells.at[real_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        genes_cells.at[real_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        genes_cells.at[real_rol_ID, 'cell_alloc'] = 1
                        genes_cells.at[real_rol_ID, 'slice'] = position
                        if cells_df.at[cell_id, 'genes'] == '':
                            cells_df.at[cell_id, 'genes'] = rolonies_df.at[real_rol_ID, 'gene']
                            cells_df.at[cell_id, 'gene_IDs'] = rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                            cells_df.at[cell_id, 'gene_counts'] = '1'
                        else:
                            #pull cells_df.at[cell_id, 'genes'], cells_df.at[cell_id, 'gene_IDs'] and cells_df.at[cell_id, 'gene_counts']into lists.
                            #check whether rolonies_df.at[real_rol_ID, 'gene'] and rolonies_df.at[real_rol_ID, 'geneID'].astype(str) exist in lists.
                            #if they do, increment corresponding element of gene_counts
                            # if they don't, append to lists
                            genes = cells_df.at[cell_id, 'genes'].split(',')
                            gene_IDs = cells_df.at[cell_id, 'gene_IDs'].split(',')
                            gene_counts = cells_df.at[cell_id, 'gene_counts'].split(',')

                            if rolonies_df.at[real_rol_ID, 'gene'] in genes:
                                gene_index = genes.index(rolonies_df.at[real_rol_ID, 'gene'])
                                gene_counts[gene_index] = str(int(gene_counts[gene_index]) + 1)
                            else:
                                genes.append(rolonies_df.at[real_rol_ID, 'gene'])
                                gene_IDs.append(str(rolonies_df.at[real_rol_ID, 'geneID']))
                                gene_counts.append('1')
                            cells_df.at[cell_id, 'genes'] = ','.join(genes)
                            cells_df.at[cell_id, 'gene_IDs'] = ','.join(gene_IDs)
                            cells_df.at[cell_id, 'gene_counts'] = ','.join(gene_counts)

                            #cells_df.at[cell_id, 'genes'] += ',' + rolonies_df.at[real_rol_ID, 'gene']
                            #cells_df.at[cell_id, 'gene_IDs'] += ',' + rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                            #cells_df.at[cell_id, 'gene_IDs'] += ',' + 1
                        cv2.circle(all_contours_image, rolony_centroid, 2, color=cell_color, thickness=-1)
            except IndexError:
                continue

            # do the same for excitatory cells
            relevant_rows = np.where((excite_rolonies[:, 1] < centroid[0] + 20) & (excite_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = excite_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]

                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        # find index of real_rol_ID in excite_rolonies_df, column rol_ID
                        row_index_rol_ID = excite_rolonies_df[excite_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        excite_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1

                        if cells_df.at[cell_id, 'excite_count'] == '':
                            cells_df.at[cell_id, 'excite_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'excite_count'] = '1'
                        else:
                            excite_IDs = cells_df.at[cell_id, 'excite_IDs'].split(',')
                            excite_count = cells_df.at[cell_id, 'excite_count']

                            # add the next excite ID and increment excite_count
                            excite_IDs.append(str(real_rol_ID))
                            excite_count = str(int(excite_count) + 1)

                            cells_df.at[cell_id, 'excite_IDs'] = ','.join(excite_IDs)
                            cells_df.at[cell_id, 'excite_count'] = excite_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, radius=4, color=250, thickness=-1)

            except IndexError:
                continue

            # do the same for inhibitory cells
            relevant_rows = np.where((inhib_rolonies[:, 1] < centroid[0] + 20) & (inhib_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = inhib_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]

                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = inhib_rolonies_df[inhib_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        inhib_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1
                        if cells_df.at[cell_id, 'inhib_count'] == '':
                            cells_df.at[cell_id, 'inhib_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'inhib_count'] = '1'
                        else:
                            inhib_IDs = cells_df.at[cell_id, 'inhib_IDs'].split(',')
                            inhib_count = cells_df.at[cell_id, 'inhib_count']

                            # add the next excite ID and increment excite_count
                            inhib_IDs.append(str(real_rol_ID))
                            inhib_count = str(int(inhib_count) + 1)

                            cells_df.at[cell_id, 'inhib_IDs'] = ','.join(inhib_IDs)
                            cells_df.at[cell_id, 'inhib_count'] = inhib_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, 4, color=100, thickness=2)

            except IndexError:
                continue

            # do the same for IT cells
            relevant_rows = np.where((IT_rolonies[:, 1] < centroid[0] + 20) & (IT_rolonies[:, 1] > centroid[0] - 20))
            relevant_rolonies = IT_rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        row_index_rol_ID = IT_rolonies_df[IT_rolonies_df['rol_ID'] == real_rol_ID].index[0]
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        IT_rolonies_df.at[row_index_rol_ID, 'cell_alloc'] = 1
                        if cells_df.at[cell_id, 'IT_count'] == '':
                            cells_df.at[cell_id, 'IT_IDs'] = str(real_rol_ID)
                            cells_df.at[cell_id, 'IT_count'] = '1'
                        else:
                            IT_IDs = cells_df.at[cell_id, 'IT_IDs'].split(',')
                            IT_count = cells_df.at[cell_id, 'IT_count']

                            # add the next excite ID and increment excite_count
                            IT_IDs.append(str(real_rol_ID))
                            IT_count = str(int(IT_count) + 1)

                            cells_df.at[cell_id, 'IT_IDs'] = ','.join(IT_IDs)
                            cells_df.at[cell_id, 'IT_count'] = IT_count
                        cv2.circle(all_hybseq_rol_image, rolony_centroid, 2, color=175, thickness=-1)

            except IndexError:
                continue


        contours_image_check = helpers.check_images_overlap(geneseq_image, all_contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_geneseq_rolony_allocation.tif', contours_image_check)

        genes_cells.to_csv(genes_position_path + 'geneseq_rolonies.csv')

        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')

        inhib_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_inhib_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'inhib_rolony_allocation_geneseq.tif', inhib_image_check)

        excite_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_excite_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'excite_rolony_allocation_geneseq.tif', excite_image_check)

        IT_image_check = np.stack((helpers.scale_to_8bit(hybseq_image[args.hybseq_IT_channel], unsigned=False), all_hybseq_rol_image), axis=0)
        tif.imwrite(checks_segmentation_position_path + 'IT_rolony_allocation_geneseq.tif', IT_image_check)

        excite_rolonies_df.to_csv(centroids_position_path + 'excite_geneseq.csv')
        inhib_rolonies_df.to_csv(centroids_position_path + 'inhib_geneseq.csv')
        IT_rolonies_df.to_csv(centroids_position_path + 'IT_geneseq.csv')

        contours_image_check = helpers.check_images_overlap(geneseq_image, contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_contours.tif', contours_image_check)

        cells_df.to_csv(genes_position_path + 'geneseq_cells.csv')
        contours_df.to_csv(centroids_position_path + 'geneseq_contours.csv')

        cell_id_offset += len(contours)
        alloc_rol_ids = genes_cells.index[genes_cells['cell_alloc'] == 1].tolist()
        print('nr rolonies allocated is', len(alloc_rol_ids))
        pos_toc = time.perf_counter()
        print('Geneseq rolony allocation for position', position, 'done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def assign_classes_hybseq(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    for position in positions:
        print('Assigning hybseq rolonies for position', position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)

        cells_dfs_names = ['geneseq_cells.csv', 'funseq_cells.csv']

        for cells_df_name in cells_dfs_names:
            if os.path.exists(genes_position_path + cells_df_name):
                cells_df = pd.read_csv(genes_position_path + cells_df_name, index_col=0)
                #if cells_df is not empty
                if len(cells_df) > 0:
                    #if values in excite_count, inhib_count and ID_count are empty, give them value 0
                    for index, row in cells_df.iterrows():
                        if pd.isnull(row['excite_count']):
                            cells_df.at[index, 'excite_count'] = 0
                        if pd.isnull(row['inhib_count']):
                            cells_df.at[index, 'inhib_count'] = 0
                        if pd.isnull(row['IT_count']):
                            cells_df.at[index, 'IT_count'] = 0

                    for index, row in cells_df.iterrows():
                        if row['excite_count'] < 2 and row['inhib_count'] < 2 and row['IT_count'] < 2:
                            cells_df.at[index, 'hyb_class'] = 'unassigned'
                            cells_df.at[index, 'hyb_subclass'] = 'unassigned'
                        elif row['excite_count'] > 3 and row['inhib_count'] > 3:
                            cells_df.at[index, 'hyb_class'] = 'unassigned'
                            cells_df.at[index, 'hyb_subclass'] = 'unassigned'
                        elif row['excite_count'] > row['inhib_count']:
                            cells_df.at[index, 'hyb_class'] = 'excitatory'
                            if row['IT_count'] > 1:
                                cells_df.at[index, 'hyb_subclass'] = 'IT'
                            else:
                                cells_df.at[index, 'hyb_subclass'] = 'unassigned'
                        elif row['inhib_count'] > row['excite_count']:
                            cells_df.at[index, 'hyb_class'] = 'inhibitory'
                            cells_df.at[index, 'hyb_subclass'] = 'unassigned'
                        elif row['excite_count'] == row['inhib_count']:
                            cells_df.at[index, 'hyb_class'] = 'unassigned'
                            cells_df.at[index, 'hyb_subclass'] = 'unassigned'
                        else:
                            cells_df.at[index, 'hyb_class'] = 'unassigned'
                            cells_df.at[index, 'hyb_subclass'] = 'unassigned'

                    cells_df.to_csv(genes_position_path + cells_df_name)
                    print(cells_df_name)
                    print(cells_df.head())
                    # print how many excitatory, inhibitory and unassigned cells are
                    # if there is 'excitatory' in hyb_class column, print how many excitatory cells are
                    if 'excitatory' in cells_df['hyb_class'].values:
                        nr_excitatory_cells = cells_df['hyb_class'].value_counts()['excitatory']
                    else:
                        nr_excitatory_cells = 0
                    if 'inhibitory' in cells_df['hyb_class'].values:
                        nr_inhibitory_cells = cells_df['hyb_class'].value_counts()['inhibitory']
                    else:
                        nr_inhibitory_cells = 0
                    if 'unassigned' in cells_df['hyb_class'].values:
                        nr_unassigned_cells = cells_df['hyb_class'].value_counts()['unassigned']
                    else:
                        nr_unassigned_cells = 0
                    print('For position ', position, 'table ', cells_df_name, ' there are ', nr_excitatory_cells, ' excitatory cells, ', nr_inhibitory_cells, ' inhibitory cells and ', nr_unassigned_cells, ' unassigned cells.')




def generate_final_tables(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_genes_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    funseq_positions = [pos for pos in positions if pos in args.funseq_positions]
    sample_funseq_cells_df = pd.read_csv(args.proc_genes_path + str(funseq_positions[0]) + '/funseq_cells.csv')
    all_funseq_cells_df = sample_funseq_cells_df.iloc[:0, :].copy()

    sample_funseq_rolonies_df = pd.read_csv(args.proc_genes_path + str(funseq_positions[0]) + '/funseq_rolonies.csv')
    all_funseq_rolonies_df = sample_funseq_rolonies_df.iloc[:0, :].copy()

    sample_barcoded_cells_df = pd.read_csv(args.proc_genes_path + str(funseq_positions[0]) + '/barcoded_cells.csv')
    all_barcoded_cells_df = sample_barcoded_cells_df.iloc[:0, :].copy()

    sample_geneseq_cells_df = pd.read_csv(args.proc_genes_path + str(positions[0]) + '/geneseq_cells.csv')
    all_geneseq_cells_df = sample_geneseq_cells_df.iloc[:0, :].copy()

    sample_geneseq_rolonies_df = pd.read_csv(args.proc_genes_path + str(positions[0]) + '/geneseq_rolonies.csv')
    all_geneseq_rolonies_df = sample_geneseq_rolonies_df.iloc[:0, :].copy()

    slices_funseq_cells = []
    slices_geneseq_cells = []
    slices_barcoded_cells = []
    slices_geneseq_rolonies = []
    slices_geneseq_allocated_rolonies = []
    geneseq_excitatory_cells = []
    geneseq_inhibitory_cells = []
    geneseq_unassigned_cells = []
    funseq_excitatory_cells = []
    funseq_inhibitory_cells = []
    funseq_unassigned_cells = []
    print('Generating final tables')
    for position in args.positions:
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        geneseq_cells_df = pd.read_csv(genes_position_path + 'geneseq_cells.csv', index_col=[0])
        all_geneseq_cells_df = all_geneseq_cells_df.append(geneseq_cells_df)
        slices_geneseq_cells.append(len(geneseq_cells_df))
        if len(geneseq_cells_df) > 0:
            if 'excitatory' in geneseq_cells_df['hyb_class'].values:
                geneseq_excitatory_cells.append(geneseq_cells_df['hyb_class'].value_counts()['excitatory'])
            else:
                geneseq_excitatory_cells.append(0)
            if 'inhibitory' in geneseq_cells_df['hyb_class'].values:
                geneseq_inhibitory_cells.append(geneseq_cells_df['hyb_class'].value_counts()['inhibitory'])
            else:
                geneseq_inhibitory_cells.append(0)
            if 'unassigned' in geneseq_cells_df['hyb_class'].values:
                geneseq_unassigned_cells.append(geneseq_cells_df['hyb_class'].value_counts()['unassigned'])
            else:
                geneseq_unassigned_cells.append(0)
        else:
            geneseq_excitatory_cells.append(0)
            geneseq_inhibitory_cells.append(0)
            geneseq_unassigned_cells.append(0)


        geneseq_rolonies_df = pd.read_csv(genes_position_path + 'geneseq_rolonies.csv', index_col=[0])
        all_geneseq_rolonies_df = all_geneseq_rolonies_df.append(geneseq_rolonies_df)
        slices_geneseq_rolonies.append(len(geneseq_rolonies_df))
        alloc_rol_ids = geneseq_rolonies_df.index[geneseq_rolonies_df['cell_alloc'] == 1].tolist()
        slices_geneseq_allocated_rolonies.append(len(alloc_rol_ids))

    for position in funseq_positions:
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)

        funseq_cells_df = pd.read_csv(genes_position_path + 'funseq_cells.csv', index_col=[0])
        all_funseq_cells_df = all_funseq_cells_df.append(funseq_cells_df)
        slices_funseq_cells.append(len(funseq_cells_df))
        if len(funseq_cells_df) > 0:
            if 'excitatory' in funseq_cells_df['hyb_class'].values:
                funseq_excitatory_cells.append(funseq_cells_df['hyb_class'].value_counts()['excitatory'])
            else:
                funseq_excitatory_cells.append(0)
            if 'inhibitory' in funseq_cells_df['hyb_class'].values:
                funseq_inhibitory_cells.append(funseq_cells_df['hyb_class'].value_counts()['inhibitory'])
            else:
                funseq_inhibitory_cells.append(0)
            if 'unassigned' in funseq_cells_df['hyb_class'].values:
                funseq_unassigned_cells.append(funseq_cells_df['hyb_class'].value_counts()['unassigned'])
            else:
                funseq_unassigned_cells.append(0)
        else:
            funseq_excitatory_cells.append(0)
            funseq_inhibitory_cells.append(0)
            funseq_unassigned_cells.append(0)

        funseq_rolonies_df = pd.read_csv(genes_position_path + 'funseq_rolonies.csv', index_col=[0])
        all_funseq_rolonies_df = all_funseq_rolonies_df.append(funseq_rolonies_df)

        barcoded_cells_df = pd.read_csv(genes_position_path + 'barcoded_cells.csv', index_col=[0])
        all_barcoded_cells_df = all_barcoded_cells_df.append(barcoded_cells_df)
        slices_barcoded_cells.append(len(barcoded_cells_df))



    # if there is an Unnamed column, remove it
    if 'Unnamed: 0' in all_geneseq_cells_df.columns:
        all_geneseq_cells_df = all_geneseq_cells_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in all_geneseq_rolonies_df.columns:
        all_geneseq_rolonies_df = all_geneseq_rolonies_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in all_funseq_rolonies_df.columns:
        all_funseq_rolonies_df = all_funseq_rolonies_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in all_funseq_cells_df.columns:
        all_funseq_cells_df = all_funseq_cells_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in all_barcoded_cells_df.columns:
        all_barcoded_cells_df = all_barcoded_cells_df.drop(columns=['Unnamed: 0'])


    #for all_funseq_cells_df, all_barcoded_cells_df, all_geneseq_cells_df:
    # 1. add a column with gene_counts. this should be the number of occurrences of each number in the gene_IDs column. each cell is a float


    all_funseq_cells_df.to_csv(args.matching_tables_path + 'all_funseq_cells.csv', index=False)
    all_funseq_cells_df.to_csv(args.proc_summary_path + 'all_funseq_cells.csv', index=False)
    funseq_slices_centroids = all_funseq_cells_df[['cell_Z', 'cell_X', 'cell_Y', 'cell_ID']].to_numpy()
    funseq_slices_centroids[:, 0] = funseq_slices_centroids[:, 0] / args.slice_thickness
    np.savetxt(args.matching_centroids_path + 'funseq_slices_centroids.txt', funseq_slices_centroids, fmt='%i')

    all_barcoded_cells_df.to_csv(args.matching_tables_path + 'all_barcoded_cells.csv', index=False)
    all_barcoded_cells_df.to_csv(args.proc_summary_path + 'all_barcoded_cells.csv', index=False)
    barcoded_slices_centroids = all_barcoded_cells_df[['cell_Z', 'cell_X', 'cell_Y', 'cell_ID']].to_numpy()
    barcoded_slices_centroids[:, 0] = barcoded_slices_centroids[:, 0] / args.slice_thickness
    np.savetxt(args.matching_centroids_path + 'barcoded_slices_centroids.txt', barcoded_slices_centroids, fmt='%i')

    all_geneseq_cells_df.to_csv(args.matching_tables_path + 'all_geneseq_cells.csv', index=False)
    all_geneseq_cells_df.to_csv(args.proc_summary_path + 'all_geneseq_cells.csv', index=False)
    geneseq_slices_centroids = all_geneseq_cells_df[['cell_Z', 'cell_X', 'cell_Y', 'cell_ID']].to_numpy()
    geneseq_slices_centroids[:, 0] = geneseq_slices_centroids[:, 0] / args.slice_thickness
    np.savetxt(args.matching_centroids_path + 'geneseq_slices_centroids.txt', geneseq_slices_centroids, fmt='%i')

    all_geneseq_rolonies_df.to_csv(args.matching_tables_path + 'all_geneseq_rolonies.csv')
    all_geneseq_rolonies_df.to_csv(args.proc_summary_path + 'all_geneseq_rolonies.csv')

    all_funseq_rolonies_df.to_csv(args.matching_tables_path + 'all_funseq_rolonies.csv')
    all_funseq_rolonies_df.to_csv(args.proc_summary_path + 'all_funseq_rolonies.csv')


    print('rol allocated', slices_geneseq_allocated_rolonies)
    print('geneseq cells', slices_geneseq_cells)
    print('slices geneseq', slices_geneseq_rolonies)
    with open(args.proc_summary_path + 'funseq_stats.txt', 'w') as f:
        f.write('Total number of infected cells: ' + str(len(all_barcoded_cells_df)))
        f.write('\n')
        f.write('Total number of cells with projections too: ' + str(len(all_funseq_cells_df)))
        f.write('\n')
        f.write('\n')
        for id, position in enumerate(funseq_positions):
            f.write('\n')
            f.write('For slice ' + position)
            f.write('\n')
            f.write( 'Cells - ' + str(slices_barcoded_cells[id]) + ' infected, ' + str(slices_funseq_cells[id]) + ' with projections')
            f.write('\n')
            f.write('Excitatory, inhibitory, and unassigned cells: ' + str(funseq_excitatory_cells[id]) + ', ' + str(funseq_inhibitory_cells[id]) + ', ' + str(funseq_unassigned_cells[id]))
            f.write('\n')
            f.write('Percentage of excitatory and inhibitory cells: ' + str(round(100*funseq_excitatory_cells[id] / (0.001 + funseq_excitatory_cells[id] + funseq_inhibitory_cells[id]))) + '%, ' + str(round(100*funseq_inhibitory_cells[id] / (0.001 +funseq_excitatory_cells[id] + funseq_inhibitory_cells[id]))) + '%')
    with open(args.proc_summary_path + 'geneseq_stats.txt', 'w') as f:
        f.write('Total number of sequenced cells: ' + str(np.sum(slices_geneseq_cells)))
        f.write('\n')
        f.write('Total number of rolonies identified: ' + str(len(all_geneseq_rolonies_df)))
        f.write('\n')
        f.write('Total number of rolonies allocated to cells: ' + str(np.sum(slices_geneseq_allocated_rolonies)))
        f.write('\n')
        f.write('Percentage allocated rolonies -> ' + str(round(100*np.sum(slices_geneseq_allocated_rolonies)/len(all_geneseq_rolonies_df), 1)))
        f.write('\n')
        f.write('\n')
        for id, position in enumerate(funseq_positions):
            f.write('\n')
            f.write('For slice ' + position)
            f.write('\n')
            f.write( 'Cells - ' + str(slices_geneseq_cells[id]) + ' sequenced, ')
            f.write('\n')
            f.write('Excitatory, inhibitory, and unassigned cells: ' + str(geneseq_excitatory_cells[id]) + ', ' + str(geneseq_inhibitory_cells[id]) + ', ' + str(geneseq_unassigned_cells[id]))
            f.write('\n')
            f.write('Percentage excitatory and inhibitory cells -> ' + str(round(100*geneseq_excitatory_cells[id]/(0.001 + geneseq_excitatory_cells[id] + geneseq_inhibitory_cells[id]))) + '%, ' + str(round(100*geneseq_inhibitory_cells[id]/(0.001 + geneseq_excitatory_cells[id] + geneseq_inhibitory_cells[id]))) + '%')
            f.write('\n')
            f.write( 'Rolonies - ' + str(slices_geneseq_rolonies[id]) + ' identified, '
                    + str(slices_geneseq_allocated_rolonies[id]) + ' allocated -> '
                     + str(round(100*slices_geneseq_allocated_rolonies[id]/slices_geneseq_rolonies[id], 1)) + '%')
            f.write('\n')


def preprocess_functional(args):
    for i, scan_id in enumerate(args.scan_idx):
        MEIs_scan_path = helpers.quick_dir(args.matching_MEIs_path, 'scan_' + str(scan_id))
        weights_scan_path = helpers.quick_dir(args.matching_weights_path, 'scan_' + str(scan_id))
        properties_scan_path = helpers.quick_dir(args.matching_properties_path, 'scan_' + str(scan_id))
        pickle_file = pickle.load(open(args.scan_path + args.functional_data_tables[i], 'rb'))
        nr_cells = len(pickle_file['MEI'])
        #nr_cells = 10
        for j in range(nr_cells):
            filename = str(pickle_file['unit_id'][j])
            tif.imwrite(MEIs_scan_path + filename + '.tif', pickle_file['MEI'][j])
            rw0 = pickle_file['readout_weights_0'][j]
            rw1 = pickle_file['readout_weights_1'][j]
            rw2 = pickle_file['readout_weights_2'][j]
            rw3 = pickle_file['readout_weights_3'][j]
            # stack the 4 matrices as (4, n, n)
            rw = np.stack((rw0, rw1, rw2, rw3), axis=0)
            np.save(weights_scan_path + filename + '.npy', rw)
        #save pickle file as csv without readout weights and MEI fields
        pickle_file.pop('readout_weights_0')
        pickle_file.pop('readout_weights_1')
        pickle_file.pop('readout_weights_2')
        pickle_file.pop('readout_weights_3')
        pickle_file.pop('MEI')
        pickle_file_df = pd.DataFrame(pickle_file)
        pickle_file_df.to_csv(properties_scan_path + 'properties.csv', index=False)

def inspect_cells(args, cell_IDs):

    cell_check_path = helpers.quick_dir(args.dataset_path, 'cell_check')
    genebook = pd.read_csv(args.matching_tables_path + 'genebook.csv')
    geneseq_rolonies = pd.read_csv(args.matching_tables_path + 'all_geneseq_rolonies.csv')


    positions = []

    for cell_ID in cell_IDs:
        geneseq_cells_df = pd.read_csv(args.matching_tables_path + 'all_geneseq_cells.csv')
        funseq_cells_df = pd.read_csv(args.matching_tables_path + 'all_funseq_cells.csv')
        geneseq_cell = geneseq_cells_df[geneseq_cells_df['cell_ID'] == cell_ID]
        assert len(geneseq_cell) == 1, 'Cell ID appears only once in the geneseq table'
        position = geneseq_cell['slice'].values[0]
        positions.append(position)

    #create a dictionary where the keys are the unique positions and the values are the cell IDs
    positions_dict = {}
    unique_position = np.unique(positions)
    for position in unique_position:
        positions_dict[position] = []
        for i in range(len(cell_IDs)):
            if positions[i] == position:
                positions_dict[position].append(cell_IDs[i])
    print('Displaying cells from dictionary: ', positions_dict)
    for position in positions_dict.keys():
        geneseq_position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        segmented_position_path = helpers.quick_dir(args.proc_segmented_somas_path, position)

        geneseq_image = tif.imread(geneseq_position_path + position + '.tif')
        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')
        segmented_image = tif.imread(segmented_position_path + position + '.tif')

        for cell_ID in positions_dict[position]:
            cell_output_path = helpers.quick_dir(cell_check_path, str(cell_ID))
            geneseq_cell_df = geneseq_cells_df[geneseq_cells_df['cell_ID'] == cell_ID]
            cell_X, cell_Y = geneseq_cell_df['cell_X'].values[0], geneseq_cell_df['cell_Y'].values[0]
            x1, x2, y1, y2 = cell_X - 50, cell_X + 50, cell_Y - 50, cell_Y + 50
            gene_IDs = geneseq_cell_df['gene_IDs'].values[0].split(',')
            rol_IDs = geneseq_rolonies[geneseq_rolonies['cell_ID'] == cell_ID]['rol_ID'].values
            # genes = geneseq_rolonies[geneseq_rolonies['cell_ID'] == cell_ID]['gene'].values
            # print('genes: ', genes)
            # genes_found = geneseq_cell_df['genes'].values
            # print('genes_found: ', genes_found)


            geneseq_cell_int16 = geneseq_image[:, x1:x2, y1:y2]
            hybseq_cell = hybseq_image[:, x1:x2, y1:y2]
            segmented_cell = segmented_image[x1:x2, y1:y2]
            #get contours in segmented_cell
            contours = measure.find_contours(segmented_cell, 0.5)

            # draw contours in all planes in geneseq_cell
            geneseq_cell = np.zeros_like(geneseq_cell_int16, dtype=np.int8)
            for plane in range(geneseq_cell.shape[0]):
                geneseq_cell[plane] = helpers.scale_to_8bit(geneseq_cell_int16[plane], unsigned=False)
                for contour in contours:
                    for i in range(len(contour)):
                        geneseq_cell[plane, int(contour[i][0]), int(contour[i][1])] = 255

            tif.imwrite(cell_output_path + 'geneseq_cell.tif', geneseq_cell)
            tif.imwrite(cell_output_path + 'segmented_cell.tif', segmented_cell)
            nr_channels, nr_cycles = 4, 7

            # create a 4x7 plot with the 4 channels and 7 cycles of geneseq_cell
            fig, axs = plt.subplots(nr_channels, nr_cycles, figsize=(nr_cycles*2, nr_channels*2))
            for i in range(nr_channels):
                for j in range(nr_cycles):
                    axs[i, j].imshow(geneseq_cell[i*nr_cycles + j], cmap='gray')
                    axs[i, j].axis('off')
            fig.savefig(cell_output_path + 'plot_geneseq_cell.png')









def matchers_matching(args):

    matcher_name = helpers.create_matcher_profile()
    matching_matched_path = helpers.quick_dir(args.matching_path, 'matcher_' + matcher_name)

    bv_slices = tif.imread(args.matching_images_path + 'blood_vessels.tif')
    somas_slices = tif.imread(args.matching_images_path + 'somas_blood_vessels.tif')
    #bv_somas_slices = tif.imread(args.matching_images_path + 'gfp_somas_blood_vessels.tif')
    genes_slices = tif.imread(args.matching_images_path + 'genes.tif')
    barcoded = tif.imread(args.matching_images_path + 'barcoded.tif')

    funseq_slices = np.loadtxt(args.matching_centroids_path + 'funseq_slices_centroids.txt')
    barcoded_slices = np.loadtxt(args.matching_centroids_path + 'barcoded_slices_centroids.txt')
    geneseq_slices = np.loadtxt(args.matching_centroids_path + 'geneseq_slices_centroids.txt')

    funseq_slices[:, [1, 2]] = funseq_slices[:, [2, 1]]
    barcoded_slices[:, [1, 2]] = barcoded_slices[:, [2, 1]]
    geneseq_slices[:, [1, 2]] = geneseq_slices[:, [2, 1]]

    slices_image = np.stack((bv_slices, somas_slices, genes_slices, barcoded), axis=0)
    bv_invivo_stack = tif.imread(args.matching_images_path + args.proc_bv_stack_name)
    gfp_invivo_stack = tif.imread(args.matching_images_path + args.proc_gfp_stack_name)
    stack_image = np.stack((bv_invivo_stack, gfp_invivo_stack), axis=0)

    padded_funseq_stack = np.loadtxt(args.matching_centroids_path + 'padded_func_centroids.txt')
    padded_funseq_stack[:, [1, 2]] = padded_funseq_stack[:, [2, 1]]
    padded_struct_stack = np.loadtxt(args.matching_centroids_path + 'padded_struct_centroids.txt')
    padded_struct_stack[:, [1, 2]] = padded_struct_stack[:, [2, 1]]


    if os.path.isfile(matching_matched_path + 'plotted_funseq_stack.npy'):
        plotted_funseq_stack = np.load(matching_matched_path + 'plotted_funseq_stack.npy')
        plotted_funseq_slices = np.load(matching_matched_path + 'plotted_funseq_slices.npy')
        plotted_matched_funseq_stack = np.load(matching_matched_path + 'matched_funseq_stack.npy')
        plotted_matched_funseq_slices = np.load(matching_matched_path + 'matched_funseq_slices.npy')
    else:
        plotted_funseq_stack = padded_funseq_stack.copy()
        plotted_funseq_slices = funseq_slices
        plotted_matched_funseq_stack = np.empty((0, 3))
        plotted_matched_funseq_slices = np.empty((0, 3))

    if os.path.isfile(matching_matched_path + 'plotted_geneseq_stack.npy'):
        plotted_geneseq_stack = np.load(matching_matched_path + 'plotted_geneseq_stack.npy')
        plotted_geneseq_slices = np.load(matching_matched_path + 'plotted_geneseq_slices.npy')
        plotted_matched_geneseq_stack = np.load(matching_matched_path + 'matched_geneseq_stack.npy')
        plotted_matched_geneseq_slices = np.load(matching_matched_path + 'matched_geneseq_slices.npy')
    else:
        plotted_geneseq_stack = padded_funseq_stack.copy()
        plotted_geneseq_slices = geneseq_slices
        plotted_matched_geneseq_stack = np.empty((0, 3))
        plotted_matched_geneseq_slices = np.empty((0, 3))

    #add centroids to stack viewer - in vivo centroids


    stack_viewer = napari.view_image(stack_image, title='In vivo stack')
    stack_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    stack_viewer.add_points(padded_struct_stack[:, :3], face_color="red", size=10, name='struct_centroids', symbol='disc', opacity=0.7, ndim=3)

    stack_viewer.add_points(plotted_matched_funseq_stack, face_color="blue", size=10, name='matched_funseq', symbol='disc', opacity=0.7, ndim=3)
    stack_viewer.add_points(plotted_funseq_stack[:, :3], face_color="green", size=10, name='funseq', symbol='disc', opacity=0.7, ndim=3)

    stack_viewer.add_points(plotted_matched_geneseq_stack, face_color="blue", size=10, name='matched_geneseq', symbol='disc', opacity=0.7, ndim=3)
    stack_viewer.add_points(plotted_geneseq_stack[:, :3], face_color="pink", size=10, name='geneseq', symbol='disc', opacity=0.7, ndim=3)

    #add centroids to slices viewer - in vitro centroids
    slice_viewer = napari.view_image(slices_image, title='In vitro slices')
    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    slice_viewer.add_points(barcoded_slices[:, :3], face_color="red", size=30, name='barcoded', symbol='disc', opacity=0.7, ndim=3)

    slice_viewer.add_points(plotted_matched_funseq_slices, face_color="blue", size=30, name='matched_funseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.add_points(plotted_funseq_slices[:, :3], face_color="green", size=30, name='funseq', symbol='disc', opacity=0.7, ndim=3)

    slice_viewer.add_points(plotted_matched_geneseq_slices, face_color="blue", size=30, name='matched_geneseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.add_points(plotted_geneseq_slices[:, :3], face_color="pink", size=30, name='geneseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.layers['matched_geneseq'].mode = 'add'


    @stack_viewer.bind_key('q')
    def save_funseq_matches(stack_viewer):
        selected_ids_stack = stack_viewer.layers['funseq'].selected_data
        selected_ids_slices = slice_viewer.layers['funseq'].selected_data

        plotted_matched_funseq_stack = stack_viewer.layers['matched_funseq'].data
        plotted_matched_funseq_slices = slice_viewer.layers['matched_funseq'].data

        selected_ids_slices = list(selected_ids_slices)
        selected_ids_stack = list(selected_ids_stack)

        cell_id_slices = plotted_funseq_slices[selected_ids_slices, 3]
        ids_slices = np.array(cell_id_slices)
        ids_slices = ids_slices.reshape((-1, 1))

        cell_id_stack = plotted_funseq_stack[selected_ids_stack, 3]
        ids_stack = np.array(cell_id_stack)
        ids_stack = ids_stack.reshape((-1, 1))

        matches_to_add = np.hstack((ids_slices, ids_stack))
        if os.path.isfile(matching_matched_path + 'funseq_matches.txt'):
            matches = np.loadtxt(matching_matched_path + 'funseq_matches.txt')
        else:
            matches = np.empty((0, 2))
        matches = np.vstack((matches, matches_to_add))
        np.savetxt(matching_matched_path + 'funseq_matches.txt', matches, fmt='%i')
        print('total number of funseq matches:', len(matches))

        for (id_stack, id_slices) in zip(selected_ids_stack, selected_ids_slices):
            for i in range(-10, 10):
                temp_id = plotted_funseq_stack[id_stack, 3]
                if plotted_funseq_stack[id_stack + i, 3] == temp_id:
                    plotted_matched_funseq_stack = np.vstack((plotted_matched_funseq_stack, plotted_funseq_stack[id_stack + i, :3]))
                    plotted_funseq_stack[id_stack + i, :3] = [0, 0, 0]
            plotted_matched_funseq_slices = np.vstack((plotted_matched_funseq_slices, plotted_funseq_slices[id_slices, :3]))
            plotted_funseq_slices[id_slices, :3] = [0, 0, 0]

        stack_viewer.layers['matched_funseq'].data = plotted_matched_funseq_stack[:, :3]
        slice_viewer.layers['matched_funseq'].data = plotted_matched_funseq_slices[:, :3]

        stack_viewer.layers['funseq'].data = plotted_funseq_stack[:, :3]
        slice_viewer.layers['funseq'].data = plotted_funseq_slices[:, :3]

        np.savetxt(matching_matched_path + 'plotted_funseq_stack.txt', plotted_funseq_stack, fmt='%i')
        np.save(matching_matched_path + 'plotted_funseq_stack.npy', plotted_funseq_stack)

        np.savetxt(matching_matched_path + 'plotted_funseq_slices.txt', plotted_funseq_slices, fmt='%i')
        np.save(matching_matched_path + 'plotted_funseq_slices.npy', plotted_funseq_slices)

        np.savetxt(matching_matched_path + 'matched_funseq_stack.txt', plotted_matched_funseq_stack, fmt='%i')
        np.save(matching_matched_path + 'matched_funseq_stack.npy', plotted_matched_funseq_stack)

        np.savetxt(matching_matched_path + 'matched_funseq_slices.txt', plotted_matched_funseq_slices, fmt='%i')
        np.save(matching_matched_path + 'matched_funseq_slices.npy', plotted_matched_funseq_slices)

    @stack_viewer.bind_key('e')
    def save_geneseq_matches(stack_viewer):
        selected_ids_stack = stack_viewer.layers['geneseq'].selected_data
        selected_ids_slices = slice_viewer.layers['geneseq'].selected_data

        plotted_matched_geneseq_stack = stack_viewer.layers['matched_geneseq'].data
        plotted_matched_geneseq_slices = slice_viewer.layers['matched_geneseq'].data

        selected_ids_slices = list(selected_ids_slices)
        selected_ids_stack = list(selected_ids_stack)

        cell_id_slices = plotted_geneseq_slices[selected_ids_slices, 3]
        ids_slices = np.array(cell_id_slices)
        ids_slices = ids_slices.reshape((-1, 1))

        cell_id_stack = plotted_geneseq_stack[selected_ids_stack, 3]
        ids_stack = np.array(cell_id_stack)
        ids_stack = ids_stack.reshape((-1, 1))

        matches_to_add = np.hstack((ids_slices, ids_stack))
        if os.path.isfile(matching_matched_path + 'geneseq_matches.txt'):
            matches = np.loadtxt(matching_matched_path + 'geneseq_matches.txt')
        else:
            matches = np.empty((0, 2))
        matches = np.vstack((matches, matches_to_add))
        np.savetxt(matching_matched_path + 'geneseq_matches.txt', matches, fmt='%i')
        print('total number of geneseq matches:', len(matches))

        for (id_stack, id_slices) in zip(selected_ids_stack, selected_ids_slices):
            for i in range(-10, 10):
                temp_id = plotted_geneseq_stack[id_stack, 3]
                if plotted_geneseq_stack[id_stack + i, 3] == temp_id:
                    plotted_matched_geneseq_stack = np.vstack((plotted_matched_geneseq_stack, plotted_geneseq_stack[id_stack + i, :3]))
                    plotted_geneseq_stack[id_stack + i, :3] = [0, 0, 0]
            plotted_matched_geneseq_slices = np.vstack((plotted_matched_geneseq_slices, plotted_geneseq_slices[id_slices, :3]))
            plotted_geneseq_slices[id_slices, :3] = [0, 0, 0]

        stack_viewer.layers['matched_geneseq'].data = plotted_matched_geneseq_stack[:, :3]
        slice_viewer.layers['matched_geneseq'].data = plotted_matched_geneseq_slices[:, :3]

        stack_viewer.layers['geneseq'].data = plotted_geneseq_stack[:, :3]
        slice_viewer.layers['geneseq'].data = plotted_geneseq_slices[:, :3]

        np.savetxt(matching_matched_path + 'plotted_geneseq_stack.txt', plotted_geneseq_stack, fmt='%i')
        np.save(matching_matched_path + 'plotted_geneseq_stack.npy', plotted_geneseq_stack)

        np.savetxt(matching_matched_path + 'plotted_geneseq_slices.txt', plotted_geneseq_slices, fmt='%i')
        np.save(matching_matched_path + 'plotted_geneseq_slices.npy', plotted_geneseq_slices)

        np.savetxt(matching_matched_path + 'matched_geneseq_stack.txt', plotted_matched_geneseq_stack, fmt='%i')
        np.save(matching_matched_path + 'matched_geneseq_stack.npy', plotted_matched_geneseq_stack)

        np.savetxt(matching_matched_path + 'matched_geneseq_slices.txt', plotted_matched_geneseq_slices, fmt='%i')
        np.save(matching_matched_path + 'matched_geneseq_slices.npy', plotted_matched_geneseq_slices)

    napari.run()

    helpers.back_up_folders(args.history_path, [matching_matched_path, args.matching_centroids_path], [args.matching_tables_path + 'all_funseq_cells.csv', args.matching_tables_path + 'all_geneseq_cells.csv'])


def correct_matches(args):
    #script that loads current matches and allows user to correct them

    funseq_matches = np.loadtxt(args.matching_matched_path + 'funseq_matches.txt')
    geneseq_matches = np.loadtxt(args.matching_matched_path + 'geneseq_matches.txt')

    funseq_centroids = np.loadtxt(args.matching_centroids_path + 'funseq_slices_centroids.txt')
    geneseq_centroids = np.loadtxt(args.matching_centroids_path + 'geneseq_slices_centroids.txt')
    in_vivo_centroids = np.loadtxt(args.matching_centroids_path + 'padded_func_centroids.txt')

    funseq_matched_centroids = np.zeros((len(funseq_matches), 8))
    funseq_matched_centroids[:, 0] = funseq_matches[:, 0]
    funseq_matched_centroids[:, 4] = funseq_matches[:, 1]

    #for each funseq match (column 0), find the corresponding id in funseq_centroids (column 3)  and copy the centroid coordinates (columns 0-3) to the funseq_matched_centroids array
    # do the same for funseq match (column 1) in vivo coordinates (columns 4-7)
    for i in range(len(funseq_matches)):
        funseq_matched_centroids[i, 1:4] = funseq_centroids[funseq_centroids[:, 3] == funseq_matches[i, 0], :3][0]
        funseq_matched_centroids[i, 5:8] = in_vivo_centroids[in_vivo_centroids[:, 3] == funseq_matches[i, 1], :3][0]

    geneseq_matched_centroids = np.zeros((len(geneseq_matches), 8))
    geneseq_matched_centroids[:, 0] = geneseq_matches[:, 0]
    geneseq_matched_centroids[:, 4] = geneseq_matches[:, 1]
    for i in range(len(geneseq_matches)):
        geneseq_matched_centroids[i, 1:4] = geneseq_centroids[geneseq_centroids[:, 3] == geneseq_matches[i, 0], :3][0]
        geneseq_matched_centroids[i, 5:8] = in_vivo_centroids[in_vivo_centroids[:, 3] == geneseq_matches[i, 1], :3][0]

    # select unique values in funseq_matched_centroids[:, 1]. that's the slice number.
    slices = np.unique(funseq_matched_centroids[:, 1])
    # for each slice, select the rows in funseq_matched_centroids that correspond to that slice
    # select as many colors as rows are
    # plot the centroids in the same slice side by side for each slide. ignore the z coordinate.
    for slice in slices:
        slice_matches = funseq_matched_centroids[funseq_matched_centroids[:, 1] == slice, :]
        #print('slice_matches', slice_matches)
        colors = np.random.rand(len(slice_matches), 3)
        #create 2 plots side by side. increase size of plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        #label every point with the id
        #increase size of points
        for i in range(len(slice_matches)):
            ax1.scatter(slice_matches[i, 2], slice_matches[i, 3], color=colors[i, :], s=150)
            ax1.annotate(slice_matches[i, 0], (slice_matches[i, 2], slice_matches[i, 3]), fontsize=10)
            ax2.scatter(slice_matches[i, 6], slice_matches[i, 7], color=colors[i, :], s=150)
            ax2.annotate(slice_matches[i, 4], (slice_matches[i, 6], slice_matches[i, 7]), fontsize=10)
        plt.title('slice ' + str(slice))
        #save plot
        funseq_check_folder = helpers.quick_dir(args.matching_path, 'funseq_check')
        plt.savefig(funseq_check_folder + 'slice_' + str(slice) + '.png')

    #do the same for geneseq
    slices = np.unique(geneseq_matched_centroids[:, 1])
    for slice in slices:
        slice_matches = geneseq_matched_centroids[geneseq_matched_centroids[:, 1] == slice, :]
        #print('slice_matches', slice_matches)
        colors = np.random.rand(len(slice_matches), 3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(len(slice_matches)):
            ax1.scatter(slice_matches[i, 2], slice_matches[i, 3], color=colors[i, :], s=150)
            ax1.annotate(slice_matches[i, 0], (slice_matches[i, 2], slice_matches[i, 3]), fontsize=7)
            ax2.scatter(slice_matches[i, 6], slice_matches[i, 7], color=colors[i, :], s=150)
            ax2.annotate(slice_matches[i, 4], (slice_matches[i, 6], slice_matches[i, 7]), fontsize=7)
        plt.title('slice ' + str(slice))
        geneseq_check_folder = helpers.quick_dir(args.matching_path, 'geneseq_check')
        plt.savefig(geneseq_check_folder + 'slice_' + str(slice) + '.png')


def update_matches(args):
    matches = np.loadtxt(args.matching_matched_path + 'funseq_matches.txt')
    matches = np.array(matches, dtype=np.int64)

    unit_to_sunit_df = pd.read_csv(args.scan_path + '[ScanToStack]Func2StructMatching_Match.pkl.csv')
    unit_to_sunit_df = unit_to_sunit_df[unit_to_sunit_df['stack_idx'] == args.stack_id]

    unit_to_sunit_df = unit_to_sunit_df[['unit_id', 'sunit_id', 'scan_idx']]

    all_cells_df = pd.read_csv(args.matching_tables_path + 'all_funseq_cells.csv')

    all_funseq_matched_df = all_cells_df.iloc[:0, :].copy()
    # add columns unit_ID and sunit_ID
    all_funseq_matched_df['unit_ID'] = np.nan
    all_funseq_matched_df['sunit_ID'] = np.nan
    all_funseq_matched_df['scan_ID'] = np.nan
    all_cells_df['sunit_ID'] = np.nan
    all_cells_df['unit_ID'] = np.nan
    all_cells_df['scan_ID'] = np.nan

    sunit_col_index = all_cells_df.columns.get_loc('sunit_ID')
    unit_col_index = all_cells_df.columns.get_loc('unit_ID')
    scan_col_index = all_cells_df.columns.get_loc('scan_ID')
    for match in matches:
        cell_id = match[0]
        sunit_id = match[1]
        #get index of sunit_id in all_cells_df
        row_index = all_cells_df.index[all_cells_df['cell_ID'] == cell_id].tolist()
        #check how many times the sunit_id is found
        #if more than once, add a new row with the new unit_id and scan_idx
        if len(unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]) > 1:
            #get the unit_id and scan_idx for the current cell
            unit_ids = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['unit_id'].values
            scan_idx = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['scan_idx'].values
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = int(unit_ids[0])
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = int(scan_idx[0])
            for i in range(1, len(unit_ids)):
                new_row = all_cells_df.iloc[int(row_index[0])].copy()
                new_row['sunit_ID'] = int(sunit_id)
                new_row['unit_ID'] = int(unit_ids[i])
                new_row['scan_ID'] = int(scan_idx[i])
                all_cells_df = all_cells_df.append(new_row, ignore_index=True)
                all_funseq_matched_df = all_funseq_matched_df.append(new_row)
        elif len(unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]) == 1:
            unit_id = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['unit_id'].values[0]
            scan_idx = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['scan_idx'].values[0]
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = int(unit_id)
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = int(scan_idx)
        else:
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = np.nan
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = np.nan
        all_funseq_matched_df = all_funseq_matched_df.append(all_cells_df.iloc[row_index[0]])

    all_funseq_matched_df.to_csv(args.matching_tables_path + 'all_matched_funseq_cells.csv')
    print('updating funseq matches ->', len(all_funseq_matched_df), 'in total')



    matches = np.loadtxt(args.matching_matched_path + 'geneseq_matches.txt')
    matches = np.array(matches, dtype=np.int64)
    all_cells_df = pd.read_csv(args.matching_tables_path + 'all_geneseq_cells.csv')
    all_geneseq_matched_df = all_cells_df.iloc[:0, :].copy()

    all_geneseq_matched_df['unit_ID'] = np.nan
    all_geneseq_matched_df['sunit_ID'] = np.nan
    all_geneseq_matched_df['scan_ID'] = np.nan
    all_cells_df['sunit_ID'] = np.nan
    all_cells_df['scan_ID'] = np.nan
    all_cells_df['unit_ID'] = np.nan

    sunit_col_index = all_cells_df.columns.get_loc('sunit_ID')
    unit_col_index = all_cells_df.columns.get_loc('unit_ID')
    scan_col_index = all_cells_df.columns.get_loc('scan_ID')

    for match in matches:
        cell_id = match[0]
        sunit_id = match[1]
        row_index = all_cells_df.index[all_cells_df['cell_ID'] == cell_id].tolist()
        if len(unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]) > 1:
            unit_ids = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['unit_id'].values
            scan_idx = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['scan_idx'].values
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = int(unit_ids[0])
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = int(scan_idx[0])
            for i in range(1, len(unit_ids)):
                new_row = all_cells_df.iloc[int(row_index[0])].copy()
                new_row['sunit_ID'] = int(sunit_id)
                new_row['unit_ID'] = int(unit_ids[i])
                new_row['scan_ID'] = int(scan_idx[i])
                all_cells_df = all_cells_df.append(new_row, ignore_index=True)
                all_geneseq_matched_df = all_geneseq_matched_df.append(new_row)
        elif len(unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]) == 1:
            unit_id = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['unit_id'].values[0]
            scan_idx = unit_to_sunit_df[unit_to_sunit_df['sunit_id'] == sunit_id]['scan_idx'].values[0]
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = int(unit_id)
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = int(scan_idx)
        else:
            all_cells_df.iloc[int(row_index[0]), sunit_col_index] = int(sunit_id)
            all_cells_df.iloc[int(row_index[0]), unit_col_index] = np.nan
            all_cells_df.iloc[int(row_index[0]), scan_col_index] = np.nan
        all_geneseq_matched_df = all_geneseq_matched_df.append(all_cells_df.iloc[row_index[0]])

    all_geneseq_matched_df.to_csv(args.matching_tables_path + 'all_matched_geneseq_cells.csv')
    print('updating geneseq matches ->', len(all_geneseq_matched_df), 'in total')


def geneseq_analysis(args):
    from sklearn.manifold import TSNE
    from bioinfokit.visuz import cluster

    geneseq_cells_df = pd.read_csv(args.matching_tables_path + 'all_geneseq_cells.csv')
    #based on gene_IDs and gene_counts column allocate to each cell (row)
    # a barcode of length 166 and with counts as values

    #get all gene_IDs
    gene_IDs = geneseq_cells_df['gene_IDs'].values
    #get all gene_counts
    gene_counts = geneseq_cells_df['gene_counts'].values

    geneseq_barcode = np.zeros((len(geneseq_cells_df), 166), dtype=np.int16)
    for i in range(len(geneseq_cells_df)):
        #check if not nan
        if not isinstance(gene_IDs[i], float):
            #get gene_IDs and gene_counts for each cell
            gene_IDs_cell = gene_IDs[i].split(',')
            gene_counts_cell = gene_counts[i].split(',')
            #allocate to each cell a barcode of length 166
            for j in range(len(gene_IDs_cell)):
                geneseq_barcode[i, int(gene_IDs_cell[j])] = int(gene_counts_cell[j])
        else:
            geneseq_barcode[i, :] = np.zeros(166, dtype=np.int16)

    # do tsne on geneseq_barcode
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(geneseq_barcode)
    os.chdir(args.matching_tables_path)
    cluster.tsneplot(score=tsne_results, show=True)
    #save figure
    #display current folder
    print(os.getcwd())
    #change directory



    #allocate barcodes to each cell and save to csv
    geneseq_cells_df['barcode'] = geneseq_barcode.tolist()
    geneseq_cells_df.to_csv(args.matching_tables_path + 'barcoded_all_geneseq_cells.csv')

    # do tsne on barcodes
#    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     tsne_results = tsne.fit_transform(geneseq_barcode)

    #plot tsne

def preprocess_mapseq_data(args):
    barcode_matrix_path = args.mapseq_path + 'M214BarcodeMatrix.mat'
    barcode_matrix = sp.io.loadmat(barcode_matrix_path)
    #barcode_matrix = barcode_matrix['B1']
    barcode_matrix = barcode_matrix['barcodematrix']
    barcode_matrix = np.array(barcode_matrix)
    #barcode_matrix = barcode_matrix[:, 29:65]
    #barcode_matrix = barcode_matrix[:, 65:]
    #barcode_matrix = barcode_matrix[:, :29]
    [no_barcodes, brain_regions] = barcode_matrix.shape

    column_names = ['barcode_ID', 'sequence', 'target_areas', 'target_areas_IDs', 'target_areas_counts']

    barcodes_df = pd.DataFrame(columns=column_names)
    barcodes_df['target_areas'] = barcodes_df['target_areas'].astype('object')
    barcodes_df['target_areas_IDs'] = barcodes_df['target_areas_IDs'].astype('object')
    barcodes_df['target_areas_counts'] = barcodes_df['target_areas_counts'].astype('object')
    #BCM27393, limits 65:end
    #areas = ['OB', 'TRN', 'LD', 'TRN', 'dLGN', 'LP', 'vLG', 'Po', 'Amy', 'dLG', 'LP', 'LM/LI', 'SC', 'LM/LI', 'SC', 'SC', 'SC', 'L1', 'H2O']

    # BCM27816, limits start-29
    #    areas = ['OB', 'CPU', 'CPU', 'CPU+GP', 'LD', 'LD', 'TRN', 'LP', 'TRN', 'dLG', 'LP', 'TRN', 'dLG', 'LP', 'vLG', 'Amg', 'dLG', 'LP', 'dLG', 'LP', 'vLG', 'dLG', 'vLG', 'LP', 'SC', 'LPl', 'SC', 'SC', 'SC']

    #BCM27679, limits 29-65
    #areas = ['OB', 'CPU', 'GP', 'CPU', 'GP', 'LD', 'TRN', 'TRN', 'dLG', 'LP', 'TRN', 'CPU', 'Amg', 'dLG', 'LP', 'vLG', 'dLG', 'LP', 'vLG', 'dLG', 'LP', 'dLG', 'PT', 'LP', 'LP', 'SC', 'V2', 'SC', 'V2', 'SC', 'SC', 'V2', 'SC', 'SC', 'SC', 'SC']

    #BCM28382 areas = [OB	DMS_contra	MO_contra	ACA30   	LD	dLG	LP	TRN	Amg	ACA30   	TeA	dLG	LP	TeA	A30  PM_contra	AL_contra	V1_contra	dLG	LP	vLG	V1_contra	LP	SN	V1_contra	LM_LI_contra	SC	LM_LI 	SN	V1_contra	SC	V1_contra	SC	V1_contra	ECT	MEC	SC	V1_contra	L1_target	H2O_control]
    areas = ['OB', 'DMS_contra', 'MO_contra', 'ACA30', 'LD', 'dLG', 'LP', 'TRN', 'Amg', 'ACA30', 'TeA', 'dLG', 'LP', 'TeA', 'A30', 'PM_contra', 'AL_contra', 'V1_contra', 'dLG', 'LP', 'vLG', 'V1_contra', 'LP', 'SN', 'V1_contra', 'LM_LI_contra', 'SC', 'LM_LI', 'SN', 'V1_contra', 'SC', 'V1_contra', 'SC', 'V1_contra', 'ECT', 'MEC', 'SC', 'V1_contra', 'L1_target', 'H2O_control']
    # put this in a pandas with columns area_ID and area
    areas_df = pd.DataFrame(columns=['area_ID', 'area'])
    areas_df['area_ID'] = np.arange(len(areas))
    areas_df['area'] = areas
    areas_df.to_csv(args.matching_tables_path + 'projectionbook.csv')


    print('len areas is', len(areas))
    print('len brain regions is ', barcode_matrix.shape[1])
    #library = mapseq_helpers.convert_ASCII_library(barcode_matrix_path, return_letters=False, library_name='B1seq')
    library = mapseq_helpers.convert_ASCII_library(barcode_matrix_path, return_letters=False, library_name='refbarcodes')

    for barcode_id in range(no_barcodes):
        barcode_sequence = library[barcode_id]
        barcode_sequence = barcode_sequence[:15]


        target_areas = []
        target_areas_IDs = []
        target_areas_counts = []
        if np.any(barcode_matrix[barcode_id, :] > 0):
            index = len(barcodes_df.index)
            barcodes_df.at[len(barcodes_df), :2] = [barcode_id, barcode_sequence]
            for brain_region_id in range(brain_regions):
                if barcode_matrix[barcode_id, brain_region_id] > 0:
                    target_areas.append(areas[brain_region_id])
                    target_areas_IDs.append(brain_region_id)
                    target_areas_counts.append(barcode_matrix[barcode_id, brain_region_id])
            barcodes_df.at[index, 'target_areas'] = target_areas
            barcodes_df.at[index, 'target_areas_IDs'] = target_areas_IDs
            barcodes_df.at[index, 'target_areas_counts'] = target_areas_counts

                    #try:
                    #    barcodes_df.at[index, 'target_areas'] = barcodes_df[index, 'target_areas'].str() + ', ' + str(areas[brain_region_id])
                    #    barcodes_df.at[index, 'target_areas_IDs'] = barcodes_df[index, 'target_areas_IDs'].str() + ',' + str(brain_region_id)
                    #    barcodes_df.at[index, 'target_areas_counts'] = barcodes_df[index, 'target_areas_counts'].str() + ',' + str(barcode_matrix[barcode_id, brain_region_id])
                    #except KeyError:
                    #    barcodes_df.at[index, 'target_areas'] = str(areas[brain_region_id])
                    #    barcodes_df.at[index, 'target_areas_IDs'] = str(brain_region_id)
                    #    barcodes_df.at[index, 'target_areas_counts'] = str(barcode_matrix[barcode_id, brain_region_id])

                    #projects_to.append(brain_region_id)
            #projects_to = np.array(projects_to)
            #barcodes_df.at[barcode_id, 'projects_to'] = projects_to
    print('len of identified barcodes is', len(barcodes_df))
    barcodes_df.to_csv(args.mapseq_path + 'barcodes_summary.csv', index=False)

def compute_stats(args):

    positions = helpers.list_files(args.proc_genes_path)
    positions = helpers.human_sort(positions)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    for position in positions:
        print('Stats for position', position)
        position_path = helpers.quick_dir(args.proc_genes_path, position)
        slice_stats_position_path = helpers.quick_dir(args.proc_slice_stats_path, position)
        genes_cells_df = pd.read_csv(position_path + 'genes_cells.csv')
        genes_cells = genes_cells_df[['m1', 'm2', 'geneID', 'cellID', 'cell_alloc']].to_numpy(dtype=np.int)
        plt.hist(genes_cells[:,2], bins=56)
        plt.savefig(slice_stats_position_path + 'genes_hist.png')
        plt.clf()
        rolonies_incells = genes_cells[genes_cells[:,4] == 1]
        cells, occurences = np.unique(rolonies_incells[:, 3], return_counts=True)
        mean_occ = np.mean(occurences)
        std_occ = np.std(occurences)
        total_rolonies = genes_cells.shape[0]
        alloc_rolonies = rolonies_incells.shape[0]
        np.savetxt(slice_stats_position_path + 'mean_std.txt', np.array([total_rolonies, alloc_rolonies, mean_occ, std_occ]), fmt='%i')
        print('for position', position,'total rol, alloc rol, mean and std of occurences', total_rolonies, alloc_rolonies, mean_occ, std_occ)

def make_codebook_from_table(args):
    codebook = pd.read_excel(args.codebook_matlab_path + '/CodeBookIEGs6genes.xlsx')
    print(codebook)
    sample_seq = codebook.loc[0, 'seq']
    print(sample_seq)
    nr_bp = 0
    for i in sample_seq:
        if i == 'G' or i == 'T' or i == 'A' or i == 'C':
            nr_bp += 1
    print(nr_bp)
    seqbook = np.zeros(shape=(nr_bp, 4, len(codebook)))
    codebook = codebook.reset_index()  # make sure indexes pair with number of rows

    for gene_id, row in codebook.iterrows():
        print(row['seq'])
        sequence = row['seq']
        bp_id = 0
        for id, i in enumerate(sequence):
            if i == 'G':
                seqbook[bp_id, 0, gene_id] = 1
                bp_id += 1
            elif i == 'T':
                seqbook[bp_id, 1, gene_id] = 1
                bp_id += 1
            elif i == 'A':
                seqbook[bp_id, 2, gene_id] = 1
                bp_id += 1
            elif i == 'C':
                seqbook[bp_id, 3, gene_id] = 1
                bp_id += 1
    np.save(args.codebook_matlab_path + '/seqbook.npy', seqbook)
    seqbook = np.load(args.codebook_matlab_path + '/seqbook.npy')
    print(seqbook)


def prepare_codebook(args, no_channels=4):
    '''
    Prepare codebook to match the number of channels and cycles we have. also, generate a randon rgb code for each
    one of them to visualise later on. Get ids for unused barcodes. Starting point is a bit awkward and I need to change it.
    Codebookforbardensr is generated by a matlab scrips from codebook.mat. need to change this to matlab
    :param helper_files_path: location of files
    :param no_channels:...
    :return: flatcodebook, genebook and unused barcode ids to use later on to assess error rate.
    '''


    codebook = sp.io.loadmat(args.codebook_matlab_path + '/' + 'codebookforbardensr.mat')
    print(args.codebook_matlab_path + '/' + 'codebookforbardensr.mat')
    codebook = codebook['codebookbin1'] > 0
    codebook = np.moveaxis(np.moveaxis(codebook, 0, -1), 0, -2)
    codeflat = codebook.reshape((no_channels, -1))

    genebook = sp.io.loadmat(args.codebook_matlab_path + '/' + 'codebook.mat')
    genebook = genebook['codebook']
    genebook = pd.DataFrame(genebook)
    genebook.reset_index(inplace=True)

    genebook.columns = ['geneID', 'gene', 'sequence']
    genebook['gene'] = genebook['gene'].str[0]
    genebook['sequence'] = genebook['sequence'].str[0]

    R = pd.Series(np.random.randint(0, 255, len(genebook)))
    G = pd.Series(np.random.randint(0, 255, len(genebook)))
    B = pd.Series(np.random.randint(0, 255, len(genebook)))

    genebook['R'] = R
    genebook['G'] = G
    genebook['B'] = B

    unused_bc_pd = genebook[genebook['gene'].str.contains('nused')]
    unused_bc_ids = np.array(unused_bc_pd['geneID'])
    #save genebook
    genebook.to_csv(args.matching_tables_path + 'genebook.csv', index=False)
    print('unused_bc_ids', unused_bc_ids)
    return codeflat, genebook, unused_bc_ids


def prepare_codebook_new(args, helper_files_path, no_channels=4):
    '''
    Prepare codebook to match the number of channels and cycles we have. also, generate a randon rgb code for each
    one of them to visualise later on. Get ids for unused barcodes. Starting point is a bit awkward and I need to change it.
    Codebookforbardensr is generated by a matlab scrips from codebook.mat. need to change this to matlab
    :param helper_files_path: location of files
    :param no_channels:...
    :return: flatcodebook, genebook and unused barcode ids to use later on to assess error rate.
    '''

#    codebook = sp.io.loadmat(helper_files_path + '/' + 'codebookforbardensr.mat')
#    codebook = codebook['codebookbin1'] > 0
#    codebook = np.moveaxis(np.moveaxis(codebook, 0, -1), 0, -2)
#    print(codebook.shape)
    codebook = np.load(args.codebook_matlab_path + '/seqbook.npy')

    codebook = codebook[:3]
    print(codebook.shape)

    codeflat = codebook.reshape((no_channels, -1))

    genebook = sp.io.loadmat(helper_files_path + '/' + 'codebook.mat')
    genebook = genebook['codebook']
    genebook = pd.DataFrame(genebook)
    genebook.reset_index(inplace=True)

    genebook.columns = ['geneID', 'gene', 'sequence']
    genebook['gene'] = genebook['gene'].str[0]
    genebook['sequence'] = genebook['sequence'].str[0]

    R = pd.Series(np.random.randint(0, 255, len(genebook)))
    G = pd.Series(np.random.randint(0, 255, len(genebook)))
    B = pd.Series(np.random.randint(0, 255, len(genebook)))

    genebook['R'] = R
    genebook['G'] = G
    genebook['B'] = B

    unused_bc_pd = genebook[genebook['gene'].str.contains('nused')]
    unused_bc_ids = np.array(unused_bc_pd['geneID'])
    print('unused_bc_ids', unused_bc_ids)
    return codeflat, genebook, unused_bc_ids



def align_geneseq_cycles(args):
    '''

    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    for pos in positions_to_avoid:
        if pos in positions:
            positions.remove(pos)
    #Align geneseq cycles
    print('Aligning geneseq cycles')
    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_geneseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
            checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

            tiles = helpers.list_files(geneseq_position_path)
            for tile in tiles:
                shutil.copy(geneseq_position_path + tile, aligned_geneseq_position_path + tile)

        for tile in tiles:
            print('Aligning tile', tile)

            reference = tif.imread(args.proc_aligned_geneseq_path + cycles[0] + '/' + position + '/' + tile)
            reference = np.max(reference[0:4], axis=0)

            for cycle in cycles:
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
                aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
                transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_geneseq_path, cycle)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
                to_align = tif.imread(aligned_geneseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[0:4], axis=0)
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)

                if os.path.exists(transformations_position_path + tile + '.txt'):
                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:
                    #on first iteration use phase correlation to register
                    for j in range(2):
                        if j == 0:
                            _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                            to_align_mp = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[0], to_align_mp.shape[1]),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            to_align = aligned
                        #on second iteration use ECC algorithm
                            print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        elif j == 1:
                            _, transformation_matrix = helpers.ECC_reg(reference, to_align_mp, number_of_iterations=200, return_warped=False)
                            np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        # apply transformation matrix on raw image for each channel

                # Export aligned image to aligned folder.
                tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def align_preseq_to_hybseq(args):
    '''

    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align geneseq cycles
    print('Aligning preseq cycles')

    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'preseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_preseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
        preseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_preseq_path, cycle)
        tiles = helpers.list_files(preseq_position_path)

        for tile in tiles:
            shutil.copy(preseq_position_path + tile, aligned_preseq_position_path + tile)

        #check to see if transformations already exist
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
            xshifts = []
            yshifts = []

            for tile in tiles:
                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = reference[0]
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)
                #to_align_mp = np.max(to_align[0:4], axis=0)
                to_align_mp = to_align[1]
                #to_align_mp = np.mean(to_align[0:4], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
    #        x_shift, y_shift = -133.0, -73.0
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)
            print('std shifts for cycle', cycle, 'are', np.std(xshifts), np.std(yshifts))

            for tile_id, tile in enumerate(tiles):
            #    print('Aligning tile', tile)

                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = reference[0]

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > 20:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > 20:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                #save transformation matrix
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_preseq_position_path + '/' + tile, aligned[:2])

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

                bv_overlap = helpers.check_images_overlap(aligned[1], reference, save_output=False)
                tif.imwrite(aligned_tile_path + 'bv_overlap.tif', bv_overlap)
        else:
            print('Transformations already exist for', cycle, position)
            for tile in tiles:
                reference = tif.imread(args.proc_aligned_hybseq_path + 'hybseq_1' + '/' + position + '/' + tile)
                # reference = np.max(reference[0:4], axis=0)
                reference = reference[0]

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_preseq_path, cycle)
                aligned_preseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_preseq_position_path + '/' + tile)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # save transformation matrix
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_preseq_position_path + '/' + tile, aligned[:2])

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

                bv_overlap = helpers.check_images_overlap(aligned[1], reference, save_output=False)
                tif.imwrite(aligned_tile_path + 'bv_overlap.tif', bv_overlap)

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def align_hybseq_to_geneseq(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 5

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align geneseq cycles
    print('Aligning hybseq cycles')

    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'hybseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_hybseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
        hybseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_hybseq_path, cycle)

        tiles = helpers.list_files(hybseq_position_path)

        for tile in tiles:
            shutil.copy(hybseq_position_path + tile, aligned_hybseq_position_path + tile)

        # check if shifts have been calculated before. if not, do phase correlation now.
        # otherwise, load previously calculated transformation matrices
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

            xshifts = []
            yshifts = []
            for tile in tiles:
                reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + tile)
                reference = np.max(reference[0:4], axis=0)
                #reference = reference[3]
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[2:5], axis=0)
                #to_align_mp = to_align[4]
                #to_align_mp = np.mean(to_align[0:4], axis=0)
                #to_align_mp = np.mean(to_align[3:5], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                #_, transformation_matrix = helpers.ORB_reg(reference, to_align_mp, return_warped=False)
                #_, transformation_matrix = helpers.ECC_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)

            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > 20:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > 20:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_hybseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[1:4])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')
        else:
            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)

                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_hybseq_path, cycle)
                aligned_hybseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_hybseq_position_path + '/' + tile)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_hybseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[2:5])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def align_first_bcseq_to_geneseq(args):

    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align geneseq cycles
    print('Aligning first bcseq cycles')

    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        cycle = 'bcseq_1'
        cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
        aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
        bcseq_position_path = helpers.quick_dir(cycle_path, position)
        aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
        checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)
        transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)

        tiles = helpers.list_files(bcseq_position_path)

        for tile in tiles:
            shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        #check to see if transformations already exist
        if not os.path.exists(transformations_cycle_path + position):
            transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

            xshifts = []
            yshifts = []
            #run a consensus strategy. go through all the tiles of the same position and take the most likely offsets.
            for tile in tiles:
                reference = tif.imread(args.proc_aligned_geneseq_path + 'geneseq_1' + '/' + position + '/' + tile)
                #reference = np.max(reference[0:4], axis=0)
                reference = np.median(reference[0:4], axis=0)
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                to_align_mp = np.median(to_align[0:4], axis=0)
                #to_align_mp = np.mean(to_align[0:4], axis=0)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                yshifts.append(transformation_matrix[0, 2])
                xshifts.append(transformation_matrix[1, 2])

            print('shifts for cycle and position', cycle, position, xshifts, yshifts)
            x_shift = np.percentile(xshifts, 50)
            y_shift = np.percentile(yshifts, 50)
            print('transformation shifts for cycle', cycle, 'are', x_shift, y_shift)

            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)

                cycle = 'bcseq_1'
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                transformation_matrix = np.array([[1, 0, yshifts[tile_id]], [0, 1, xshifts[tile_id]]])

                if abs(transformation_matrix[0, 2] - y_shift) > 20:
                    print('Changing abnormal y_shift', transformation_matrix[0, 2], 'to', y_shift)
                    transformation_matrix[0, 2] = y_shift

                if abs(transformation_matrix[1, 2] - x_shift) > 20:
                    print('Changing abnormal x_shift', transformation_matrix[1, 2], 'to', x_shift)
                    transformation_matrix[1, 2] = x_shift

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        else:
            for tile_id, tile in enumerate(tiles):
                print('Aligning tile', tile)

                cycle = 'bcseq_1'
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)

                transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')

                # apply transformation matrix on raw image for each channel
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def align_bcseq_cycles(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    cycles.remove('bcseq_1')
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Aligning bcseq cycles')
    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
            bcseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
            checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)


            tiles = helpers.list_files(bcseq_position_path)
            for tile in tiles:
                shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        for tile in tiles:
            print('Aligning tile', tile)

            reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_1' + '/' + position + '/' + tile)
            reference = np.max(reference[0:4], axis=0)

            for cycle in cycles:
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[0:4], axis=0)
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)

                #check to see if transformation matrix already exists
                if os.path.isfile(transformations_position_path + tile + '.txt'):
                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:

                    #on first iteration use phase correlation to register
                    for j in range(1):
                        if j == 0:
                            _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                            to_align_mp = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[0], to_align_mp.shape[1]),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            to_align = aligned
                        #on second iteration use ECC algorithm
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        elif j == 1:
                            _, transformation_matrix = helpers.ECC_reg(reference, to_align_mp, number_of_iterations=200, return_warped=False)
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        #for i in range(no_channels):
                         #   aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                          #                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # apply transformation matrix on raw image for each channel

                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def align_geneseq_cycles_parallel1(args):
    '''

    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_geneseq_path)
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_geneseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    for pos in positions_to_avoid:
        if pos in positions:
            positions.remove(pos)
    #Align geneseq cycles
    print('Aligning geneseq cycles')

    @ray.remote
    def align_position(position, cycles, no_channels, args):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_geneseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
            geneseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
            checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

            tiles = helpers.list_files(geneseq_position_path)
            for tile in tiles:
                shutil.copy(geneseq_position_path + tile, aligned_geneseq_position_path + tile)

        for tile in tiles:
            print('Aligning tile', tile)

            reference = tif.imread(args.proc_aligned_geneseq_path + cycles[0] + '/' + position + '/' + tile)
            reference = np.max(reference[0:4], axis=0)

            for cycle in cycles:
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_geneseq_path, cycle)
                aligned_geneseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
                transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_geneseq_path, cycle)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)
                to_align = tif.imread(aligned_geneseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[0:4], axis=0)
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)

                if os.path.exists(transformations_position_path + tile + '.txt'):
                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:
                    #on first iteration use phase correlation to register
                    for j in range(2):
                        if j == 0:
                            _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                            to_align_mp = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[0], to_align_mp.shape[1]),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            to_align = aligned
                        #on second iteration use ECC algorithm
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        elif j == 1:
                            _, transformation_matrix = helpers.ECC_reg(reference, to_align_mp, number_of_iterations=200, return_warped=False)
                            np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        # apply transformation matrix on raw image for each channel

                # Export aligned image to aligned folder.
                tif.imwrite(aligned_geneseq_position_path + '/' + tile, aligned)

                resized = np.zeros((aligned.shape[0], int(aligned.shape[1] / args.downsample_factor),
                                    int(aligned.shape[2] / args.downsample_factor)), dtype=np.int16)
                for i in range(aligned.shape[0]):
                    resized[i] = cv2.resize(aligned[i], (resized.shape[1], resized.shape[2]))

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(resized[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

    #use n cores for parallel processing
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position, cycles, no_channels, args) for position in positions])
    ray.shutdown()

def align_bcseq_cycles_parallel1(args):
    '''
    This function aligns all sequencing cycles to each other. Works in a few stages and it relies on previous segmentation of all sequencing folders
    in prior step: geneseq, bcseq, and hyb seq. In this function:
    1. Align geneseq cycles to each other using feature based methods on raw images. I do 2 rounds of registration - first Phase Correlation and second ECC.
    2. Use segmented images from previous steps to align bcseq and hybseq cycles to newly aligned geneseq cycles. Registration algorithm is much more
    straightforward on binary images (segmented masks), hence why I'm doing this. In addition, registration precision does not have to be that
    high compared to geneseq cycles.
    3. Generate checks
    :param args:
    :return:
    '''
    print('Starting alignment of cycles')
    no_channels = 4
    cycles = helpers.list_files(args.proc_original_bcseq_path)
    cycles.remove('bcseq_1')
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_original_bcseq_path + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    #Align bcseq cycles
    print('Aligning bcseq cycles')
    @ray.remote
    def align_position(position, cycles, no_channels, args):
        print('Aligning position', position)
        pos_tic = time.perf_counter()
        #first copy all images to aligned folders and then overwrite those
        for cycle_id, cycle in enumerate(cycles):
            cycle_path = helpers.quick_dir(args.proc_original_bcseq_path, cycle)
            aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
            bcseq_position_path = helpers.quick_dir(cycle_path, position)
            aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)
            checks_alignment_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

            tiles = helpers.list_files(bcseq_position_path)
            for tile in tiles:
                shutil.copy(bcseq_position_path + tile, aligned_bcseq_position_path + tile)

        for tile in tiles:
            print('Aligning tile', tile)

            reference = tif.imread(args.proc_aligned_bcseq_path + 'bcseq_1' + '/' + position + '/' + tile)
            reference = np.max(reference[0:4], axis=0)

            for cycle in cycles:
                aligned_cycle_path = helpers.quick_dir(args.proc_aligned_bcseq_path, cycle)
                aligned_bcseq_position_path = helpers.quick_dir(aligned_cycle_path, position)

                transformations_cycle_path = helpers.quick_dir(args.proc_transf_align_bcseq_path, cycle)
                transformations_position_path = helpers.quick_dir(transformations_cycle_path, position)

                to_align = tif.imread(aligned_bcseq_position_path + '/' + tile)
                to_align_mp = np.max(to_align[0:4], axis=0)
                to_align = to_align[:no_channels]
                aligned = np.zeros_like(to_align)

                #check to see if transformation matrix already exists
                if os.path.isfile(transformations_position_path + tile + '.txt'):
                    transformation_matrix = np.loadtxt(transformations_position_path + tile + '.txt')
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:

                    #on first iteration use phase correlation to register
                    for j in range(1):
                        if j == 0:
                            _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align_mp, return_warped=False)
                            to_align_mp = cv2.warpAffine(to_align_mp, transformation_matrix, (to_align_mp.shape[0], to_align_mp.shape[1]),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            for i in range(no_channels):
                                aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            to_align = aligned
                        #on second iteration use ECC algorithm
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        elif j == 1:
                            _, transformation_matrix = helpers.ECC_reg(reference, to_align_mp, number_of_iterations=200, return_warped=False)
                            #print('transformation matrix for cycle', cycle, 'is', transformation_matrix)
                        #for i in range(no_channels):
                        #    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                         #                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        np.savetxt(transformations_position_path + tile + '.txt', transformation_matrix, fmt='%i')
                    # apply transformation matrix on raw image for each channel

                # Export aligned image to aligned folder.
                tif.imwrite(aligned_bcseq_position_path + '/' + tile, aligned)

                aligned_tile_path = helpers.quick_dir(checks_alignment_raw_position_path, tile)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(aligned_tile_path + cycle + '.tif')

        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')
    ray.init(num_cpus=args.nr_cpus)
    ray.get([align_position.remote(position, cycles, no_channels, args) for position in positions])
    ray.shutdown()



############################

def do_manual_matching(args):
    bv_slices = tif.imread(args.matching_images_path + 'blood_vessels.tif')
    somas_slices = tif.imread(args.matching_images_path + 'somas_blood_vessels.tif')
    #bv_somas_slices = tif.imread(args.matching_images_path + 'gfp_somas_blood_vessels.tif')
    genes_slices = tif.imread(args.matching_images_path + 'genes.tif')
    barcoded = tif.imread(args.matching_images_path + 'barcoded.tif')

    funseq_slices = np.loadtxt(args.matching_centroids_path + 'funseq_slices_centroids.txt')
    barcoded_slices = np.loadtxt(args.matching_centroids_path + 'barcoded_slices_centroids.txt')
    geneseq_slices = np.loadtxt(args.matching_centroids_path + 'geneseq_slices_centroids.txt')

    funseq_slices[:, [1, 2]] = funseq_slices[:, [2, 1]]
    barcoded_slices[:, [1, 2]] = barcoded_slices[:, [2, 1]]
    geneseq_slices[:, [1, 2]] = geneseq_slices[:, [2, 1]]

    slices_image = np.stack((bv_slices, somas_slices, genes_slices, barcoded), axis=0)
    bv_invivo_stack = tif.imread(args.matching_images_path + args.proc_bv_stack_name)
    gfp_invivo_stack = tif.imread(args.matching_images_path + args.proc_gfp_stack_name)
    stack_image = np.stack((bv_invivo_stack, gfp_invivo_stack), axis=0)

    padded_funseq_stack = np.loadtxt(args.matching_centroids_path + 'padded_func_centroids.txt')
    padded_funseq_stack[:, [1, 2]] = padded_funseq_stack[:, [2, 1]]
    padded_struct_stack = np.loadtxt(args.matching_centroids_path + 'padded_struct_centroids.txt')
    padded_struct_stack[:, [1, 2]] = padded_struct_stack[:, [2, 1]]


    if os.path.isfile(args.matching_matched_path + 'plotted_funseq_stack.npy'):
        plotted_funseq_stack = np.load(args.matching_matched_path + 'plotted_funseq_stack.npy')
        plotted_funseq_slices = np.load(args.matching_matched_path + 'plotted_funseq_slices.npy')
        plotted_matched_funseq_stack = np.load(args.matching_matched_path + 'matched_funseq_stack.npy')
        plotted_matched_funseq_slices = np.load(args.matching_matched_path + 'matched_funseq_slices.npy')
    else:
        plotted_funseq_stack = padded_funseq_stack.copy()
        plotted_funseq_slices = funseq_slices
        plotted_matched_funseq_stack = np.empty((0, 3))
        plotted_matched_funseq_slices = np.empty((0, 3))

    if os.path.isfile(args.matching_matched_path + 'plotted_geneseq_stack.npy'):
        plotted_geneseq_stack = np.load(args.matching_matched_path + 'plotted_geneseq_stack.npy')
        plotted_geneseq_slices = np.load(args.matching_matched_path + 'plotted_geneseq_slices.npy')
        plotted_matched_geneseq_stack = np.load(args.matching_matched_path + 'matched_geneseq_stack.npy')
        plotted_matched_geneseq_slices = np.load(args.matching_matched_path + 'matched_geneseq_slices.npy')
    else:
        plotted_geneseq_stack = padded_funseq_stack.copy()
        plotted_geneseq_slices = geneseq_slices
        plotted_matched_geneseq_stack = np.empty((0, 3))
        plotted_matched_geneseq_slices = np.empty((0, 3))

    #add centroids to stack viewer - in vivo centroids


    stack_viewer = napari.view_image(stack_image, title='In vivo stack')
    stack_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    stack_viewer.add_points(padded_struct_stack[:, :3], face_color="red", size=10, name='struct_centroids', symbol='disc', opacity=0.7, ndim=3)

    stack_viewer.add_points(plotted_matched_funseq_stack, face_color="blue", size=10, name='matched_funseq', symbol='disc', opacity=0.7, ndim=3)
    stack_viewer.add_points(plotted_funseq_stack[:, :3], face_color="green", size=10, name='funseq', symbol='disc', opacity=0.7, ndim=3)

    stack_viewer.add_points(plotted_matched_geneseq_stack, face_color="blue", size=10, name='matched_geneseq', symbol='disc', opacity=0.7, ndim=3)
    stack_viewer.add_points(plotted_geneseq_stack[:, :3], face_color="pink", size=10, name='geneseq', symbol='disc', opacity=0.7, ndim=3)

    #add centroids to slices viewer - in vitro centroids
    slice_viewer = napari.view_image(slices_image, title='In vitro slices')
    slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
    slice_viewer.add_points(barcoded_slices[:, :3], face_color="red", size=30, name='barcoded', symbol='disc', opacity=0.7, ndim=3)

    slice_viewer.add_points(plotted_matched_funseq_slices, face_color="blue", size=30, name='matched_funseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.add_points(plotted_funseq_slices[:, :3], face_color="green", size=30, name='funseq', symbol='disc', opacity=0.7, ndim=3)

    slice_viewer.add_points(plotted_matched_geneseq_slices, face_color="blue", size=30, name='matched_geneseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.add_points(plotted_geneseq_slices[:, :3], face_color="pink", size=30, name='geneseq', symbol='disc', opacity=0.7, ndim=3)
    slice_viewer.layers['matched_geneseq'].mode = 'add'


    @stack_viewer.bind_key('q')
    def save_funseq_matches(stack_viewer):
        selected_ids_stack = stack_viewer.layers['funseq'].selected_data
        selected_ids_slices = slice_viewer.layers['funseq'].selected_data

        plotted_matched_funseq_stack = stack_viewer.layers['matched_funseq'].data
        plotted_matched_funseq_slices = slice_viewer.layers['matched_funseq'].data

        selected_ids_slices = list(selected_ids_slices)
        selected_ids_stack = list(selected_ids_stack)

        cell_id_slices = plotted_funseq_slices[selected_ids_slices, 3]
        ids_slices = np.array(cell_id_slices)
        ids_slices = ids_slices.reshape((-1, 1))

        cell_id_stack = plotted_funseq_stack[selected_ids_stack, 3]
        ids_stack = np.array(cell_id_stack)
        ids_stack = ids_stack.reshape((-1, 1))

        matches_to_add = np.hstack((ids_slices, ids_stack))
        if os.path.isfile(args.matching_matched_path + 'funseq_matches.txt'):
            matches = np.loadtxt(args.matching_matched_path + 'funseq_matches.txt')
        else:
            matches = np.empty((0, 2))
        matches = np.vstack((matches, matches_to_add))
        np.savetxt(args.matching_matched_path + 'funseq_matches.txt', matches, fmt='%i')
        print('total number of funseq matches:', len(matches))

        for (id_stack, id_slices) in zip(selected_ids_stack, selected_ids_slices):
            for i in range(-10, 10):
                temp_id = plotted_funseq_stack[id_stack, 3]
                if plotted_funseq_stack[id_stack + i, 3] == temp_id:
                    plotted_matched_funseq_stack = np.vstack((plotted_matched_funseq_stack, plotted_funseq_stack[id_stack + i, :3]))
                    plotted_funseq_stack[id_stack + i, :3] = [0, 0, 0]
            plotted_matched_funseq_slices = np.vstack((plotted_matched_funseq_slices, plotted_funseq_slices[id_slices, :3]))
            plotted_funseq_slices[id_slices, :3] = [0, 0, 0]

        stack_viewer.layers['matched_funseq'].data = plotted_matched_funseq_stack[:, :3]
        slice_viewer.layers['matched_funseq'].data = plotted_matched_funseq_slices[:, :3]

        stack_viewer.layers['funseq'].data = plotted_funseq_stack[:, :3]
        slice_viewer.layers['funseq'].data = plotted_funseq_slices[:, :3]

        np.savetxt(args.matching_matched_path + 'plotted_funseq_stack.txt', plotted_funseq_stack, fmt='%i')
        np.save(args.matching_matched_path + 'plotted_funseq_stack.npy', plotted_funseq_stack)

        np.savetxt(args.matching_matched_path + 'plotted_funseq_slices.txt', plotted_funseq_slices, fmt='%i')
        np.save(args.matching_matched_path + 'plotted_funseq_slices.npy', plotted_funseq_slices)

        np.savetxt(args.matching_matched_path + 'matched_funseq_stack.txt', plotted_matched_funseq_stack, fmt='%i')
        np.save(args.matching_matched_path + 'matched_funseq_stack.npy', plotted_matched_funseq_stack)

        np.savetxt(args.matching_matched_path + 'matched_funseq_slices.txt', plotted_matched_funseq_slices, fmt='%i')
        np.save(args.matching_matched_path + 'matched_funseq_slices.npy', plotted_matched_funseq_slices)

    @stack_viewer.bind_key('e')
    def save_geneseq_matches(stack_viewer):
        selected_ids_stack = stack_viewer.layers['geneseq'].selected_data
        selected_ids_slices = slice_viewer.layers['geneseq'].selected_data

        plotted_matched_geneseq_stack = stack_viewer.layers['matched_geneseq'].data
        plotted_matched_geneseq_slices = slice_viewer.layers['matched_geneseq'].data

        selected_ids_slices = list(selected_ids_slices)
        selected_ids_stack = list(selected_ids_stack)

        cell_id_slices = plotted_geneseq_slices[selected_ids_slices, 3]
        ids_slices = np.array(cell_id_slices)
        ids_slices = ids_slices.reshape((-1, 1))

        cell_id_stack = plotted_geneseq_stack[selected_ids_stack, 3]
        ids_stack = np.array(cell_id_stack)
        ids_stack = ids_stack.reshape((-1, 1))

        matches_to_add = np.hstack((ids_slices, ids_stack))
        if os.path.isfile(args.matching_matched_path + 'geneseq_matches.txt'):
            matches = np.loadtxt(args.matching_matched_path + 'geneseq_matches.txt')
        else:
            matches = np.empty((0, 2))
        matches = np.vstack((matches, matches_to_add))
        np.savetxt(args.matching_matched_path + 'geneseq_matches.txt', matches, fmt='%i')
        print('total number of geneseq matches:', len(matches))

        for (id_stack, id_slices) in zip(selected_ids_stack, selected_ids_slices):
            for i in range(-10, 10):
                temp_id = plotted_geneseq_stack[id_stack, 3]
                if plotted_geneseq_stack[id_stack + i, 3] == temp_id:
                    plotted_matched_geneseq_stack = np.vstack((plotted_matched_geneseq_stack, plotted_geneseq_stack[id_stack + i, :3]))
                    plotted_geneseq_stack[id_stack + i, :3] = [0, 0, 0]
            plotted_matched_geneseq_slices = np.vstack((plotted_matched_geneseq_slices, plotted_geneseq_slices[id_slices, :3]))
            plotted_geneseq_slices[id_slices, :3] = [0, 0, 0]

        stack_viewer.layers['matched_geneseq'].data = plotted_matched_geneseq_stack[:, :3]
        slice_viewer.layers['matched_geneseq'].data = plotted_matched_geneseq_slices[:, :3]

        stack_viewer.layers['geneseq'].data = plotted_geneseq_stack[:, :3]
        slice_viewer.layers['geneseq'].data = plotted_geneseq_slices[:, :3]

        np.savetxt(args.matching_matched_path + 'plotted_geneseq_stack.txt', plotted_geneseq_stack, fmt='%i')
        np.save(args.matching_matched_path + 'plotted_geneseq_stack.npy', plotted_geneseq_stack)

        np.savetxt(args.matching_matched_path + 'plotted_geneseq_slices.txt', plotted_geneseq_slices, fmt='%i')
        np.save(args.matching_matched_path + 'plotted_geneseq_slices.npy', plotted_geneseq_slices)

        np.savetxt(args.matching_matched_path + 'matched_geneseq_stack.txt', plotted_matched_geneseq_stack, fmt='%i')
        np.save(args.matching_matched_path + 'matched_geneseq_stack.npy', plotted_matched_geneseq_stack)

        np.savetxt(args.matching_matched_path + 'matched_geneseq_slices.txt', plotted_matched_geneseq_slices, fmt='%i')
        np.save(args.matching_matched_path + 'matched_geneseq_slices.npy', plotted_matched_geneseq_slices)

    napari.run()



def display_rolonies(local_path):

    processed_path = helpers.quick_dir(local_path, 'processed')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    segmented_checks_path = helpers.quick_dir(checks_path, 'segmentation')
    gene_basecalling_path = helpers.quick_dir(processed_path, 'gene_basecalling')
    display_rolonies_path = helpers.quick_dir(processed_path, 'display_rolonies')

    dataset_path = aligned_path
    positions = helpers.list_files(dataset_path)
    for position in positions:


        position_path = helpers.quick_dir(dataset_path, position)
        gene_basecalling_position_path = helpers.quick_dir(gene_basecalling_path, position)
        display_rolonies_position_path = helpers.quick_dir(display_rolonies_path, position)
        segmented_checks_position_path = helpers.quick_dir(segmented_checks_path, position)

        genes_cells = pd.read_csv(gene_basecalling_position_path + 'genes_cells.csv')

        all_cycles = helpers.list_files(position_path)
        cycles = [cycle for cycle in all_cycles if 'geneseq' in cycle]
        cycles = sorted(cycles, key=lambda x: x[:-1])

        image = tif.imread(position_path + cycles[0])
        image = image[0:3]
        centers = genes_cells[['m1', 'm2', 'cell_alloc']].to_numpy(dtype=np.int)
        R = genes_cells[['R']].to_numpy(dtype=np.int)
        G = genes_cells[['G']].to_numpy(dtype=np.int)
        B = genes_cells[['B']].to_numpy(dtype=np.int)

        #contour_image = np.zeros_like(image[0])
        contour_image = tif.imread(segmented_checks_position_path + 'cells_contour.tif')
        for rol_id in range(len(genes_cells)):
            center = (int(centers[rol_id, 1]), int(centers[rol_id, 0]))
            image[0] = cv2.circle(image[0], center, radius=3, color=int(R[rol_id]), thickness=-1)
            image[1] = cv2.circle(image[1], center, radius=3, color=int(G[rol_id]), thickness=-1)
            image[2] = cv2.circle(image[2], center, radius=3, color=int(B[rol_id]), thickness=-1)

            if int(centers[rol_id, 2]) == 1: color = 250
            else: color = 70
            cv2.circle(contour_image, center, radius=2, color=color, thickness=-1)

        image_rgb = Image.fromarray(np.uint8(np.dstack(image)))
        image_rgb.save(display_rolonies_position_path + position + '.tif')

        #combined = np.ndarray((2, image.shape[1], image.shape[2]), dtype=np.uint16)
        #combined[0] = contour_image
        #combined[1] = tif.imread(segmented_checks_position_path + 'cells_contour.tif')
        tif.imwrite(segmented_checks_position_path + 'check_contours_alloc.tif', contour_image)

def color_correct(X):
    colormixing_matrix = np.array([
        [1, 0.1383, 0, 0],
        [0.3674, 1, 0, 0],
        [0, 0, 1, 0.6551],
        [0, 0, 0.0799, 1],
    ])
#    colormixing_matrix = np.array([
#        [1, 0.23, 0, 0],
#        [0.6, 1, 0, 0],
 #       [0, 0, 1, 0.5],
  #      [0, 0, 0.3, 1],
   # ])

    fix = np.linalg.inv(colormixing_matrix)
    #fix = colormixing_matrix
    XcolorCorr = np.clip(np.einsum('rcxyz,cd->rdxyz', X, fix), 0, None)
    del X

    return XcolorCorr

def register_imagestack(Xnorm, codeflat):
    corrections = bardensr.registration.find_translations_using_model(Xnorm, codeflat, niter=50)
    Xnorm_registered, newt = bardensr.registration.apply_translations(Xnorm, corrections.corrections)
    print('corrections are ', corrections.corrections)
    print('newt is ', newt)
    del Xnorm
    return Xnorm_registered

def stack_preprocessing(X, check_bool, color_corr_checks_position_path, checks_bardensr_alignment_position_path, codeflat):
    [rounds, channels, z, x, y] = np.shape(X)
    # Colorbleeding correct and subtract background
    XcolorCorr = color_correct(X)
    check_bool = False
    if check_bool is True:
        bardensr.preprocessing.colorbleed_plot(XcolorCorr[0, 0], XcolorCorr[0, 1])
        plt.savefig(color_corr_checks_position_path + 'after_colorCorr0_1.jpg')
        plt.clf()
        bardensr.preprocessing.colorbleed_plot(X[0, 0], X[0, 1])
        plt.savefig(color_corr_checks_position_path + 'before_colorCorr0_1.jpg')

        plt.clf()
        bardensr.preprocessing.colorbleed_plot(XcolorCorr[0, 2], XcolorCorr[0, 3])
        plt.savefig(color_corr_checks_position_path + 'after_colorCorr2_3.jpg')
        plt.clf()
        bardensr.preprocessing.colorbleed_plot(X[0, 2], X[0, 3])
        plt.savefig(color_corr_checks_position_path + 'before_colorCorr2_3.jpg')

    del X
    XcolorCorr = XcolorCorr.astype(np.int16)

    Xflat = XcolorCorr.reshape((rounds * channels,) + XcolorCorr.shape[-3:])
    del XcolorCorr

    Xnorm = bardensr.preprocessing.minmax(Xflat)

    del Xflat
    Xnorm = bardensr.preprocessing.background_subtraction(Xnorm, [0, 10, 10])
    Xnorm = bardensr.preprocessing.minmax(Xnorm)

    #Xnorm = register_imagestack(Xnorm, codeflat)

    return Xnorm

def denoise_images(dataset_path, train_on_tile='Pos8_003_003', max_checks=5):
    '''
    Program uses noise2void to get rid of unpredictable noise in the image. It also uses rolling ball to get rid of
    uneven illumination. N2V has to be trained on an image with tissue on it preferably. Still have to
    :param dataset_path: path to data
    :param train_on_tile: images with a lot of tissue to train on
    :param max_checks:
    :return:
    '''
    print('Denoising images')
    denoise_tic = time.perf_counter()
    processed_path = helpers.quick_dir(dataset_path, 'processed')
    original_path = helpers.quick_dir(processed_path, 'original')
    denoised_path = helpers.quick_dir(processed_path, 'denoised')
    n2v_path = helpers.quick_dir(processed_path, 'n2v')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    denoised_checks_path = helpers.quick_dir(checks_path, 'denoised')

    dataset_path = original_path

    positions = helpers.list_files(dataset_path)
    positions = helpers.human_sort(positions)
    check_positions = random.sample(positions, min(max_checks, len(positions)))

    #train model on an image.
    sample_cycles_path = dataset_path + train_on_tile
    sample_cycles = helpers.list_files(sample_cycles_path)
    sample_image = tif.imread(sample_cycles_path + '/' + sample_cycles[0])
    denoised = N2V.noise2void(sample_image[0], model_name='geneseq', localsubjectpath=n2v_path, patch_size=64,
                              rolling_ball_radius=50, double_gaussian=[None, None])

    for position in positions:
        if position in check_positions:
            check_result = True
            denoised_checks_position_path = helpers.quick_dir(denoised_checks_path, position)
        else:
            check_result = False
        position_path = helpers.quick_dir(dataset_path, position)
        denoised_position_path = helpers.quick_dir(denoised_path, position)

        cycles = helpers.list_files(position_path)
        cycles = helpers.human_sort(cycles)

        for cycle in cycles:
            cycle_image = tif.imread(position_path + cycle)
            denoised_cycle_image = np.zeros_like(cycle_image)
            for chan_id, channel in enumerate(cycle_image):
                #denoised_cycle_image[chan_id] = N2V.noise2void(channel, 'geneseq', n2v_path, patch_size=64)
                denoised_cycle_image[chan_id] = N2V.noise2void(channel, model_name='geneseq', localsubjectpath=n2v_path, patch_size=64,
                                                                   rolling_ball_radius=400, double_gaussian=[None, None])
            tif.imwrite(denoised_position_path + cycle, denoised_cycle_image)

            if check_result is True:
                tif.imwrite(denoised_checks_position_path + 'post_denoising_' + cycle, denoised_cycle_image)
                tif.imwrite(denoised_checks_position_path + 'pre_denoising_' + cycle, cycle_image)

    denoise_toc = time.perf_counter()
    print('Denoising images finished in ' + f'{denoise_toc - denoise_tic:0.4f}' + ' seconds')

def align_cycles_older(dataset_path, max_checks = 150, no_channels = 4):

    '''
    This function corrects for the shift between sequencing cycles. Starting point is a folder with max projections,
    with individual positions, before stitching. It creates a separate folder with aligned images and also one with RGB images for a quick check to
    see if it worked.
    :param dataset_path: string - the path to where the images are.
    :param max_checks: int - the maximum number of positions extracted to check whether alignment worked.
    :param no_channels: int - Choose the number of channels to include in final image. Here I disregard brightfield.
    :return: nothing, it creates aligns images and copies those in separate folders.
    '''

    #Make a list of all the positions in the folder. Check for 'MAX' to make sure we don't get unwanted folders. Can remove if problematic
    print('Aligning cycles')
    align_tic = time.perf_counter()

    processed_path = helpers.quick_dir(dataset_path, 'processed')
    original_path = helpers.quick_dir(processed_path, 'original')
    denoised_path = helpers.quick_dir(processed_path, 'denoised')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    hyb_path = helpers.quick_dir(processed_path, 'hyb')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    rgb_path = helpers.quick_dir(checks_path, 'rgb')

    dataset_path = denoised_path

    positions = helpers.list_files(dataset_path)

    #Loop through each position. Probably parallelize here
    for position in positions:
        tic = time.perf_counter() #start the clock
        print('Aligning position ', position)

        #Create folders to copy and paste output - aligned and rgb.
        position_path = helpers.quick_dir(dataset_path, position)
        aligned_position_path = helpers.quick_dir(aligned_path, position)
        hyb_position_path = helpers.quick_dir(hyb_path, position)

        #Make list of all sequencing cycles in a position folder. Remove the reference image from that list
        cycles = helpers.list_files(position_path)
        seqcycles = [cycle for cycle in cycles if 'geneseq' in cycle]
        bccycles = [cycle for cycle in cycles if 'bcseq' in cycle]
        hybcycles = [cycle for cycle in cycles if 'hybseq' in cycle]

        if len(seqcycles) > 0:
            ref_name = seqcycles[0]
            seqcycles.remove(ref_name)
        elif len(bccycles) > 0:
            ref_name = bccycles[0]
            ref_name = bccycles[0]
            bccycles.remove(ref_name)
        elif len(hybcycles) > 0:
            ref_name = hybcycles[0]
            hybcycles.remove(ref_name)
        else:
            print('No seq, hyb or bc cycles found')

        #Read in reference image and save its rgb.
        ref = tif.imread(position_path + ref_name)
        print(position_path + ref_name)
        ref = ref[0:4]
        ref = ref.squeeze()

        #Compute max proj for this to use for registration
        #ref_maxproj = np.max(ref, axis=0)
        ref_maxproj = ref[0]

        tif.imwrite(aligned_position_path + ref_name, ref)

        #Loop through all cycles and align them.
        for cycle in seqcycles:
            print('Aligning cycle ', cycle)
            no_channels = 4
            #Prepare a numpy array to store the new aligned image. Same dimensions as the reference.
            aligned = np.empty((no_channels, ref_maxproj.shape[0], ref_maxproj.shape[1]), dtype=np.uint16)

            #Read in cycle to align
            toalign = tif.imread(position_path + cycle)
            toalign = toalign[0:no_channels]
            toalign = toalign.squeeze()
            toalign_maxproj = toalign[0]

            #Get transformation matrix between reference and cycle maxproj
            #_, transformation_matrix = ORB_reg(ref_maxproj, toalign_maxproj, nfeatures=10000, view_matches=False)
            #_, transformation_matrix = ECC_reg(ref_maxproj, toalign_maxproj)
            _, transformation_matrix = helpers.PhaseCorr_reg(ref_maxproj, toalign_maxproj)
            print('transformation matrix is', transformation_matrix)
            #Use the same transformation matrix to align channels
            for i in range(no_channels):
                aligned[i] = cv2.warpAffine(toalign[i], transformation_matrix, (ref_maxproj.shape[0], ref_maxproj.shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            #Export aligned image to aligned folder.
            tif.imwrite(aligned_position_path + cycle, aligned)

        for cycle in bccycles:
            print('Aligning cycle ', cycle)
            no_channels = 4
            #Prepare a numpy array to store the new aligned image. Same dimensions as the reference.
            aligned = np.empty((no_channels, ref_maxproj.shape[0], ref_maxproj.shape[1]), dtype=np.uint16)
            toalign = tif.imread(position_path + cycle)

            #Read in cycle to align
            toalign = toalign[0:no_channels]
            toalign = toalign.squeeze()
            toalign_maxproj = toalign[0]

            #Get transformation matrix between reference and cycle maxproj
            #_, transformation_matrix = ORB_reg(ref_maxproj, toalign_maxproj, nfeatures=100000, view_matches=False)
            #_ transformation_matrix = ECC_reg(ref_maxproj, toalign_maxproj)
            _, transformation_matrix = helpers.PhaseCorr_reg(ref_maxproj, toalign_maxproj)
            for i in range(no_channels):
                aligned[i] = cv2.warpAffine(toalign[i], transformation_matrix, (ref_maxproj.shape[0], ref_maxproj.shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            print('transformation matrix is', transformation_matrix)

            tif.imwrite(aligned_position_path + cycle, aligned)

        for cycle in hybcycles:

            #Prepare a numpy array to store the new aligned image. Same dimensions as the reference.
            aligned = np.empty((no_channels, ref_maxproj.shape[0], ref_maxproj.shape[1]), dtype=np.uint16)

            #Read in cycle to align
            toalign = tif.imread(position_path + cycle)
            toalign_template = toalign[5]
            toalign_template = toalign_template.squeeze()
            toalign_maxproj = toalign[0]

            #Get transformation matrix between reference and cycle maxproj
            _, transformation_matrix = helpers.ECC_reg(ref_maxproj, toalign_template)
            #transformation_matrix = CrossCorr_reg(ref_maxproj, toalign_maxproj)
            print('transformation matrix is', transformation_matrix)


            #Use the same transformation matrix to align channels
            for i in range(no_channels):
                aligned[i] = cv2.warpAffine(toalign[i], transformation_matrix, (ref_maxproj.shape[0], ref_maxproj.shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            #Export aligned image to aligned folder.
            tif.imwrite(hyb_position_path + cycle, aligned)
        toc = time.perf_counter()
        print('Completed in ' + f'{toc - tic:0.4f}' + ' seconds')

    # Write RGB images to checks folder to make sure all is ok
    check_positions = random.sample(positions, min(max_checks, len(positions)))
    print('Exporting RGB images to check results for ', str(len(check_positions)),'positions')
    for position in check_positions:
        aligned_position_path = helpers.quick_dir(aligned_path, position)
        hyb_position_path = helpers.quick_dir(hyb_path, position)
        rgb_position_path = helpers.quick_dir(rgb_path, position)

        cycles = helpers.list_files(aligned_position_path)
        seqcycles = [cycle for cycle in cycles if 'geneseq' in cycle]
        bccycles = [cycle for cycle in cycles if 'bcseq' in cycle]

        for cycle in seqcycles:
            aligned = tif.imread(aligned_position_path + cycle)
            aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
            aligned_rgb.save(rgb_position_path + 'rgb_aligned_' + cycle)

        for cycle in bccycles:
            aligned = tif.imread(aligned_position_path + cycle)
            aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
            aligned_rgb.save(rgb_position_path + 'rgb_aligned_' + cycle)


        hybcycles = helpers.list_files(hyb_position_path)
        hybcycles = [cycle for cycle in hybcycles if 'hyb' in cycle]

        for cycle in hybcycles:
            aligned = tif.imread(hyb_position_path + cycle)
            aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[2:5])))
            aligned_rgb.save(rgb_position_path + 'rgb_aligned_' + cycle)

    align_toc = time.perf_counter()
    print('Alignment finished in ' + f'{align_toc - align_tic:0.4f}' + ' seconds')

def stardist_initial_segmentation(args):
    '''
    Segments images for all 3 cycles. this will be used mainly for alignment and a second segmentation will follow
    :param args:
    :return:
    '''
    model = StarDist2D(None, name=args.stardist_model_name, basedir=args.stardist_model_path)

    #create lists with the 3 type of sequencing to be segmented to use for loops


    sequencing_folder_paths = [args.proc_original_hybseq_path]
    output_segmentation_paths = [args.proc_segmented_original_hybseq_path]
    checks_segmentation_paths = [args.proc_checks_segmentation_hybseq_path]
    segmentation_channels = [args.hybseq_dapi_channel]

    sequencing_folder_paths = [args.proc_original_bcseq_path]
    output_segmentation_paths = [args.proc_segmented_original_bcseq_path]
    checks_segmentation_paths = [args.proc_checks_segmentation_bcseq_path]
    segmentation_channels = [args.bcseq_segmentation_channel]

    sequencing_folder_paths = [args.proc_original_geneseq_path, args.proc_original_bcseq_path, args.proc_original_hybseq_path]
    output_segmentation_paths = [args.proc_segmented_original_geneseq_path, args.proc_segmented_original_bcseq_path, args.proc_segmented_original_hybseq_path]
    checks_segmentation_paths = [args.proc_checks_segmentation_geneseq_path, args.proc_checks_segmentation_bcseq_path, args.proc_checks_segmentation_hybseq_path]
    segmentation_channels = [args.geneseq_segmentation_channel, args.bcseq_segmentation_channel, args.hybseq_dapi_channel]

    cycles = helpers.list_files(sequencing_folder_paths[0])
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_folder_paths[0] + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    #positions_to_avoid = ['Pos12']
    #for pos in positions_to_avoid:
    #    if pos in all_positions:
    #        all_positions.remove(pos)

    #get a list of all positions to randomly select a few for checks
    check_positions = random.sample(positions, min(args.max_checks, len(positions)))

    for (sequencing_folder_path, output_segmentation_path, checks_segmentation_path, segmentation_channel) \
            in zip(sequencing_folder_paths, output_segmentation_paths, checks_segmentation_paths, segmentation_channels):
        print('Segmenting sequencing folder ', sequencing_folder_path)
        for position in positions:
            print('Segmenting position', position)
            pos_tic = time.perf_counter()

            cycles = helpers.list_files(sequencing_folder_path)
            cycles = helpers.human_sort(cycles)
            if sequencing_folder_path == args.proc_original_geneseq_path:
                cycles = cycles[:1]

            for cycle_id, cycle in enumerate(cycles):
                print('Segmenting cycle', cycle)

                cycle_path = helpers.quick_dir(sequencing_folder_path, cycle)
                position_path = helpers.quick_dir(cycle_path, position)
                cycle_output_path = helpers.quick_dir(output_segmentation_path, cycle)
                position_output_path = helpers.quick_dir(cycle_output_path, position)

                tiles = helpers.list_files(position_path)
                tiles = helpers.human_sort(tiles)

                for tile in tiles:
                    tile_image = tif.imread(position_path + tile)
                    tile_image = tile_image[segmentation_channel]

                    axis_norm = (0, 1)
                    norm_tile_image = normalize(tile_image, 1, 99.8, axis=axis_norm)

                    # Labels is an array with the masks and details contains the centroids.
                    labels, details = model.predict_instances(norm_tile_image, prob_thresh=args.stardist_probability_threshold)

                    labels_image = norm_tile_image * (labels > 0)
                    binary_labels_image = np.full_like(labels_image, 255, dtype=np.int8)
                    binary_labels_image = binary_labels_image * (labels > 0)
                    tif.imwrite(position_output_path + tile, binary_labels_image)

                #export checks if position is in check_positions
                    if (position in check_positions) and (cycle_id == 0):
                        tile_checks_path = helpers.quick_dir(checks_segmentation_path, tile)
                        cycle_image_check = helpers.check_images_overlap(tile_image, labels_image, save_output=False)
                        tif.imwrite(tile_checks_path + cycle + '.tif', cycle_image_check)

                        contours_image = np.zeros_like(binary_labels_image, dtype=np.int8)

                        slice_centroids = np.array(details['points'])
                        for center in slice_centroids:
                            cen = tuple([int(center[1]), int(center[0])])
                            contours_image = cv2.circle(contours_image, cen, 27, 200, 1)

                        contours_image_check = helpers.check_images_overlap(labels_image, contours_image, save_output=False)
                        tif.imwrite(tile_checks_path + 'contours' + cycle + '.tif', contours_image_check)
            pos_toc = time.perf_counter()
            print('Position segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def align_cycles_consensus(args):
    print('Starting alignment')
    no_channels = 4
    all_positions = helpers.list_files(args.proc_segmented_hybseq_path)
    segmented_folder_paths = [args.proc_segmented_geneseq_path, args.proc_segmented_bcseq_path]
    sequencing_folder_paths = [args.proc_original_geneseq_path, args.proc_original_bcseq_path]
    aligned_paths = [args.proc_aligned_geneseq_path, args.proc_aligned_bcseq_path]

#    for position in all_positions:
#        hybseq_position_path = helpers.quick_dir(args.proc_original_hybseq_path, position)
#        aligned_hybseq_position_path = helpers.quick_dir(args.proc_aligned_hybseq_path, position)
#        hyb_cycle = helpers.list_files(hybseq_position_path)
#        shutil.copy(hybseq_position_path + hyb_cycle[0], aligned_hybseq_position_path + hyb_cycle[0])

    positions_dict = helpers.sort_position_folders(all_positions)
    for (segmented_folder_path, sequencing_folder_path, aligned_path) \
            in zip(segmented_folder_paths, sequencing_folder_paths, aligned_paths):

        for key in positions_dict:
            positions = positions_dict[key]

            cycles = helpers.list_files(segmented_folder_path + positions[0])
            cycles = helpers.human_sort(cycles)

            for cycle in cycles:
                x_shifts = []
                y_shifts = []

                for position in positions:
                    reference_position_path = helpers.quick_dir(args.proc_original_geneseq_path, position)
                    reference_image = helpers.list_files(reference_position_path)
                    reference_image = helpers.human_sort(reference_image)
                    reference = tif.imread(reference_position_path + reference_image[0])
                    #reference = reference[0]
                    reference = np.max(reference[0:4], axis=0)

                    position_path = helpers.quick_dir(sequencing_folder_path, position)
                    to_align = tif.imread(position_path + cycle)
                    #to_align = to_align[0]
                    to_align = np.max(to_align[0:4], axis=0)

                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align, return_warped=False)

                    y_shifts.append([transformation_matrix[0, 2]])
                    x_shifts.append([transformation_matrix[1, 2]])

                print('shifts for cycle and position', cycle, position, x_shifts, y_shifts)

                x_shift = np.percentile(x_shifts, 50)
                y_shift = np.percentile(y_shifts, 50)
                print('transformation matrix for cycle', cycle, 'is', x_shift, y_shift)
                transformation_matrix = np.array([[1, 0, y_shift], [0, 1, x_shift]])

                for position in positions:

                    sequencing_position_path = helpers.quick_dir(sequencing_folder_path, position)
                    aligned_position_path = helpers.quick_dir(aligned_path, position)

                    to_align = tif.imread(sequencing_position_path + cycle)
                    to_align = to_align[:no_channels]

                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_position_path + cycle, aligned)

def align_cycles_using_segmentations(args):
    print('Starting alignment using segmentation')
    no_channels = 4
    all_positions = helpers.list_files(args.proc_segmented_hybseq_path)
    segmented_folder_paths = [args.proc_segmented_geneseq_path, args.proc_segmented_bcseq_path]
    sequencing_folder_paths = [args.proc_original_geneseq_path, args.proc_original_bcseq_path]
    aligned_paths = [args.proc_aligned_geneseq_path, args.proc_aligned_bcseq_path]

    for position in all_positions:
        hybseq_position_path = helpers.quick_dir(args.proc_original_hybseq_path, position)
        aligned_hybseq_position_path = helpers.quick_dir(args.proc_aligned_hybseq_path, position)
        hyb_cycle = helpers.list_files(hybseq_position_path)
        shutil.copy(hybseq_position_path + hyb_cycle[0], aligned_hybseq_position_path + hyb_cycle[0])

    for (segmented_folder_path, sequencing_folder_path, aligned_path) \
            in zip(segmented_folder_paths, sequencing_folder_paths, aligned_paths):
        for position in all_positions:
            print('Aligning position', position)
            reference_position_path = helpers.quick_dir(args.proc_segmented_hybseq_path, position)
            #reference_position_path = helpers.quick_dir(args.proc_segmented_geneseq_path, position)
            position_path = helpers.quick_dir(segmented_folder_path, position)
            sequencing_position_path = helpers.quick_dir(sequencing_folder_path, position)
            aligned_position_path = helpers.quick_dir(aligned_path, position)
            checks_aligned_segmented_position_path = helpers.quick_dir(args.proc_checks_alignment_segmented_path, position)
            reference_image = helpers.list_files(reference_position_path)
            reference = tif.imread(reference_position_path + reference_image[0])

            cycles = helpers.list_files(position_path)
            cycles = helpers.human_sort(cycles)

            for cycle in cycles:
                to_align = tif.imread(position_path + cycle)

                _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align, return_warped=False)
                print('transformation matrix for cycle', cycle, 'is', transformation_matrix)

                to_align = to_align.astype(np.uint8)
                aligned_segmented = cv2.warpAffine(to_align, transformation_matrix, (to_align.shape[-2], to_align.shape[-1]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                tif.imwrite(checks_aligned_segmented_position_path + cycle, aligned_segmented)

                to_align = tif.imread(sequencing_position_path + cycle)
                to_align = to_align[:no_channels]

                aligned = np.zeros_like(to_align)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # Export aligned image to aligned folder.
                tif.imwrite(aligned_position_path + cycle, aligned)

def align_bcseq_hybseq_to_geneseq(args):

    print('Starting alignment of cycles')
    no_channels = 4

    sequencing_paths = [args.proc_original_bcseq_path, args.proc_original_hybseq_path]
    aligned_sequencing_paths = [args.proc_aligned_bcseq_path, args.proc_aligned_hybseq_path]
    segmented_sequencing_paths = [args.proc_segmented_original_bcseq_path, args.proc_segmented_original_hybseq_path]

    #sequencing_paths = [args.proc_original_hybseq_path]
    #aligned_sequencing_paths = [args.proc_aligned_hybseq_path]
    #segmented_sequencing_paths = [args.proc_segmented_original_hybseq_path]

    cycles = helpers.list_files(sequencing_paths[0])
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_paths[0] + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    for pos in positions_to_avoid:
        if pos in positions:
            positions.remove(pos)

    for position in positions:
        print('Aligning position', position)
        pos_tic = time.perf_counter()

        ref_cycles = helpers.list_files(args.proc_segmented_original_geneseq_path)
        ref_cycle = ref_cycles[0]
        ref_cycle = 'geneseq_1'
        ref_cycle_path = helpers.quick_dir(args.proc_segmented_original_geneseq_path, ref_cycle)
        ref_segmented_position_path = helpers.quick_dir(ref_cycle_path, position)

        for (sequencing_path, segmented_sequencing_path, aligned_sequencing_path) in zip(sequencing_paths, segmented_sequencing_paths, aligned_sequencing_paths):

            cycles = helpers.list_files(segmented_sequencing_path)
            cycles = helpers.human_sort(cycles)

            for cycle in cycles:
                print('Aligning cycle', cycle)

                cycle_path = helpers.quick_dir(sequencing_path, cycle)
                position_path = helpers.quick_dir(cycle_path, position)

                segmented_cycle_path = helpers.quick_dir(segmented_sequencing_path, cycle)
                segmented_position_path = helpers.quick_dir(segmented_cycle_path, position)

                aligned_cycle_path = helpers.quick_dir(aligned_sequencing_path, cycle)
                aligned_position_path = helpers.quick_dir(aligned_cycle_path, position)

                checks_aligned_segmented_position_path = helpers.quick_dir(args.proc_checks_alignment_segmented_path, position)
                checks_aligned_raw_position_path = helpers.quick_dir(args.proc_checks_alignment_raw_path, position)

                tiles = helpers.list_files(position_path)
                xshifts = []
                yshifts = []
                for tile in tiles:

                    reference = tif.imread(ref_segmented_position_path + tile)
                    to_align = tif.imread(segmented_position_path + tile)

                    # get transformation matrices using segmented images and apply to raw ones too
                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align, return_warped=False)
                    #print('transformation matrix for position', position, 'is', transformation_matrix)
                    if abs(transformation_matrix[0, 2]) < 150:
                        yshifts.append([transformation_matrix[0, 2]])
                    if abs(transformation_matrix[1, 2]) < 150:
                        xshifts.append([transformation_matrix[1, 2]])
                xshifts = np.array(xshifts)
                x_cycle_shift = np.mean(xshifts)
                yshifts = np.array(yshifts)
                y_cycle_shift = np.mean(yshifts)
                print('cycle x shift and yshift', x_cycle_shift, y_cycle_shift)
                for tile in tiles:

                    reference = tif.imread(ref_segmented_position_path + tile)
                    to_align = tif.imread(segmented_position_path + tile)

                    # get transformation matrices using segmented images and apply to raw ones too
                    _, transformation_matrix = helpers.PhaseCorr_reg(reference, to_align, return_warped=False)

                    if abs(transformation_matrix[0, 2]) > 150:
                        print('transformation matrix for position', tile, 'is', transformation_matrix)
                        transformation_matrix[0, 2] = y_cycle_shift
                        print('transformation matrix for position', tile, 'is', transformation_matrix)

                    if abs(transformation_matrix[1, 2]) > 150:
                        print('transformation matrix for position', tile, 'is', transformation_matrix)
                        transformation_matrix[1, 2] = x_cycle_shift
                        print('transformation matrix for position', tile, 'is', transformation_matrix)

                    to_align = to_align.astype(np.uint8)
                    aligned_segmented = cv2.warpAffine(to_align, transformation_matrix, (to_align.shape[-2], to_align.shape[-1]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

                    to_align = tif.imread(position_path + tile)
                    to_align = to_align[:no_channels]

                    aligned = np.zeros_like(to_align)
                    for i in range(no_channels):
                        aligned[i] = cv2.warpAffine(to_align[i], transformation_matrix, (aligned.shape[-2], aligned.shape[-1]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    # Export aligned image to aligned folder.
                    tif.imwrite(aligned_position_path + tile, aligned)

                    checks_aligned_raw_tile_path = helpers.quick_dir(checks_aligned_raw_position_path, tile)
                    if sequencing_path is args.proc_original_hybseq_path:
                        tif.imwrite(checks_aligned_raw_tile_path + cycle + '.tif', aligned[0]*4)
                    else:
                        aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                        aligned_rgb.save(checks_aligned_raw_tile_path + cycle + '.tif')
                    checks_aligned_segmented_tile_path = helpers.quick_dir(checks_aligned_segmented_position_path, tile)
                    tif.imwrite(checks_aligned_segmented_tile_path + cycle + '.tif', aligned_segmented)
        pos_toc = time.perf_counter()
        print('Position aligned in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')



def align_cycles_old(args):

    '''
    This function corrects for the shift between sequencing cycles. Starting point is a folder with max projections,
    with individual positions, before stitching. It creates a separate folder with aligned images and also one with RGB images for a quick check to
    see if it worked.
    In this implementation we're using a consensus strategy to calculate the offsets. We calculate offsets for all tiles in a position
    and then get the 90th percentile of those offsets, i.e. most common occurence. We're doing this because Phase works well when
    slice contains a lot of the tissue but fails when there's very little, i.e. outside the slice. So we're prioritizing offsets from
    slices which contain tissue.
    :param dataset_path: string - the path to where the images are.
    :param max_checks: int - the maximum number of positions extracted to check whether alignment worked.
    :param no_channels: int - Choose the number of channels to include in final image. Here I disregard brightfield.
    :return: nothing, it creates aligns images and copies those in separate folders.
    '''

    #Make a list of all the positions in the folder. Check for 'MAX' to make sure we don't get unwanted folders. Can remove if problematic
    assert args.registration_method == 'PHASE' or 'ORB' or 'ECC', f'only options for registration are PHASE, ORB or ECC'
    print('Aligning cycles')
    align_tic = time.perf_counter()

    processed_path = args.proc_path
    original_path = args.proc_original_path
    aligned_path = args.proc_aligned_path
    hyb_path = helpers.quick_dir(processed_path, 'hyb')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    checks_alignment_path = helpers.quick_dir(checks_path, 'alignment')

    all_positions = helpers.list_files(args.proc_original_geneseq_path)
    positions_dict = helpers.sort_position_folders(all_positions)

    no_channels = 4
    for key in positions_dict:
        positions = positions_dict[key]
        cycles = helpers.list_files(args.proc_original_geneseq_path + positions[0])
        cycles = helpers.human_sort(cycles)
        seqcycles = [cycle for cycle in cycles if 'geneseq' in cycle]

        if len(seqcycles) > 0:
            ref_name = seqcycles[0]
            seqcycles.remove(ref_name)

        for cycle in seqcycles:
            x_shifts = []
            y_shifts = []

            for position in positions:
                #Create folders to copy and paste output - aligned and rgb.
                position_path = helpers.quick_dir(args.proc_original_geneseq_path, position)
                aligned_position_path = helpers.quick_dir(aligned_path, position)

                #Read in reference image and save it.
                ref = tif.imread(position_path + ref_name)
                ref = ref[0:no_channels]
                tif.imwrite(aligned_position_path + ref_name, ref)
                ref = ref[0]
                ref = ref.squeeze()

                toalign = tif.imread(position_path + cycle)
                toalign = toalign[0]
                toalign = toalign.squeeze()

                #Get transformation matrix between reference and cycle maxproj
                if registration_method == 'PHASE':
                    _, transformation_matrix = helpers.PhaseCorr_reg(ref, toalign, return_warped=False)
                elif registration_method =='ECC':
                    _, transformation_matrix = helpers.ECC_reg(ref, toalign, return_warped=False)
                elif registration_method =='ORB':
                    _, transformation_matrix = helpers.ORB_reg(ref, toalign, return_warped=False)

                #Use the same transformation matrix to align channels
                y_shifts.append([transformation_matrix[0, 2]])
                x_shifts.append([transformation_matrix[1, 2]])
            print(y_shifts)
            print(x_shifts)
            #y_shifts = [i for i in y_shifts if np.abs(i) != 0.0]
# Here we're creating a consensus calculation to get the offset. Choose the most common occurrence - 90th percentile.
            x_shift = np.percentile(x_shifts, 50)
            y_shift = np.percentile(y_shifts, 50)

            transformation_matrix = np.array([[1, 0, y_shift], [0, 1, x_shift]])
            print('final TM is ', transformation_matrix)
            print('Aligning cycle ', cycle)
            for position in positions:

                # Create folders to copy and paste output - aligned and rgb.
                position_path = helpers.quick_dir(dataset_path, position)
                aligned_position_path = helpers.quick_dir(aligned_path, position)

                # Prepare a numpy array to store the new aligned image. Same dimensions as the reference.
                # Read in cycle to align
                toalign = tif.imread(position_path + cycle)
                toalign = toalign[0:no_channels]

                aligned = np.empty((no_channels, toalign.shape[1], toalign.shape[2]), dtype=np.uint16)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(toalign[i], transformation_matrix,
                                                (aligned.shape[1], aligned.shape[2]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

                # Export aligned image to aligned folder.
                tif.imwrite(aligned_position_path + cycle, aligned)


                #take care of hybridisation cycle now
                '''
                                hyb_position_path = helpers.quick_dir(hyb_path, position)
                hyb_aligned_position_path = helpers.quick_dir(hyb_aligned_path, position)

                hyb_cycle = tif.imread(hyb_position_path + 'hybseq.tif')

                _, transformation_matrix = helpers.PhaseCorr_reg(aligned[0], hyb_cycle[0], return_warped=False)

                aligned = np.zeros_like(hyb_cycle)
                for i in range(no_channels):
                    aligned[i] = cv2.warpAffine(hyb_cycle[i], transformation_matrix,
                                                (aligned.shape[1], aligned.shape[2]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                tif.imwrite(hyb_aligned_position_path + 'hybseq.tif', aligned)
                '''





        # Write RGB images to checks folder to make sure all is ok
        check_positions = random.sample(positions, min(max_checks, len(positions)))
        print('Exporting RGB images to check results for ', str(len(check_positions)),'positions')
        for position in check_positions:
            aligned_position_path = helpers.quick_dir(aligned_path, position)
            hyb_position_path = helpers.quick_dir(hyb_path, position)
            checks_alignment_position_path = helpers.quick_dir(checks_alignment_path, position)

            cycles = helpers.list_files(aligned_position_path)
            seqcycles = [cycle for cycle in cycles if 'geneseq' in cycle]
            bccycles = [cycle for cycle in cycles if 'bcseq' in cycle]

            for cycle in seqcycles:
                aligned = tif.imread(aligned_position_path + cycle)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(checks_alignment_position_path + 'rgb_aligned_' + cycle)

            for cycle in bccycles:
                aligned = tif.imread(aligned_position_path + cycle)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[0:3])))
                aligned_rgb.save(checks_alignment_position_path + 'rgb_aligned_' + cycle)


            hybcycles = helpers.list_files(hyb_position_path)
            hybcycles = [cycle for cycle in hybcycles if 'hyb' in cycle]

            for cycle in hybcycles:
                aligned = tif.imread(hyb_position_path + cycle)
                aligned_rgb = Image.fromarray(np.uint8(np.dstack(aligned[2:5])))
                aligned_rgb.save(checks_alignment_position_path + 'rgb_aligned_' + cycle)

    align_toc = time.perf_counter()
    print('Alignment finished in ' + f'{align_toc - align_tic:0.4f}' + ' seconds')


def flip_horizontally(filepath, overwrite_images=False):
    '''
    Script that flips images ina a folder horizontally.
    :param filepath: specigy folder where images are
    :param overwrite_images: bool parameter. if true it overwrites images in current folder, otherwise creates a new folder.
    :return: path of folder where flipped images are
    '''
    print('Flipping images')
    if overwrite_images is True:
        flipped_path = filepath
    else:
        path = pathlib.PurePath(filepath)
        folder_name = 'flipped' + path.name + '/'
        flipped_path = helpers.quick_dir(filepath + '../', folder_name)

    images = helpers.list_files(filepath)

    for image_name in images:
        print('flipping image', image_name)
        image = tif.imread(filepath + image_name)
        if image.ndim == 2:
            flipped_image = np.flip(image, axis=1)
        elif image.ndim == 3:
            flipped_image = np.flip(image, axis=2)
        else:
            print('Ndim for image is ', image.ndim, '. Ndim 2 or 3 expected')
        tif.imwrite(flipped_path + image_name, flipped_image)

    return flipped_path

def stitch_images_imperfectly_folder(filepath, input_overlap=0.15, specific_chan=None, output_folder_name=None, output_path=None, reduce_size=False, reduce_size_factor=3):
    '''
    Stitches images taken in tiles sequence by microscope. Really messy way of doing it, but couldn't be bothered to find a better one.
    Input:
        :param filepath: Path to where files are
        :param input_overlap: known overlap between images
        :param specific_chan: integer; if specific channel is wanted only, specify here
    Returns:
        nothing - it creates a folder 'stitched' where it adds all images
    '''

    print('Stitching images..')

    # Starts from a folder where max projections are created. Read all position files and create a separate folder -stitched-

    maxproj_path = filepath
    path = pathlib.PurePath(filepath)

    if output_folder_name is None:
        folder_name = 'stitched_' + path.name + '_' + str(specific_chan) + '/'
    else:
        folder_name = output_folder_name + '/'

    if output_path is None:
        stitched_path = helpers.quick_dir(filepath + '../', folder_name)
    else:
        stitched_path = helpers.quick_dir(output_path, folder_name)

    positions_list = helpers.list_files(maxproj_path)
    # Extract a sample image to get dimensions and number of channels.

    sample_image = tif.imread(maxproj_path + positions_list[0])

    if sample_image.ndim == 2 or specific_chan is not None:
        no_channels = 1
        pixel_dim = sample_image.shape[1]
    elif sample_image.ndim == 3:
        no_channels = min(sample_image.shape)
        pixel_dim = sample_image.shape[1]
    else:
        print('I only prepared this function for 2 or 3 channel images')

    # Sometimes images are not max projections, so their naming scheme is different. Namely, it's not 'MAX_Pos1_1_1', but just 'Pos1_1_1'.
    # Distinguish between the two cases by getting the starting segment before 'Pos'.
    segment = re.search('(.*)Pos(.*)', positions_list[0])
    starting_segment = segment.group(1)

    # Get a list of positions.
    positions_int = []
    for position_name in positions_list:
        segment = re.search(starting_segment + 'Pos(.*)_(.*)_(.*)', position_name)
        pos = segment.group(1);
        pos = int(pos);
        if pos not in positions_int:
            positions_int.append(pos)

    # Create a dictionary to allocate all tiles to a specific position
    keys_to_search = ['Pos' + str(pos_int) for pos_int in positions_int]
    positions_dict = {key: [] for key in keys_to_search}
    x_max_dict = {key: 0 for key in keys_to_search}
    y_max_dict = {key: 0 for key in keys_to_search}

    # Get max number of tiles in each dimension.
    for position_name in positions_list:
        segment = re.search(starting_segment + '(.*)_(.*)_(.*).tif', position_name)
        pos = segment.group(1);
        y_max = segment.group(2);
        y_max = int(y_max);
        x_max = segment.group(3);
        x_max = int(x_max);
        for key in keys_to_search:
            if key == pos:
                positions_dict[key].append(position_name)
                if y_max > y_max_dict[key]:
                    y_max_dict[key] = y_max
                if x_max > x_max_dict[key]:
                    x_max_dict[key] = x_max

    # for each position, stitch images. start by stitching images into individual columns, and the stitch columns.
    # The maths is messy, but it works
    for position in positions_dict:
        tic = time.perf_counter()
        x_max = x_max_dict[position] + 1
        y_max = y_max_dict[position] + 1

        reduced_pixel_dim = int((1 - input_overlap) * pixel_dim)

        x_pixels = (x_max - 1) * reduced_pixel_dim + pixel_dim
        y_pixels = (y_max - 1) * reduced_pixel_dim + pixel_dim

        stitched = np.empty((no_channels, x_pixels, y_pixels), dtype=np.float16)
        stitched_cols = np.empty((no_channels, y_max, x_pixels, pixel_dim), dtype=np.float16)

        for column in range(y_max):
            for row in range(x_max):
                image_name = starting_segment + position + '_00' + str(y_max - column - 1) + '_00' + str(x_max - row - 1)

                # image = tif.imread(maxproj_path+ '/' + image_name + '/' + 'geneseq1.tiff')

                image = tif.imread(maxproj_path + '/' + image_name + '.tif')

                if np.where(image.shape == no_channels) == 2:
                    image = np.transpose(image, (2, 0, 1))

                if specific_chan is not None:
                    image = image[specific_chan, :, :]
                if image.ndim == 2: image = np.expand_dims(image, axis=0)

                if row != (x_max - 1):
                    reduced_image = image[:, 0:reduced_pixel_dim, :]
                    stitched_cols[:, column, row * reduced_pixel_dim: (row + 1) * reduced_pixel_dim, :] = reduced_image
                else:
                    stitched_cols[:, column, row * reduced_pixel_dim:, :] = image

        for column in range(y_max):
            if column != (y_max - 1):
                reduced_col = stitched_cols[:, column, :, 0:reduced_pixel_dim]
                stitched[:, :, column * reduced_pixel_dim:(column + 1) * reduced_pixel_dim] = reduced_col
            else:
                stitched[:, :, column * reduced_pixel_dim:(column * reduced_pixel_dim + pixel_dim)] = stitched_cols[:, column]

        # stitched = stitched.astype('uint8')
        stitched = stitched.astype(np.uint16)
        if reduce_size is True:
            if no_channels > 1:
                resized = np.zeros((stitched.shape[0], int(stitched.shape[1] / reduce_size_factor),
                                    int(stitched.shape[2] / reduce_size_factor)), dtype=np.int16)
                for i in range(stitched.shape[0]):
                    resized[i, :, :] = cv2.resize(stitched[i, :, :], (int(stitched.shape[2] / reduce_size_factor),
                                                                      int(stitched.shape[1] / reduce_size_factor)))

                    #resized[i] = cv2.resize(stitched[i], (resized.shape[1], resized.shape[2]))
            else:
                resized = cv2.resize(stitched, (int(stitched.shape[1] / reduce_size_factor),
                                                int(stitched.shape[1] / reduce_size_factor)))
            stitched = resized

        tif.imwrite(stitched_path + position + '.tif', stitched)
        toc = time.perf_counter()
        print('stitching of ' + position + ' finished in ' + f'{toc - tic:0.4f}' + ' seconds')
    print('Stitching done.')

    return stitched_path


def stardist_segment_stitched_hyb(args):
    model = StarDist2D(None, name=args.stardist_model_name, basedir=args.stardist_model_path)
    export_checks_bool = True

    #get a list of all positions to randomly select a few for checks

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_hybseq_path)
        #positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    for pos in positions_to_avoid:
        if pos in positions:
            positions.remove(pos)
    print('Segmenting sequencing folder ', args.proc_transformed_hybseq_path)
#    print('Segmenting sequencing folder ', args.proc_transformed_geneseq_path)
    cell_id_offset = 0
    for position in positions:
        print('Segmenting position', position)
        pos_tic = time.perf_counter()
        position_index = helpers.get_trailing_number(position)
#        position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_final_geneseq_path, position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)

        position_image = tif.imread(position_path + position + '.tif')
        position_image = position_image[args.hybseq_dapi_channel]
        #position_image = position_image[args.geneseq_segmentation_channel]
        # set normalization and normalize the image
        #position_image = np.mean(position_image, axis=0)

        #vmin, vmax = np.percentile(position_image, q=(0, 99.5))

        #clipped_data = skimage.exposure.rescale_intensity(data, in_range=(vmin, vmax), out_range=np.float32)


        #sigma = 2
        #for i in range(5):
        #    position_image = skimage.filters.gaussian(position_image, sigma=(sigma, sigma), truncate=3)

        axis_norm = (0, 1)
        #position_image = normalize(position_image, 1, 99.8, axis=axis_norm)

        #position_image = skimage.exposure.equalize_hist(position_image)
        #position_image = skimage.exposure.equalize_adapthist(position_image, clip_limit=0.03)

        norm_position_image = normalize(position_image, 1, 99.8, axis=axis_norm)

        #Labels is an array with the masks and details contains the centroids.
        labels, details = model.predict_instances(norm_position_image, prob_thresh=args.stardist_probability_threshold)

        labels_image = norm_position_image * (labels > 0)
        binary_labels_image = np.full_like(labels_image, 255, dtype=np.int8)
        binary_labels_image = binary_labels_image * (labels > 0)
        tif.imwrite(position_output_path + position + '.tif', binary_labels_image)

        column_names = ['cell_ID', 'cell_X', 'cell_Y', 'cell_Z']
        cells_df = pd.DataFrame(columns=column_names)
        #cells_df['contour'] = cells_df['contour'].astype('object')

        contours = np.array(details['coord'], dtype=np.int32)
        centroids = np.array(details['points'])
        xcontours = contours[:, 1, :]
        ycontours = contours[:, 0, :]

        for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
            cells_df.at[cell_id, 'cell_ID'] = cell_id + cell_id_offset
            cells_df.at[cell_id, 'cell_X'] = centroid[1]
            cells_df.at[cell_id, 'cell_Y'] = centroid[0]
            cells_df.at[cell_id, 'cell_Z'] = position_index * 20

        if export_checks_bool is True:
            contours_image = np.zeros_like(binary_labels_image, dtype=np.int8)
            centroids_image = np.zeros_like(binary_labels_image, dtype=np.int8)
            for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
                plot_contour = [np.hstack((xcontours[cell_id].reshape((-1, 1)), ycontours[cell_id].reshape((-1, 1))))]
                cv2.drawContours(image=contours_image, contours=plot_contour, contourIdx=-1, color=120, thickness=1)
                center = tuple([int(centroid[1]), int(centroid[0])])
                cv2.circle(centroids_image, center, 15, 200, -1)

            contours_image_check = helpers.check_images_overlap(labels_image, contours_image, save_output=False)
            tif.imwrite(checks_segmentation_position_path + 'check_contours.tif', contours_image_check)

            centroids_image_check = helpers.check_images_overlap(labels_image, centroids_image, save_output=False)
            tif.imwrite(checks_segmentation_position_path + 'check_centroids.tif', centroids_image_check)

            cycle_image_check = helpers.check_images_overlap(position_image, labels_image, save_output=False)
            tif.imwrite(checks_segmentation_position_path + 'check_segmentation.tif', cycle_image_check)

        #cells_df.to_csv(genes_position_path + 'cells.csv', index=False)
        cells_df.to_csv(genes_position_path + 'cells.csv')
        np.savetxt(centroids_position_path + 'xcontours.txt', xcontours, fmt='%i')
        np.savetxt(centroids_position_path + 'ycontours.txt', ycontours, fmt='%i')

        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds, with', len(contours), 'cells found')
        cell_id_offset += len(contours)
        print('cell id offset is', cell_id_offset)


def cellpose_segment_stitched1(args):
    model = StarDist2D(None, name=args.stardist_model_name, basedir=args.stardist_model_path)
    export_checks_bool = True
    model = models.Cellpose(model_type='cyto2')

    hyb_mask_val = 150
    rol_mask_val = 254

    #get a list of all positions to randomly select a few for checks

    if len(args.positions) == 0:
        #positions = helpers.list_files(args.proc_transformed_hybseq_path)
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    positions_to_avoid = []
    for pos in positions_to_avoid:
        if pos in positions:
            positions.remove(pos)
#    print('Segmenting sequencing folder ', args.proc_transformed_hybseq_path)
    print('Segmenting sequencing folder ', args.proc_transformed_geneseq_path)
    cell_id_offset = 0
    for position in positions:
        print('Segmenting position', position)
        pos_tic = time.perf_counter()
        position_index = helpers.get_trailing_number(position)
        position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_final_geneseq_path, position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)

        position_image = tif.imread(position_path + position + '.tif')
        hybseq_image = tif.imread(hybseq_position_path + position + '.tif')
        hybseq_image = hybseq_image[4]

        #position_image = position_image[args.hybseq_dapi_channel]
        #position_image = position_image[args.geneseq_segmentation_channel]
        # set normalization and normalize the image
        position_image = np.max(position_image, axis=0)
        position_image = skimage.exposure.match_histograms(position_image, hybseq_image)
        #tif.imwrite(position_output_path + 'cyto.tif', position_image)
        #tif.imwrite(position_output_path + 'nuclei.tif', hybseq_image)

        #breakpoint()
        #geneseq_image = position_image.copy()
        #hybseq_image = skimage.exposure.match_histograms(hybseq_image, position_image, multichannel=True)

        #position_image = skimage.exposure.equalize_hist(position_image)

        #position_image = np.stack((position_image, hybseq_image), axis=0)
        #position_image = np.mean(position_image, axis=0)

        #vmin, vmax = np.percentile(position_image, q=(0, 99.5))

        #clipped_data = skimage.exposure.rescale_intensity(data, in_range=(vmin, vmax), out_range=np.float32)

        #sigma = 2
        #for i in range(1):
        #    position_image = skimage.filters.gaussian(position_image, sigma=(sigma, sigma), truncate=3)

        #axis_norm = (0, 1)
        #position_image = normalize(position_image, 1, 99.8, axis=axis_norm)

        #position_image = skimage.exposure.equalize_hist(position_image)
        #position_image = skimage.exposure.equalize_adapthist(position_image, clip_limit=0.03)

        #norm_position_image = normalize(position_image, 1, 99.8, axis=axis_norm)
        #norm_hyb_position_image = normalize(hybseq_image, 1, 99.8, axis=axis_norm)

        position_image = np.max((position_image, hybseq_image), axis=0)
        position_image = position_image[1000:1300, 1700:2200]

        #Labels is an array with the masks and details contains the centroids.
        #labels, details = model.predict_instances(norm_position_image, prob_thresh=args.stardist_probability_threshold)
        #hyb_labels, hyb_details = model.predict_instances(norm_hyb_position_image, prob_thresh=args.stardist_probability_threshold)
        channels = [0, 0]
        mask, flows, styles, diams = model.eval(position_image, diameter=28.98, channels=channels, flow_threshold=0.4, cellprob_threshold=0)

        contours, centroids = helpers.find_cellpose_contours(mask)

        labels_image = position_image * (mask > 0)
        binary_labels_image = np.full_like(labels_image, rol_mask_val, dtype=np.int8)
        binary_labels_image = binary_labels_image * (mask > 0)
        tif.imwrite(position_output_path + position + '.tif', binary_labels_image)

        column_names = ['cell_ID', 'cell_X', 'cell_Y', 'cell_Z']
        cells_df = pd.DataFrame(columns=column_names)
        column_name = ['contour']
        contours_df = pd.DataFrame(columns=column_name)
        #contours_df['contour'] = contours_df['contour'].astype('object')

        #contours = np.array(details['coord'], dtype=np.int32)
        #centroids = np.array(details['points'])


        for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
            cells_df.at[cell_id, 'cell_ID'] = cell_id + cell_id_offset
            cells_df.at[cell_id, 'cell_X'] = centroid[1]
            cells_df.at[cell_id, 'cell_Y'] = centroid[0]
            cells_df.at[cell_id, 'cell_Z'] = position_index * 20

            contours_image = binary_labels_image
            centroids_image = np.zeros_like(binary_labels_image, dtype=np.int8)
            cv2.drawContours(image=contours_image, contours=[contour], contourIdx=-1, color=120, thickness=1)
            center = tuple([int(centroid[1]), int(centroid[0])])
            cv2.circle(centroids_image, center, 15, 200, -1)
            contours_df.at[cell_id, 'contour'] = contour

        contours_image_check = helpers.check_images_overlap(position_image, contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_contours.tif', contours_image_check)

        cycle_image_check = helpers.check_images_overlap(position_image, labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_segmentation.tif', cycle_image_check)

        cells_df.to_csv(genes_position_path + 'cells.csv')
        contours_df.to_csv(centroids_position_path + 'contours.csv')

        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds, with', len(contours), 'cells found')
        cell_id_offset += len(contours)
        print('cell id offset is', cell_id_offset)
def allocate_rolonies_old(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_genes_path)
    else:
        positions = args.positions

    for position in positions:
        print('Allocating rolonies for position', position)
        pos_tic = time.perf_counter() #start the clock
        position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        geneseq_position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        geneseq_image = tif.imread(geneseq_position_path + position + '.tif')
        geneseq_image = np.max(geneseq_image[:4], axis=0)
        geneseq_image = geneseq_image[1000:1300, 1700:2200]
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)

        cells_df = pd.read_csv(position_path + 'cells.csv', index_col=[0])
        cells = cells_df[['cell_ID', 'cell_X', 'cell_Y']].to_numpy()

        cells_df['genes'] = ''
        cells_df['gene_IDs'] = ''
        cells_df['sunit_ID'] = ''
        rolonies_df = pd.read_csv(position_path + 'rolonies.csv', index_col=[0])
        rolonies = rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        genes_cells = rolonies_df.copy()

        #xcontours = np.loadtxt(centroids_position_path + 'xcontours.txt')
        #ycontours = np.loadtxt(centroids_position_path + 'ycontours.txt')
        #xcontours = xcontours.astype(np.int32)
        #ycontours = ycontours.astype(np.int32)
        contours_pd = pd.read_csv(centroids_position_path + 'contours.csv', index_col=[0])
        #contours_pd = pd.read_csv(centroids_position_path + 'contours.csv', converters={'contour': literal_eval}, index_col=[0])
        #contours_pd['contour'] = contours_pd['contour'].map(lambda x: [y.replace('\n', '') for y in x])

        #contours_pd['contour'] = pd.eval(contours_pd['contour'])
        #contours = np.array(contours_pd['contour'])
        contours = contours_pd['contour']

        print('len contours', len(contours))


        contours_image = np.zeros_like(geneseq_image, dtype=np.int8)

        printcounter = 0
        for cell_id, cell in enumerate(cells):

            if printcounter == 2000:
                print('cell id is', cell_id, 'out of', len(cells))
                printcounter = 0
            printcounter += 1
            #contour = [np.hstack((xcontours[cell_id].reshape((-1, 1)), ycontours[cell_id].reshape((-1, 1))))]

            contours[cell_id] = contours[cell_id].replace('\n', "")
            contour = contours[cell_id]

            print(contour)
            cell_color = np.random.randint(255)
            cv2.drawContours(image=contours_image, contours=contour, contourIdx=-1, color=cell_color, thickness=1)

            relevant_rows = np.where((rolonies[:, 1] < cell[1] + 30) & (rolonies[:, 1] > cell[1] - 30))
            relevant_rolonies = rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < cell[2] + 30) & (relevant_rolonies[:, 2] > cell[2] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]
                #print(len(relevant_rolonies))

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour[0], rolony_centroid, False) >= 0:
                        genes_cells.at[real_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        genes_cells.at[real_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        genes_cells.at[real_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        genes_cells.at[real_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        genes_cells.at[real_rol_ID, 'cell_alloc'] = 1
                        cells_df.at[cell_id, 'genes'] = cells_df.at[cell_id, 'genes'] + ', ' + rolonies_df.at[real_rol_ID, 'gene']
                        cells_df.at[cell_id, 'gene_IDs'] = cells_df.at[cell_id, 'gene_IDs'] + ',' + rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                        cv2.circle(contours_image, rolony_centroid, 2, color=cell_color, thickness=-1)
            except IndexError:
                continue
        contours_image_check = helpers.check_images_overlap(geneseq_image, contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_rolony_allocation.tif', contours_image_check)

        genes_cells.to_csv(position_path + 'genes_cells.csv')
        cells_df.to_csv(position_path + 'cells.csv')
        pos_toc = time.perf_counter()
        alloc_rol_ids = genes_cells.index[genes_cells['cell_alloc'] == 1].tolist()
        print('nr rolonies allocated is', len(alloc_rol_ids))
        print('Rolony allocation for position', position, 'done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


def find_barcoded_somas_old(args):

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_bcseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    coordinates_path = helpers.quick_dir(args.proc_path, 'coordinates')
    if 0 == 1:
        bcseq_position_path = helpers.quick_dir(args.proc_transformed_bcseq_path, positions[0])
        bcseq_image = tif.imread(bcseq_position_path + positions[0] + '.tif')
        bcseq_image = np.max(bcseq_image[0:4], axis=0)

        slice_viewer = napari.view_image(bcseq_image, title='In vitro slice')
        slice_viewer.window.qt_viewer.setGeometry(0, 0, 50, 50)
        slice_viewer.add_shapes(shape_type='rectangle', name='rectangle', edge_width=5, edge_color='coral', face_color='royalblue')

        @slice_viewer.bind_key('q')
        def move_on(stack_viewer):
            corners = slice_viewer.layers['rectangle'].data
            corners = np.array(corners[0])
            np.savetxt(coordinates_path + 'bcseq_corners.txt', corners, fmt='%i')

        napari.run()

    corners = np.loadtxt(coordinates_path + 'bcseq_corners.txt')
    x_min, x_max = corners[0,1], corners[2,1]
    y_min, y_max = corners[0,0], corners[2,0]

    no_channels = 4
    no_cycles = 15
    for position in positions:
        print('Finding barcoded somas in ', position)
        pos_tic = time.perf_counter()

        position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        bcseq_position_path = helpers.quick_dir(args.proc_transformed_bcseq_path, position)
        checks_basecalling_position_path = helpers.quick_dir(args.proc_checks_basecalling_path, position)

        bcseq_image = tif.imread(bcseq_position_path + position + '.tif')
        #axis_norm = (1, 2)
        #bcseq_image = normalize(bcseq_image, 1, 99.8, axis=axis_norm)

        xcontours = np.loadtxt(centroids_position_path + 'xcontours.txt')
        ycontours = np.loadtxt(centroids_position_path + 'ycontours.txt')
        xcontours = xcontours.astype(np.int32)
        ycontours = ycontours.astype(np.int32)

        cells_df = pd.read_csv(position_path + 'cells.csv', index_col=[0])
        cells = cells_df[['cell_ID', 'cell_X', 'cell_Y']].to_numpy()

        relevant_rows = np.where((cells[:, 1] >= x_min) & (cells[:, 1] <= x_max))
        cells = cells[relevant_rows]
        xcontours = xcontours[relevant_rows]
        ycontours = ycontours[relevant_rows]
        relevant_rows = np.where((cells[:, 2] >= y_min) & (cells[:, 2] <= y_max))
        cells = cells[relevant_rows]
        xcontours = xcontours[relevant_rows]
        ycontours = ycontours[relevant_rows]
        print('size of cells', cells.shape)

        printcounter = 0
        #cells = cells[:500]
        cells_intensities = np.zeros(shape=(len(cells), bcseq_image.shape[0]))

        for cell_id, cell in enumerate(cells):
            if printcounter == 50:
                print('cell id is', cell_id, 'out of', len(cells))
                printcounter = 0
            printcounter += 1
            contour = [np.hstack((xcontours[cell_id].reshape((-1, 1)), ycontours[cell_id].reshape((-1, 1))))]
            mask = np.zeros_like(bcseq_image[0])
            cell_contours = bcseq_image
            cv2.drawContours(image=mask, contours=contour, contourIdx=-1, color=255, thickness=-1)
            pts = np.where(mask == 255)
            for chan_id in range(bcseq_image.shape[0]):
                cells_intensities[cell_id, chan_id] = int(np.mean(bcseq_image[chan_id][pts[0], pts[1]]))

        for chan_id in range(cells_intensities.shape[1]):
            chan_mean = np.mean(cells_intensities[:, chan_id])
            cells_intensities[chan_id] = cells_intensities[chan_id]/chan_mean
        np.savetxt(centroids_position_path + 'intensities.txt', cells_intensities, fmt='%i')


        no_channels = 4
        cycles = int(bcseq_image.shape[0]/no_channels)

        cells_seqs = np.zeros((2, cells.shape[0], cycles))
        for cell_id, cell in enumerate(cells):
            for cycle_id in range(cycles):
                max_int = np.max(cells_intensities[cell_id, cycle_id * no_channels : (cycle_id+1) * no_channels])
                max_id = np.argmax(cells_intensities[cell_id, cycle_id * no_channels : (cycle_id+1) * no_channels])
                sum_quare_int = 0
                for i in range(4):
                    sum_quare_int += cells_intensities[cell_id, cycle_id * no_channels + i]**2
                cells_seqs[0, cell_id, cycle_id] = max_id
                cells_seqs[1, cell_id, cycle_id] = max_int / np.sqrt(sum_quare_int)

        first_n_bp = 15
        for cell_id in range(cells.shape[0]):
            if np.mean(cells_seqs[1, cell_id, :first_n_bp] > 0.85):
                real_cell_id = cells[cell_id, 0]
                row_index = cells_df.index[cells_df['cell_ID'] == real_cell_id].tolist()

                cells_df.at[row_index[0], 'barcoded'] = 1
                cells_df.at[row_index[0], 'bc_seq'] = str(cells_seqs[0, cell_id])
                contour = [np.hstack((xcontours[cell_id].reshape((-1, 1)), ycontours[cell_id].reshape((-1, 1))))]

                for chan_id in range(bcseq_image.shape[0]):
                    cv2.drawContours(image=cell_contours[chan_id], contours=contour, contourIdx=-1, color=200, thickness=1)
                    if chan_id % 4 == cells_seqs[0, cell_id, int(chan_id/4)]:
                        cv2.circle(cell_contours[chan_id], center=tuple([int(cells[cell_id,1]), int(cells[cell_id,2])]), radius=30, color=350, thickness=3)

        resized = np.zeros((cell_contours.shape[0], 2048, 2048), dtype=np.int16)
        for i in range(cell_contours.shape[0]):
            resized[i] = cv2.resize(cell_contours[i], (2048, 2048))
        tif.imwrite(checks_basecalling_position_path + 'check_soma_basecalling.tif', resized)

        cells_df.to_csv(position_path + 'barcoded_cells.csv')
        try:
            barcoded_cells_ids = cells_df.index[cells_df['barcoded'] == 1].tolist()
            print('nr barcodes cells is', len(barcoded_cells_ids))
        except KeyError:
            print('no barcodes cells found')
        np.savetxt(centroids_position_path + 'basecalling_scores.txt', cells_seqs[1])
        pos_toc = time.perf_counter()
        print('Barcoded somas identified for position', position, 'done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def find_barcoded_somas1(args):
    model = models.Cellpose(model_type='cyto2')

    if len(args.positions) == 0:
        positions = helpers.list_files(args.proc_transformed_geneseq_path)
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    coordinates_path = helpers.quick_dir(args.proc_path, 'coordinates')

    cell_id_offset = 0
    for position in positions:
        print('Segmenting position', position)
        pos_tic = time.perf_counter()
        position_index = helpers.get_trailing_number(position)
        coordinates_position_path = helpers.quick_dir(coordinates_path, position)

        position_path = helpers.quick_dir(args.proc_transformed_geneseq_path, position)
        hybseq_position_path = helpers.quick_dir(args.proc_transformed_hybseq_path, position)
        bcseq_position_path = helpers.quick_dir(args.proc_transformed_bcseq_path, position)
        position_output_path = helpers.quick_dir(args.proc_segmented_final_geneseq_path, position)
        genes_position_path = helpers.quick_dir(args.proc_genes_path, position)
        centroids_position_path = helpers.quick_dir(args.proc_centroids_path, position)
        checks_segmentation_position_path = helpers.quick_dir(args.proc_checks_segmentation_final_path, position)
        checks_basecalling_position_path = helpers.quick_dir(args.proc_checks_basecalling_path, position)

        bcseq_image = tif.imread(bcseq_position_path + position + '.tif')

        corners = np.loadtxt(coordinates_position_path + 'barcodes_area.txt')
        x_min, x_max = int(corners[0, 1]), int(corners[2, 1])
        y_min, y_max = int(corners[0, 0]), int(corners[2, 0])

        position_image = np.max(bcseq_image, axis=0)
        # small_position_image = position_image[x_min:x_max, y_min:y_max]
        small_position_image = position_image[y_min:y_max, x_min:x_max]

        channels = [0, 0]
        # mask, flows, styles, diams = model.eval(small_position_image, diameter=34.74, channels=channels, flow_threshold=0.4, cellprob_threshold=0)
        mask, flows, styles, diams = model.eval(small_position_image, diameter=None, channels=channels, flow_threshold=0.4, cellprob_threshold=0)

        big_mask = np.zeros_like(bcseq_image[0], dtype=mask.dtype)
        # big_mask[x_min:x_max, y_min:y_max] = mask
        big_mask[y_min:y_max, x_min:x_max] = mask
        mask = big_mask
        contours, centroids = helpers.find_cellpose_contours(mask)

        labels_image = position_image * (mask > 0)
        binary_labels_image = np.full_like(labels_image, 0, dtype=np.int8)
        binary_labels_image = binary_labels_image * (mask > 0)
        tif.imwrite(position_output_path + position + '.tif', binary_labels_image)

        cells_df = pd.DataFrame(columns=['cell_ID', 'cell_X', 'cell_Y', 'cell_Z'])
        contours_df = pd.DataFrame(columns=['contour'])
        cells_df['genes'] = ''
        cells_df['gene_IDs'] = ''
        cells_df['sunit_ID'] = ''

        rolonies_df = pd.read_csv(genes_position_path + 'rolonies.csv', index_col=[0])
        rolonies = rolonies_df[['rol_ID', 'rol_X', 'rol_Y']].to_numpy(dtype=np.int32)
        genes_cells = rolonies_df.copy()

        all_contours_image = np.zeros_like(position_image, dtype=np.int8)

        cells_intensities = np.empty(shape=(0, bcseq_image.shape[0]))

        arange = 2000
        x_min, x_max = int(bcseq_image.shape[-2] / 2) - arange, int(bcseq_image.shape[-2] / 2) + arange
        y_min, y_max = int(bcseq_image.shape[-1] / 2) - arange, int(bcseq_image.shape[-1] / 2) + arange

        printcounter = 0
        barcoded_cells = []

        for cell_id, (centroid, contour) in enumerate(zip(centroids, contours)):
            cells_df.at[cell_id, 'cell_ID'] = cell_id + cell_id_offset
            cells_df.at[cell_id, 'cell_X'] = centroid[0]
            cells_df.at[cell_id, 'cell_Y'] = centroid[1]
            cells_df.at[cell_id, 'cell_Z'] = position_index * 20
            cells_df.at[cell_id, 'genes'] = ''
            cells_df.at[cell_id, 'gene_IDs'] = ''
            contours_df.at[cell_id, 'contour'] = contour

            contours_image = binary_labels_image
            centroids_image = np.zeros_like(binary_labels_image, dtype=np.int8)
            cv2.drawContours(image=contours_image, contours=[contour], contourIdx=-1, color=120, thickness=1)
            center = tuple([int(centroid[1]), int(centroid[0])])
            cv2.circle(centroids_image, center, 15, 200, -1)

            if printcounter == 2000:
                print('cell id is', cell_id, 'out of', len(centroids))
                printcounter = 0
            printcounter += 1

            cell_color = np.random.randint(255)
            cv2.drawContours(image=all_contours_image, contours=[contour], contourIdx=-1, color=cell_color, thickness=1)

            relevant_rows = np.where((rolonies[:, 1] < centroid[0] + 30) & (rolonies[:, 1] > centroid[0] - 30))
            relevant_rolonies = rolonies[relevant_rows]
            try:
                relevant_rows = np.where((relevant_rolonies[:, 2] < centroid[1] + 30) & (relevant_rolonies[:, 2] > centroid[1] - 30))
                relevant_rolonies = relevant_rolonies[relevant_rows]

                for rol_ID, rolony in enumerate(relevant_rolonies):
                    rolony_centroid = tuple([int(rolony[1]), int(rolony[2])])
                    real_rol_ID = rolony[0]
                    if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                        genes_cells.at[real_rol_ID, 'cell_ID'] = cells_df.at[cell_id, 'cell_ID']
                        genes_cells.at[real_rol_ID, 'cell_X'] = cells_df.at[cell_id, 'cell_X']
                        genes_cells.at[real_rol_ID, 'cell_Y'] = cells_df.at[cell_id, 'cell_Y']
                        genes_cells.at[real_rol_ID, 'cell_Z'] = cells_df.at[cell_id, 'cell_Z']
                        genes_cells.at[real_rol_ID, 'cell_alloc'] = 1
                        cells_df.at[cell_id, 'genes'] = cells_df.at[cell_id, 'genes'] + ', ' + rolonies_df.at[real_rol_ID, 'gene']
                        cells_df.at[cell_id, 'gene_IDs'] = cells_df.at[cell_id, 'gene_IDs'] + ',' + rolonies_df.at[real_rol_ID, 'geneID'].astype(str)
                        cv2.circle(all_contours_image, rolony_centroid, 2, color=cell_color, thickness=-1)
            except IndexError:
                continue

            #        for cell_id, cell in enumerate(cells):
            if (centroid[0] >= x_min) & (centroid[0] <= x_max) & (centroid[1] >= y_min) & (centroid[1] <= y_max):
                barcoded_cells.append(cell_id)
                mask = np.zeros_like(bcseq_image[0])
                cv2.drawContours(image=mask, contours=[contour], contourIdx=-1, color=255, thickness=-1)
                pts = np.where(mask == 255)
                chan_intensities = np.zeros(shape=(1, bcseq_image.shape[0]))
                for chan_id in range(bcseq_image.shape[0]):
                    chan_intensities[0, chan_id] = int(np.mean(bcseq_image[chan_id][pts[0], pts[1]]))
                cells_intensities = np.vstack((cells_intensities, chan_intensities))
        for chan_id in range(cells_intensities.shape[1]):
            chan_mean = np.mean(cells_intensities[:, chan_id])
            cells_intensities[:, chan_id] = cells_intensities[:, chan_id] / chan_mean
        np.savetxt(centroids_position_path + 'intensities.txt', cells_intensities, fmt='%f2')

        no_channels = 4
        cycles = int(bcseq_image.shape[0] / no_channels)
        cells_seqs = np.zeros((2, cells_intensities.shape[0], cycles))

        for cell_id, cell in enumerate(barcoded_cells):
            for cycle_id in range(cycles):
                max_int = np.max(cells_intensities[cell_id, cycle_id * no_channels: (cycle_id + 1) * no_channels])
                max_id = np.argmax(cells_intensities[cell_id, cycle_id * no_channels: (cycle_id + 1) * no_channels])
                sum_quare_int = 0
                for i in range(4):
                    sum_quare_int += cells_intensities[cell_id, cycle_id * no_channels + i] ** 2
                cells_seqs[0, cell_id, cycle_id] = max_id
                cells_seqs[1, cell_id, cycle_id] = max_int / np.sqrt(sum_quare_int)

        first_n_bp = 15
        for cell_id, real_cell_id in enumerate(barcoded_cells):
            contour = [contours[real_cell_id]]
            for chan_id in range(bcseq_image.shape[0]):
                cv2.drawContours(image=bcseq_image[chan_id], contours=contour, contourIdx=-1, color=200, thickness=2)
            if np.mean(cells_seqs[1, cell_id, :first_n_bp] > args.basecalling_score_thresh):
                row_index = cells_df.index[cells_df['cell_ID'] == real_cell_id].tolist()

                cells_df.at[row_index[0], 'barcoded'] = 1
                cells_df.at[row_index[0], 'bc_seq'] = str(cells_seqs[0, cell_id])

                for chan_id in range(bcseq_image.shape[0]):
                    cv2.drawContours(image=bcseq_image[chan_id], contours=contour, contourIdx=-1, color=100, thickness=2)
                    if chan_id % 4 == cells_seqs[0, cell_id, int(chan_id / 4)]:
                        centroid = centroids[real_cell_id]
                        cell_centroid = tuple([int(centroid[0]), int(centroid[1])])
                        cv2.circle(bcseq_image[chan_id], center=cell_centroid, radius=30, color=350, thickness=3)

        resized = np.zeros(
            (bcseq_image.shape[0], int(bcseq_image.shape[1] / args.downsample_factor), int(bcseq_image.shape[2] / args.downsample_factor)),
            dtype=np.int16)
        for i in range(bcseq_image.shape[0]):
            resized[i] = cv2.resize(bcseq_image[i], (resized.shape[1], resized.shape[2]))
        tif.imwrite(checks_basecalling_position_path + 'check_soma_basecalling.tif', resized)

        cells_df.to_csv(genes_position_path + 'barcoded_cells.csv')
        try:
            barcoded_cells_ids = cells_df.index[cells_df['barcoded'] == 1].tolist()
            print('nr barcodes cells is', len(barcoded_cells_ids))
        except KeyError:
            print('no barcodes cells found')
        np.savetxt(centroids_position_path + 'basecalling_scores.txt', cells_seqs[1], fmt='%f2')

        contours_image_check = helpers.check_images_overlap(position_image, all_contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_rolony_allocation.tif', contours_image_check)

        genes_cells.to_csv(genes_position_path + 'genes_cells.csv')
        pos_toc = time.perf_counter()

        contours_image_check = helpers.check_images_overlap(position_image, contours_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_contours.tif', contours_image_check)

        cycle_image_check = helpers.check_images_overlap(position_image, labels_image, save_output=False)
        tif.imwrite(checks_segmentation_position_path + 'check_segmentation.tif', cycle_image_check)

        cells_df.to_csv(genes_position_path + 'cells.csv')
        contours_df.to_csv(centroids_position_path + 'contours.csv')

        pos_toc = time.perf_counter()
        print('Position', position, 'segmented in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds, with', len(contours), 'cells found')
        cell_id_offset += len(contours)
        print('cell id offset is', cell_id_offset)

        alloc_rol_ids = genes_cells.index[genes_cells['cell_alloc'] == 1].tolist()
        print('nr rolonies allocated is', len(alloc_rol_ids))
        print('Rolony allocation for position', position, 'done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

def preprocess_images_hybseq(args):
    darkfield = tif.imread(args.darkfield_path)
    flatfield = tif.imread(args.flatfield_path)
    no_channels = 5

    sequencing_folder_paths = [args.proc_original_hybseq_path]
    output_segmentation_paths = [args.proc_segmented_original_hybseq_path]
    checks_segmentation_paths = [args.proc_checks_segmentation_hybseq_path]

    #get a list of all positions to randomly select a few for checks
    cycles = helpers.list_files(sequencing_folder_paths[0])
    cycles = helpers.human_sort(cycles)

    if len(args.positions) == 0:
        positions = helpers.list_files(sequencing_folder_paths[0] + cycles[0])
        positions = helpers.human_sort(positions)
    else:
        positions = args.positions

    check_positions = random.sample(positions, min(args.max_checks, len(positions)))


    for (sequencing_folder_path, output_segmentation_path, checks_segmentation_path) \
            in zip(sequencing_folder_paths, output_segmentation_paths, checks_segmentation_paths):
        print('Preprocessing sequencing folder ', sequencing_folder_path)

        cycles = helpers.list_files(sequencing_folder_path)
        cycles = helpers.human_sort(cycles)

        #cycles = cycles[:1]

        for cycle in cycles:
            print('Preprocessing cycle', cycle)
            pos_tic = time.perf_counter()
            cycle_path = helpers.quick_dir(sequencing_folder_path, cycle)
            cycle_output_path = helpers.quick_dir(output_segmentation_path, cycle)

            for position in positions:
                position_path = helpers.quick_dir(cycle_path, position)
                position_output_path = helpers.quick_dir(cycle_output_path, position)

                tiles = helpers.list_files(position_path)
                #tiles = ['MAX_Pos12_001_001.tif']

                if position in check_positions:
                    color_corr_checks_position_path = helpers.quick_dir(args.proc_checks_color_correction_path, position)
                    export_checks_bool = True
                export_checks_bool = False
                for tile in tiles:
                    tile_image = tif.imread(position_path + tile)
                    tile_image = tile_image[:no_channels]
                    for chan in range(no_channels):
                        tile_image[chan] = tile_image[chan] - darkfield[chan]
                        tile_image[chan] = tile_image[chan] / flatfield[chan]

                    if sequencing_folder_path != args.proc_original_hybseq_path:

                        tile_image = np.expand_dims(tile_image, axis=(0,2))
                        colormixing_matrix = np.array([
                            [1, 0.02, 0, 0],
                            [0.9, 1, 0, 0],
                            [0, 0, 1, 0.99],
                            [0, 0, 0, 1]])

                        fix = np.linalg.inv(colormixing_matrix)
                        tile_image_ColorCorr = np.clip(np.einsum('rcxyz,cd->rdxyz', tile_image, fix), 0, None)


                        if export_checks_bool is True:
                            bardensr.preprocessing.colorbleed_plot(tile_image_ColorCorr[0, 0], tile_image_ColorCorr[0, 1])
                            plt.savefig(color_corr_checks_position_path + cycle + tile +'_after_colorCorr0_1.jpg')
                            plt.clf()
                            bardensr.preprocessing.colorbleed_plot(tile_image[0, 0], tile_image[0, 1])
                            plt.savefig(color_corr_checks_position_path + cycle + tile + '_before_colorCorr0_1.jpg')

                            plt.clf()
                            bardensr.preprocessing.colorbleed_plot(tile_image_ColorCorr[0, 2], tile_image_ColorCorr[0, 3])
                            plt.savefig(color_corr_checks_position_path + cycle + tile + '_after_colorCorr2_3.jpg')
                            plt.clf()
                            bardensr.preprocessing.colorbleed_plot(tile_image[0, 2], tile_image[0, 3])
                            plt.savefig(color_corr_checks_position_path + cycle + tile + '_before_colorCorr2_3.jpg')

                        tile_image_ColorCorr = tile_image_ColorCorr.astype(np.int16)
                        #tif.imwrite(position_output_path + tile, tile_image_ColorCorr)
                        tile_image_ColorCorr = np.squeeze(tile_image_ColorCorr)
                        tile_image = tile_image_ColorCorr
                    tif.imwrite(position_path + tile, tile_image)

            pos_toc = time.perf_counter()
            print('Cycle preprocessed in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')


