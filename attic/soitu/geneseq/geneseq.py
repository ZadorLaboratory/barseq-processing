#!/usr/bin/env python
# coding: utf-8

'''
Workflow for analysing barseq data:
--Arrange images by position
--Align sequencing cycles
--Basecalling using bardensr
--Segment cells with cellpose and allocate rolonies to cells
--Display rolonies
--Stitch images to visualize results
'''

import geneseq_helpers
import argparse

def parse_me_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-lcl', '--local_bool', type=bool, default=True, help='boolean for whether data is local or remote')
    parser.add_argument('-lcl_path', '--local_path', type=str, default='/Users/soitu/Desktop/datasets/BCM27393-1/geneseq/', help='if working local, location of dataset')
    parser.add_argument('-reg_method', '--registration_method', default='PHASE', type=str, help='registration method. 3 options: ECC, ORB, PHASE')

    parser.add_argument('-maxproj', '--maxproj_name', type=str, default='maxproj', help='default name used for max projections')
    parser.add_argument('-codebook_path', '--codebook_matlab_path', type=str, default='/Users/soitu/Desktop/code/bardensr/helper/codebook_56', help='path to codebook')

    args = parser.parse_args()

    return args

def main(args):
    #geneseq_helpers.arrange_by_pos(dataset_path=args.local_path, maxproj_folder=args.maxproj_name)
    #geneseq_helpers.align_cycles(dataset_path=args.local_path, registration_method=args.registration_method)
    #geneseq_helpers.bardensr_gene_basecalling(local_path=args.local_path, max_checks=20, helper_files_path=args.codebook_matlab_path)
    #geneseq_helpers.allocate_rolonies(local_path=args.local_path)
    geneseq_helpers.display_rolonies(local_path=args.local_path)
    geneseq_helpers.compute_stats(local_path=args.local_path)
    geneseq_helpers.stitch_images_imperfectly_folder(local_path=args.local_path)


    #geneseq_helpers.denoise_images(dataset_path=args.local_path, train_on_tile='Pos8_003_003')

args = parse_me_args()
main(args)



