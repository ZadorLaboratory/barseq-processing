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


''' Folder structure will look like this:
-----------------Must have-----------------------------
/dataset/config.toml
        /funseq/
               slices/
                    /geneseq_1/maxproj/MAX_PosZ_00X_00Y...
                    /geneseq_2/maxproj/MAX_PosZ_00X_00Y...
                    :
                    :
                    /geneseq_7/maxproj/MAX_PosZ_00X_00Y...
                    /hybseq/maxproj/MAX_PosZ_00X_00Y...
                    /bcseq_1/maxproj/MAX_PosZ_00X_00Y...
                    :
                    :
                    /bcseq_14/maxproj/MAX_PosZ_00X_00Y...
                    /preseq/maxproj/MAX_PosZ_00X_00Y...
               scan/
               stack/
--------------------------------------------------------
---------------Will generate----------------------------
            /processed
                    /original
                            /geneseq
                            /bcseq
                            /hybseq
                            /preseq
                    /aligned
                            /geneseq
                            /bcseq
                            /hybseq
                            /preseq
                    /checks
                            /alignment
                                    /segmented
                                    /raw
                            /basecalling
                            /color_correction
                            /segmentation
                                    /geneseq
                                    /bcseq
                                    /hybseq
                                    /final
                    /coordinates
                    /samples
                    /genes
                    /centroids
                    /stitched
                            /geneseq
                            /bcseq
                            /hybseq
                            /preseq
                            /funseq
                                    /bv
                                    /somas
                    /transformed
                            /geneseq
                            /bcseq
                            /hybseq
                    /segmented
                            /original
                                    /geneseq
                                    /bcseq
                                    /hybseq
                            /aligned
                                    /geneseq
                                    /bcseq
                                    /hybseq
                            /final
                                    /geneseq
                                    /bcseq
                                    /hybseq
                    /display_rolonies
                    /slice_stats
---------------------------------------------------------
'''

import barseq_helpers

#dataset_path = '/Users/soitu/Desktop/datasets/BCM28382/'
#dataset_path = '/grid/zador/data/Cristian/datasets/BCM28382/'

dataset_path = '/Users/soitu/Desktop/datasets/BCM27679_1/'
#dataset_path = '/grid/zador/data/Cristian/datasets/BCM27679_1/'

#dataset_path = '/Users/soitu/Desktop/datasets/BCM27679_2/'
#dataset_path = '/grid/zador/data/Cristian/datasets/BCM27679_2/'

#dataset_path = '/Users/soitu/Desktop/datasets/Kings/'

#dataset_path = '/Users/soitu/Desktop/datasets/BCM27393_2/'

def main(args):

#    args.nr_cpus = 4
#    args.positions = ['Pos25']

# to redo alignment for BCM28382. realign hybseq and preseq
#    args.positions = ['Pos15', 'Pos22', 'Pos25', 'Pos29', 'Pos31', 'Pos34', 'Pos35', 'Pos44', 'Pos51']


#    barseq_helpers.preprocess_slices(args)
#    barseq_helpers.preprocess_stack(args)
#    barseq_helpers.preprocess_scan(args)
#    barseq_helpers.mark_barcodes_areas(args)
#    breakpoint()

#    barseq_helpers.arrange_by_pos(args)
#    barseq_helpers.preprocess_images(args)

#    barseq_helpers.manual_align_hybseq_cycle(args)

#    barseq_helpers.scale_align_geneseq_cycles(args)
#    barseq_helpers.scale_align_hybseq_cycle(args)
#    barseq_helpers.scale_align_preseq_cycle(args)
#    barseq_helpers.scale_align_bcseq_cycle(args)
#    barseq_helpers.scale_align_all_bcseq_cycles(args)

#    barseq_helpers.align_geneseq_cycles_parallel(args)
#    barseq_helpers.align_hybseq_to_geneseq_parallel(args)
#    barseq_helpers.align_preseq_to_hybseq_parallel(args)
#    barseq_helpers.align_first_bcseq_to_geneseq_parallel(args)
#    barseq_helpers.align_bcseq_cycles_parallel(args)

#    barseq_helpers.stitch_tiles_middle(args)
#    barseq_helpers.apply_slice_transformations(args)
#    barseq_helpers.create_funseq_stacks(args)

#    barseq_helpers.bardensr_gene_basecalling(args)

#    barseq_helpers.cellpose_segment_barcoded(args)
#    barseq_helpers.stardist_segment_hybseq_rolonies(args)
#    barseq_helpers.allocate_rolonies_to_barcoded(args)
#    barseq_helpers.allocate_projection_areas(args)

#    barseq_helpers.cellpose_segment_cells(args)
#    barseq_helpers.allocate_rolonies_geneseq(args)
#    barseq_helpers.assign_classes_hybseq(args)

#    barseq_helpers.generate_final_tables(args)
#    barseq_helpers.preprocess_functional(args)

    barseq_helpers.inspect_cells(args, cell_IDs=[59141])
#    barseq_helpers.inspect_cells(args, cell_IDs=[91579, 91591])

#    barseq_helpers.matchers_matching(args)
#    barseq_helpers.update_matches(args)
#    barseq_helpers.correct_matches(args)


#    barseq_helpers.geneseq_analysis(args)

#    barseq_helpers.preprocess_mapseq_data(args)
#    barseq_helpers.generate_metadata(args)
#    barseq_helpers.make_codebook_from_table(args)


##### One time only

args = barseq_helpers.parse_me_args(dataset_path)
main(args)



