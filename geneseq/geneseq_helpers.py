import tifffile as tif
import os
import helpers
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
from cellpose import models
import random
from PIL import Image
import pathlib
#import N2V


def arrange_by_pos(dataset_path, maxproj_folder):
    '''
    :param local_bool:
    :param local_path:
    :param channels_order:
    :param remote_path:
    :param server:
    :param username:
    :param password:
    :return:
    '''


    print('Arranging images in local folder')

    processed_path = helpers.quick_dir(dataset_path, 'processed')
    original_path = helpers.quick_dir(processed_path, 'original')

    seq_folders = helpers.list_files(dataset_path)
    geneseq_folders = helpers.remove_outstringers(seq_folders, 'geneseq')
    geneseq_folders = sorted(geneseq_folders, key=lambda x: x[-1])

    for cycle in geneseq_folders:
        cycle_no = helpers.get_trailing_number(cycle)

        cycle_path = dataset_path + cycle + '/' + maxproj_folder + '/'
        all_pos = helpers.list_files(cycle_path)

        for position_name in all_pos:
            segment = re.search('MAX_Pos(.*).tif', position_name)
            pos = segment.group(1)
            if cycle_no == 1:
                helpers.quick_dir(original_path, 'Pos' + pos)
            new_pos_path = original_path + 'Pos' + pos + '/'
            shutil.copy(cycle_path + position_name, new_pos_path + 'geneseq' + str(cycle_no) + '.tif')

    seq_folders = helpers.list_files(dataset_path)
    bcseq_folders = helpers.remove_outstringers(seq_folders, 'bcseq')
    bcseq_folders = sorted(bcseq_folders, key=lambda x: x[-1])

    for cycle in bcseq_folders:
        cycle_no = helpers.get_trailing_number(cycle)

        cycle_path = dataset_path + cycle + '/' + maxproj_folder + '/'
        all_pos = helpers.list_files(cycle_path)

        for position_name in all_pos:
            segment = re.search('MAX_Pos(.*).tif', position_name)
            pos = segment.group(1)
            if cycle_no == 1:
                helpers.quick_dir(original_path, 'Pos' + pos)
            new_pos_path = original_path + 'Pos' + pos + '/'
            shutil.copy(cycle_path + position_name, new_pos_path + 'bcseq' + str(cycle_no) + '.tif')


    seq_folders = helpers.list_files(dataset_path)
    hybseq_folders = helpers.remove_outstringers(seq_folders, 'hyb')
    hybseq_folders = sorted(hybseq_folders, key=lambda x: x[-1])

    '''
                                                                
    for cycle in hybseq_folders:
        cycle_no = helpers.get_trailing_number(cycle)

        cycle_path = dataset_path + cycle + '/' + maxproj_folder + '/'
        all_pos = helpers.list_files(cycle_path)

        for position_name in all_pos:
            segment = re.search('MAX_Pos(.*).tif', position_name)
            pos = segment.group(1)
            if cycle_no == 1:
                helpers.quick_dir(original_path, 'Pos' + pos)
            new_pos_path = original_path + 'Pos' + pos + '/'
            shutil.copy(cycle_path + position_name, new_pos_path + 'hybseq' + str(cycle_no) + '.tif')

    '''

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

def align_cycles(dataset_path, max_checks=150, no_channels=4, registration_method='PHASE'):

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
    assert registration_method == 'PHASE' or 'ORB' or 'ECC', f'only options for registration are PHASE, ORB or ECC'
    print('Aligning cycles')
    align_tic = time.perf_counter()

    processed_path = helpers.quick_dir(dataset_path, 'processed')
    original_path = helpers.quick_dir(processed_path, 'original')
    denoised_path = helpers.quick_dir(processed_path, 'denoised')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    hyb_aligned_path = helpers.quick_dir(processed_path, 'hyb_aligned')
    hyb_path = helpers.quick_dir(processed_path, 'hyb')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    checks_alignment_path = helpers.quick_dir(checks_path, 'alignment')

    dataset_path = original_path
    #dataset_path = denoised_path

    all_positions = helpers.list_files(dataset_path)
    positions_dict = helpers.sort_position_folders(all_positions)

    no_channels = 4
    for key in positions_dict:
        positions = positions_dict[key]


        cycles = helpers.list_files(dataset_path + positions[0])
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
                position_path = helpers.quick_dir(dataset_path, position)
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

                #print('position and cycle, ref name', position, cycle, ref_name)
                #print('trans mat is ', transformation_matrix)
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

def bardensr_gene_basecalling(local_path, helper_files_path, no_channels = 4, basecalling_thresh=0.65, noisefloor=0.05, max_checks=100, flat_result=False):
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

    processed_path = helpers.quick_dir(local_path, 'processed')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    checks_basecalling_path = helpers.quick_dir(checks_path, 'basecalling')
    checks_bardensr_alignment_path = helpers.quick_dir(checks_path, 'bardensr_alignment')
    color_corr_checks_path = helpers.quick_dir(checks_path, 'color_corr')
    gene_basecalling_path = helpers.quick_dir(processed_path, 'gene_basecalling')
    segmentation_path = helpers.quick_dir(processed_path, 'segmentation')

    dataset_path = aligned_path
    positions = helpers.list_files(dataset_path)

    _, genebook, _ = prepare_codebook(helper_files_path)

    check_positions = random.sample(positions, min(max_checks, len(positions)))

    for position in positions:
        pos_tic = time.perf_counter()
        if position in check_positions: check_bool = True
        else: check_bool = False

        position_path = helpers.quick_dir(dataset_path, position)
        checks_basecalling_position_path = helpers.quick_dir(checks_basecalling_path, position)
        checks_bardensr_alignment_position_path = helpers.quick_dir(checks_bardensr_alignment_path, position)
        gene_basecalling_position_path = helpers.quick_dir(gene_basecalling_path, position)
        color_corr_checks_position_path = helpers.quick_dir(color_corr_checks_path, position)
        #segmentation_position_path = helpers.quick_dir(segmentation_path, position)

        all_cycles = helpers.list_files(position_path)
        cycles = [cycle for cycle in all_cycles if 'geneseq' in cycle]
        cycles = sorted(cycles, key=lambda x: x[:-1])

        sample_image = tif.imread(position_path + cycles[0])
        sample_image = sample_image.squeeze()

        X = np.empty((len(cycles), no_channels, 1, sample_image.shape[1], sample_image.shape[2]), dtype=sample_image.dtype)

        for index, cycle in enumerate(cycles):
            cycle_image = tif.imread(position_path + cycle)
            X[index, :, 0, :, :] = cycle_image[0:no_channels]

        flat_shape = len(cycles) * no_channels
        codeflat, _, unused_bc_ids = prepare_codebook(helper_files_path, flat_shape)

        Xflat = stack_preprocessing(X, check_bool, color_corr_checks_position_path, checks_bardensr_alignment_position_path, codeflat)

        #tif.imwrite(segmentation_position_path + position, np.max(Xflat, axis=0))

        evidence_tensor = bardensr.spot_calling.estimate_density_singleshot(Xflat, codeflat, noisefloor=noisefloor)
        result = bardensr.spot_calling.find_peaks(evidence_tensor, basecalling_thresh)

        genes_result = result.merge(genebook, left_on='j', right_on='geneID')
        del genes_result['j']
        #print('first few centers is', centers[:100])
        genes_result.to_csv(gene_basecalling_position_path + 'rolonies.csv')

        centers = genes_result[['m1', 'm2', 'geneID']].to_numpy(dtype=np.int)

        #centers = genes_result[['m1', 'm2']].to_numpy(dtype=np.int)
        R = genes_result[['R']].to_numpy(dtype=np.int)
        G = genes_result[['G']].to_numpy(dtype=np.int)
        B = genes_result[['B']].to_numpy(dtype=np.int)

        Xsmall = np.squeeze(X[0, 0:4])
        maxp = np.max(Xsmall, axis=0)
        image = np.zeros((sample_image.shape[1], sample_image.shape[2]), dtype=np.uint16)

        for rol_id in range(len(centers)):
            center = (int(centers[rol_id, 1]), int(centers[rol_id, 0]))
            if centers[rol_id, 2] < 106: color = 250
            else: color = 90
            cv2.circle(image, center, radius=2, color=color, thickness=-1)


        #image_rgb = Image.fromarray(np.uint8(np.dstack(image)))
        combined = np.zeros((2, sample_image.shape[1], sample_image.shape[2]), dtype=np.uint16)
        combined[0] = maxp.astype(np.uint16)
        combined[1] = image
        tif.imwrite(checks_basecalling_position_path + 'quick_look.tif', combined)

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

def allocate_rolonies(local_path, max_checks=100):
    model = models.Cellpose(model_type='cyto')

    processed_path = helpers.quick_dir(local_path, 'processed')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    segmented_checks_path = helpers.quick_dir(checks_path, 'segmentation')
    gene_basecalling_path = helpers.quick_dir(processed_path, 'gene_basecalling')
    segmentation_path = helpers.quick_dir(processed_path, 'segmentation')
    hybseq_path = helpers.quick_dir(local_path, 'hybseq')

    dataset_path = aligned_path
    positions = helpers.list_files(dataset_path)
    check_positions = random.sample(positions, min(max_checks, len(positions)))

    channels = [[0, 0]]

    for position in positions:
        pos_tic = time.perf_counter() #start the clock

        position_path = helpers.quick_dir(dataset_path, position)
        gene_basecalling_position_path = helpers.quick_dir(gene_basecalling_path, position)
        segmented_checks_position_path = helpers.quick_dir(segmented_checks_path, position)
        segmentation_position_path = helpers.quick_dir(segmentation_path, position)

        all_cycles = helpers.list_files(position_path)
        cycles = [cycle for cycle in all_cycles if 'geneseq' in cycle]
        cycles = sorted(cycles, key=lambda x: x[:-1])


        
        cycle_stack = tif.imread(position_path + cycles[0])
        #cycle_stack = tif.imread(segmentation_position_path + position)
        cycle_image = np.max(cycle_stack, axis=0)
        cycle_image = np.squeeze(cycle_image)

        hyb_image = tif.imread(hybseq_path + 'MAX_' + position + '.tif')
        hyb_image = hyb_image[0]
        print(hyb_image.shape)

        hyb_mask, flow, style, diam = model.eval(hyb_image, diameter=25.0, channels=channels, flow_threshold=0.4)
        tif.imwrite(segmentation_position_path + 'hyb_mask.tif', hyb_mask)

        image_mask, flow, style, diam = model.eval(cycle_image, diameter=25.0, channels=channels, flow_threshold=0.4)

        _, transformation_matrix = helpers.PhaseCorr_reg(image_mask, hyb_mask, return_warped=False)

        aligned_hyb_mask = np.zeros_like(hyb_image)

        aligned_hyb_mask = cv2.warpAffine(hyb_mask, transformation_matrix,
                                        (aligned_hyb_mask.shape[0], aligned_hyb_mask.shape[1]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        print('transformation matrix for', position, transformation_matrix)

        tif.imwrite(segmentation_position_path + 'mask.tif', aligned_hyb_mask)
        tif.imwrite(segmentation_position_path + 'cycle_mask.tif', image_mask)

        mask = tif.imread(segmentation_position_path + 'mask.tif')
        mask = mask.astype(np.uint8)
        ret, thresh = cv2.threshold(mask, 1, 255, 0)


        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        column_names = ['cellID', 'cell_X', 'cell_Y', 'cell_alloc', 'contour']
        cells_df = pd.DataFrame(columns=column_names)
        cells_df['contour'] = cells_df['contour'].astype('object')

        rolonies = pd.read_csv(gene_basecalling_position_path + 'rolonies.csv')
        rolonies_centroids = rolonies[['m1', 'm2']].to_numpy(dtype=np.int32)
        genes_cells = rolonies.copy()

        genes_cells['cellID'] = np.nan
        genes_cells['cell_X'] = np.nan
        genes_cells['cell_Y'] = np.nan
        genes_cells['cell_alloc'] = np.nan

        contours_mask = np.zeros_like(cycle_image)
        for cell_id, contour in enumerate(contours):
            cv2.drawContours(image=contours_mask, contours=contours, contourIdx=cell_id, color=120, thickness=1)
            M = cv2.moments(contour)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                cX, cY = 0, 0
            cells_df.at[cell_id, 'cellID'] = cell_id
            cells_df.at[cell_id, 'cell_X'] = cX
            cells_df.at[cell_id, 'cell_Y'] = cY
            cells_df.at[cell_id, 'contour'] = contour
            cells_df.at[cell_id, 'cell_alloc'] = 1

            for rol_ID, rolony_centroid in enumerate(rolonies_centroids):
                rolony_centroid = tuple([int(rolony_centroid[1]), int(rolony_centroid[0])])
                if cv2.pointPolygonTest(contour, rolony_centroid, False) >= 0:
                    genes_cells.at[rol_ID, 'cellID'] = cell_id
                    genes_cells.at[rol_ID, 'cell_X'] = cX
                    genes_cells.at[rol_ID, 'cell_Y'] = cY
                    genes_cells.at[rol_ID, 'cell_alloc'] = 1


        tif.imwrite(segmented_checks_position_path + 'cells_contour.tif', contours_mask)
        genes_cells.to_csv(gene_basecalling_position_path + 'genes_cells.csv')
        cells_df.to_csv(gene_basecalling_position_path + 'cells.csv')

        if position in check_positions:
            combined = np.ndarray((2, cycle_image.shape[0], cycle_image.shape[1]), dtype=np.uint16)
            combined[0] = cycle_image.astype(np.uint16)
            combined[1] = thresh.astype(np.uint16)
            tif.imwrite(segmented_checks_position_path + 'segmented.tif', combined)
        pos_toc = time.perf_counter()
        print('Segmentation of position', position, 'with', len(contours), 'cells done in ' + f'{pos_toc - pos_tic:0.4f}' + ' seconds')

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

def compute_stats(local_path):

    processed_path = helpers.quick_dir(local_path, 'processed')
    aligned_path = helpers.quick_dir(processed_path, 'aligned')
    checks_path = helpers.quick_dir(processed_path, 'checks')
    segmented_checks_path = helpers.quick_dir(checks_path, 'segmentation')
    gene_basecalling_path = helpers.quick_dir(processed_path, 'gene_basecalling')
    display_rolonies_path = helpers.quick_dir(processed_path, 'display_rolonies')
    slice_stats = helpers.quick_dir(processed_path, 'slice_stats')

    dataset_path = aligned_path
    positions = helpers.list_files(dataset_path)
    positions = helpers.human_sort(positions)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    positions_dict = helpers.sort_position_folders(positions)

    for position in positions_dict:
        print('Stats for position', position)
        slice_stats_position_path = helpers.quick_dir(slice_stats, position)
        rolonies = np.empty(shape=(0, 5))
        for tile in positions_dict[position]:
            gene_basecalling_position_path = helpers.quick_dir(gene_basecalling_path, tile)
            display_rolonies_position_path = helpers.quick_dir(display_rolonies_path, tile)
            genes_cells = pd.read_csv(gene_basecalling_position_path + 'genes_cells.csv')
            tile_rolonies = genes_cells[['m1', 'm2', 'geneID', 'cellID', 'cell_alloc']].to_numpy(dtype=np.int)
            rolonies = np.vstack((rolonies, tile_rolonies))

        plt.hist(rolonies[:,2])
        plt.savefig(slice_stats_position_path + 'genes_hist.png')
        plt.clf()
        rolonies_incells = rolonies[rolonies[:,4] == 1]
        cells, occurences = np.unique(rolonies_incells[:,3], return_counts=True)
        mean_occ = np.mean(occurences)
        std_occ = np.std(occurences)
        total_rolonies = rolonies.shape[0]
        alloc_rolonies = rolonies_incells.shape[0]
        np.save(slice_stats_position_path + 'mean_std.npy', np.array([total_rolonies, alloc_rolonies, mean_occ, std_occ]))
        np.savetxt(slice_stats_position_path + 'mean_std.txt', np.array([total_rolonies, alloc_rolonies, mean_occ, std_occ]), fmt='%i')
        print('for position', position,'total rol, alloc rol, mean and std of occurences', total_rolonies, alloc_rolonies, mean_occ, std_occ)


def stitch_images_imperfectly_folder(local_path, input_overlap = 0.15, specific_chan = None):
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

    #Starts from a folder where max projections are created. Read all position files and create a separate folder -stitched-

    processed_path = helpers.quick_dir(local_path, 'processed')
    display_rolonies_path = helpers.quick_dir(processed_path, 'display_rolonies')

    maxproj_path = display_rolonies_path
    path = pathlib.PurePath(maxproj_path)
    folder_name = 'stitched' + path.name + '/'
    stitched_path = helpers.quick_dir(maxproj_path + '../', folder_name)

    positions_list = helpers.list_files(maxproj_path)
    #Extract a sample image to get dimensions and number of channels.

    sample_image = tif.imread(maxproj_path + positions_list[0])

    if sample_image.ndim == 2 or specific_chan is not None:
        no_channels = 1
        pixel_dim = sample_image.shape[1]
    elif sample_image.ndim == 3:
        no_channels = min(sample_image.shape)
        pixel_dim = sample_image.shape[1]
    else:
        print('I only prepared this function for 2 or 3 channel images')

    #Sometimes images are not max projections, so their naming scheme is different. Namely, it's not 'MAX_Pos1_1_1', but just 'Pos1_1_1'.
    #Distinguish between the two cases by getting the starting segment before 'Pos'.
    segment = re.search('(.*)Pos(.*)', positions_list[0])
    starting_segment = segment.group(1)

    #Get a list of positions.
    positions_int = []
    for position_name in positions_list:
        segment = re.search(starting_segment + 'Pos(.*)_(.*)_(.*)', position_name)
        pos = segment.group(1); pos = int(pos);
        if pos not in positions_int:
            positions_int.append(pos)

    #Create a dictionary to allocate all tiles to a specific position
    keys_to_search = ['Pos' + str(pos_int) for pos_int in positions_int]
    positions_dict = {key: [] for key in keys_to_search}
    x_max_dict = {key: 0 for key in keys_to_search}
    y_max_dict = {key: 0 for key in keys_to_search}

    #Get max number of tiles in each dimension.
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

    #for each position, stitch images. start by stitching images into individual columns, and the stitch columns.
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

                #image = tif.imread(maxproj_path+ '/' + image_name + '/' + 'geneseq1.tiff')

                image = tif.imread(maxproj_path + '/' + image_name + '.tif')

                if image.shape[2] == no_channels:
                    image = np.transpose(image, (2, 0, 1))

                if specific_chan is not None: image = image[specific_chan, :, :]
                if image.ndim == 2: image = np.expand_dims(image, axis=0)

                if row != (x_max-1):
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

        #stitched = stitched.astype('uint8')
        stitched = stitched.astype(np.uint16)
        tif.imwrite(stitched_path + position + '.tif', stitched)
        toc = time.perf_counter()
        print('stitching of ' + position + ' finished in ' + f'{toc - tic:0.4f}' + ' seconds')
    print('Stitching done.')

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

    return XcolorCorr

def register_imagestack(Xnorm, codeflat):
    corrections = bardensr.registration.find_translations_using_model(Xnorm, codeflat, niter=50)
    Xnorm_registered, newt = bardensr.registration.apply_translations(Xnorm, corrections.corrections)
    print('corrections are ', corrections.corrections)
    print('newt is ', newt)
    return Xnorm_registered

def stack_preprocessing(X, check_bool, color_corr_checks_position_path, checks_bardensr_alignment_position_path, codeflat):
    [rounds, channels, z, x, y] = np.shape(X)
    # Colorbleeding correct and subtract background
    XcolorCorr = color_correct(X)

    Xflat = XcolorCorr.reshape((rounds * channels,) + XcolorCorr.shape[-3:])

    Xnorm = bardensr.preprocessing.minmax(Xflat)
    Xnorm = bardensr.preprocessing.background_subtraction(Xnorm, [0, 10, 10])
    Xnorm = bardensr.preprocessing.minmax(Xnorm)

    #Xnorm = register_imagestack(Xnorm, codeflat)

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

        tif.imwrite(checks_bardensr_alignment_position_path + 'before_bardensr_alignment.tif', X.astype(np.int16))
        tif.imwrite(checks_bardensr_alignment_position_path + 'after_bardensr_alignment.tif', Xnorm.astype(np.int16))

    return Xnorm

def prepare_codebook(helper_files_path, no_channels=4):
    '''
    Prepare codebook to match the number of channels and cycles we have. also, generate a randon rgb code for each
    one of them to visualise later on. Get ids for unused barcodes. Starting point is a bit awkward and I need to change it.
    Codebookforbardensr is generated by a matlab scrips from codebook.mat. need to change this to matlab
    :param helper_files_path: location of files
    :param no_channels:...
    :return: flatcodebook, genebook and unused barcode ids to use later on to assess error rate.
    '''

    codebook = sp.io.loadmat(helper_files_path + '/' + 'codebookforbardensr.mat')
    codebook = codebook['codebookbin1'] > 0
    codebook = np.moveaxis(np.moveaxis(codebook, 0, -1), 0, -2)
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

    #unused_bc_ids = [106, 107, 108, 109, 110]
    #unused_bc_ids = [0,1]
    unused_bc_ids = [158,159,160,161,162]

    return codeflat, genebook, unused_bc_ids




############################



def align_cycles_old(dataset_path, max_checks = 150, no_channels = 4):

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
