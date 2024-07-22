#!/usr/bin/env python

import argparse
import numpy as np
import pathlib
import re
import tifffile as tif
import os
import time
import cv2



def stitch_tiles(filepath, input_overlap=0.25, split_ratio=0.5, specific_chan=None, output_folder_name=None, output_path=None, reduce_size=True, reduce_size_factor=4):
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
        stitched_path = quick_dir(filepath + '../', folder_name)
    else:
        stitched_path = quick_dir(output_path, folder_name)

    positions_list = list_files(maxproj_path)
    # Extract a sample image to get dimensions and number of channels.

    sample_image = tif.imread(maxproj_path + positions_list[0])

    if (sample_image.ndim == 2) or (specific_chan is not None):
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
        pixel_dim = sample_image.shape[1]

        x_max = x_max_dict[position] + 1
        y_max = y_max_dict[position] + 1

        reduced_pixel_dim = int((1 - input_overlap) * pixel_dim)

        x_pixels = (x_max - 1) * reduced_pixel_dim + pixel_dim
        y_pixels = (y_max - 1) * reduced_pixel_dim + pixel_dim

   

        stitched = np.empty((no_channels, x_pixels, y_pixels), dtype=np.float16)
        stitched_cols = np.empty((no_channels, y_max, x_pixels, pixel_dim), dtype=np.float16)

        buffer = int(split_ratio * input_overlap * pixel_dim)
        pixel_dim -= buffer
        for column in range(y_max):
            for row in range(x_max):
                # image_name = starting_segment + position + '_00' + str(y_max - column - 1) + '_00' + str(x_max - row - 1)
                #image_name = starting_segment + position + f'_{(y_max - column - 1):03d}_{(x_max - row - 1):03d}'
                image_name = starting_segment + position + f'_{(y_max - column - 1):03d}_{(row):03d}'
                #print(image_name)
                #image_name = starting_segment + position + f'_{(x_max - row - 1):03d}_{(column):03d}'

                # image = tif.imread(maxproj_path+ '/' + image_name + '/' + 'geneseq1.tiff')

                image = tif.imread(maxproj_path + '/' + image_name + '.tif')

                if np.where(image.shape == no_channels) == 2:
                    image = np.transpose(image, (2, 0, 1))

                if specific_chan is not None:
                    image = image[specific_chan, :, :]
                if image.ndim == 2: image = np.expand_dims(image, axis=0)
                
                if row == 0:
                    stitched_cols[:, column, :buffer + reduced_pixel_dim, :] = image[:, 0:buffer + reduced_pixel_dim, :]

                if row != (x_max - 1):                   
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

        # stitched = stitched.astype('uint8')
        stitched = stitched.astype(np.uint16)
        if reduce_size is True:
            if no_channels > 1:
                resized = np.zeros((stitched.shape[0], int(stitched.shape[1] / reduce_size_factor),
                                    int(stitched.shape[2] / reduce_size_factor)), dtype=np.int16)
                for i in range(stitched.shape[0]):
                    resized[i, :, :] = cv2.resize(stitched[i, :, :], (int(stitched.shape[2] / reduce_size_factor),
                                                                      int(stitched.shape[1] / reduce_size_factor)))

                    # resized[i] = cv2.resize(stitched[i], (resized.shape[1], resized.shape[2]))
            else:
                resized = cv2.resize(stitched[0], (int(stitched.shape[2] / reduce_size_factor), int(stitched.shape[1] / reduce_size_factor)))
            stitched = resized

        tif.imwrite(stitched_path + position + '.tif', stitched)
        toc = time.perf_counter()
        print('stitching of ' + position + ' finished in ' + f'{toc - tic:0.4f}' + ' seconds')
    print('Stitching done.')

    return stitched_path


def quick_dir(location, folder_name):
    '''
    Check if a directory exists, otherwise creates one
    :param location:
    :param name:
    :return:
    '''

    if location[-1] == '/': folder_path = location + folder_name + '/'
    else: folder_path = location + '/' + folder_name + '/'

    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)

    return folder_path

def list_files(path):
    files = os.listdir(path)
    if '.DS_Store' in files: files.remove('.DS_Store')
    return files
                 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create tiles from input file')
    parser.add_argument('--filepath', type=str, help='Input file with positions')
    parser.add_argument('--input_overlap', type=float, default=0.25, help='Overlap between tiles')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='Ratio of overlap to split between tiles')
    parser.add_argument('--specific_chan', type=int, default=None, help='Number of tiles in the xdirection')
    parser.add_argument('--output_folder_name', type=str, default='', help='Number of tiles in the y direction')
    parser.add_argument('--output_path', type=str, default='', help='size of each tile')
    parser.add_argument('--reduce_size_factor', type=float, default=1, help='downsample images by this factor')


    args = parser.parse_args()

    stitch_tiles(filepath=args.filepath, input_overlap=args.input_overlap, split_ratio=args.split_ratio, specific_chan=args.specific_chan, reduce_size=True, reduce_size_factor=args.reduce_size_factor)

    