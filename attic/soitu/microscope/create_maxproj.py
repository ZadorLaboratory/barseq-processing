#!/usr/bin/env python

import numpy as np
import tifffile as tif
import os

import argparse

def str2list(v):
    v = v.replace('[', '')
    v = v.replace(']', '')
    v = v.replace("'", '')
    v = v.replace('"', '')
    v = v.replace(' ', '')
    v = v.split(',')
    return v


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


def create_maxproj1(args):
    if not os.path.exists(args.input_path + '/maxproj'):
        os.makedirs(args.input_path + '/maxproj')

    maxproj_path = args.input_path + '/maxproj/'

    #list all folders containing 'Pos'
    positions = [f for f in os.listdir(args.input_path) if 'Pos' in f]

    for position in positions:
        position_path = os.path.join(args.input_path, position)
        print('Processing position: ', position)
        # list all files ending with .tif
        files = [f for f in os.listdir(position_path) if f.endswith('.tif')]
        keys_to_search = ['-' + channel + '_' for channel in args.channels_order]

        stacks_dict = {key: [] for key in keys_to_search}

        for file in files:
            for key in keys_to_search:
                if key in file:
                    stacks_dict[key].append(file)

        sample_image = tif.imread(position_path + '/' + stacks_dict[keys_to_search[0]][0])
        x_pixels, y_pixels = sample_image.shape[-2], sample_image.shape[-1]
        maxproj_file = np.zeros((len(args.channels_order), x_pixels, y_pixels), dtype=np.uint16)

        for index, stack_key in enumerate(stacks_dict):
            #find index of stack_key in args.channels_order
            #remove '-' and '_' from stack_key
            
            channel_key = stack_key[1:-1]
            channel_id = args.channels_order.index(channel_key)
            stack = stacks_dict[stack_key]
            
            stack_image = np.zeros((len(stack), x_pixels, y_pixels), dtype=np.uint16)
            for i, file in enumerate(stack):
                stack_image[i] = tif.imread(os.path.join(position_path, file))
            
            maxproj = np.max(stack_image, axis=0)
            maxproj_file[channel_id] = maxproj   
        tif.imwrite(maxproj_path + 'MAX_' + position + '.tif', maxproj_file)


def create_maxproj(args):
    maxproj_path = quick_dir(args.input_path + '../', 'maxproj')
    GFP_folder = quick_dir(args.input_path + '../', 'GFP')
    RFP_folder = quick_dir(args.input_path + '../', 'RFP')
    DIC_folder = quick_dir(args.input_path + '../', 'DIC')
    GFP_maxproj_folder = quick_dir(args.input_path + '../', 'GFP_maxproj')
    RFP_maxproj_folder = quick_dir(args.input_path + '../', 'RFP_maxproj')
    DIC_maxproj_folder = quick_dir(args.input_path + '../', 'DIC_maxproj')
    
    tiles = list_files(args.input_path)
    
    #read in pos_matrix.mat

    #labels_list = 
    #labels_list = labels_list[labellist]

    # copy all ending in c1.tif to a 'GFP' folder without the c1 ending
    # copy all ending in c2.tif to a 'RFP' folder without the c2 ending
    # copy all ending in c3.tif to a 'DIC' folder without the c3 ending
    import shutil
    for tile in tiles:
        if tile.endswith('c1.tif'):
            shutil.copy(args.input_path + tile, GFP_folder + tile[:-6] + '.tif')
        elif tile.endswith('c2.tif'):
            shutil.copy(args.input_path + tile, RFP_folder + tile[:-6] + '.tif')
        elif tile.endswith('c3.tif'):
            shutil.copy(args.input_path + tile, DIC_folder + tile[:-6] + '.tif')


    # max proj all the tiles with the same xy for gfp and rfp files
    # save to the same folder
    gfp_tiles = list_files(GFP_folder)
    rfp_tiles = list_files(RFP_folder)
    dic_tiles = list_files(DIC_folder)

    import re
    xy_dict = {}
    for tile in gfp_tiles:
        # xy()z().tif
        segment = re.search('xy(.*)z(.*)\.tif', tile)
        xy = segment.group(1)        
        z = segment.group(2)
        if xy not in xy_dict: xy_dict[xy] = []
        xy_dict[xy].append(tile)

    for xy in xy_dict:
        stack = []
        for tile in xy_dict[xy]:
            stack.append(tif.imread(GFP_folder + tile))
        maxproj = np.max(stack, axis=0)
        tif.imwrite(GFP_maxproj_folder + 'MAX_' + xy + '.tif', maxproj)
        # delete all the tiles in the stack

    xy_dict = {} 
        
    for tile in rfp_tiles:
        # xy()z().tif
        segment = re.search('xy(.*)z(.*)\.tif', tile)
        xy = segment.group(1)        
        z = segment.group(2)
        if xy not in xy_dict: xy_dict[xy] = []
        xy_dict[xy].append(tile)

    for xy in xy_dict:
        stack = []
        for tile in xy_dict[xy]:
            stack.append(tif.imread(RFP_folder + tile))
        maxproj = np.max(stack, axis=0)
        tif.imwrite(RFP_maxproj_folder + 'MAX_' + xy + '.tif', maxproj)
        
    # for dic select the one with z4 and stack with np.min(z3-z5) and np.std(z1-z5)
    # save to dic_maxproj folder without the z ending
    xy_dict = {}
    for tile in dic_tiles:
        # xy()z().tif
        segment = re.search('xy(.*)z(.*)\.tif', tile)
        xy = segment.group(1)        
        z = segment.group(2)
        if xy not in xy_dict: xy_dict[xy] = []
        xy_dict[xy].append(tile)
    for xy in xy_dict:
        std_stack = []
        min_stack = []
        maxproj_stack = []

        for tile in xy_dict[xy]:
            std_stack.append(tif.imread(DIC_folder + tile))
            if tile.endswith('z3.tif') or tile.endswith('z4.tif') or tile.endswith('z5.tif'):
                min_stack.append(tif.imread(DIC_folder + tile))
            if tile.endswith('z4.tif'):
                maxproj_stack.append(tif.imread(DIC_folder + tile))
        std_stack = np.std(std_stack, axis=0)
        std_stack = std_stack.astype(np.uint16)
        min_stack = np.min(min_stack, axis=0)
        maxproj_stack.append(min_stack)
        maxproj_stack.append(std_stack)
        # turn maxproj_stack into a numpy array
        maxproj_stack = np.array(maxproj_stack)
        tif.imwrite(DIC_maxproj_folder + 'MAX_' + xy + '.tif', maxproj_stack)

            
    # delete files in DIC, GFP, RFP folders
    shutil.rmtree(GFP_folder)
    shutil.rmtree(RFP_folder)
    shutil.rmtree(DIC_folder)

    # in maxproj folder, stack gfp, rfp, dic into one file
    # save to maxproj folder
    files = list_files(GFP_maxproj_folder)
    for file in files:
        gfp = tif.imread(GFP_maxproj_folder + file)
        rfp = tif.imread(RFP_maxproj_folder + file)
        dic = tif.imread(DIC_maxproj_folder + file)
        maxproj = np.zeros((5, gfp.shape[0], gfp.shape[1]), dtype=np.uint16)
        maxproj[0] = gfp
        maxproj[1] = rfp
        maxproj[2:] = dic
        tif.imwrite(maxproj_path + file, maxproj)

    # delete files in GFP_maxproj_folder, RFP_maxproj_folder, DIC_maxproj_folder
    shutil.rmtree(GFP_maxproj_folder)
    shutil.rmtree(RFP_maxproj_folder)
    shutil.rmtree(DIC_maxproj_folder)

    # read in pos_matrix.mat
    import scipy.io as sio
    labels_list = sio.loadmat(args.pos_matrix)['labellist']
    labels_list = [x[0][0] for x in labels_list]

    print(labels_list)
    # for each file in maxproj folder, find the corresponding xy in labels_list
    # rename the file to MAX_xy.tif
    files = list_files(maxproj_path)
    for file in files:
        segment = re.search('MAX_(.*)\.tif', file)
        xy = segment.group(1)
        xy = int(xy)
        os.rename(maxproj_path + file, maxproj_path + 'MAX_' + labels_list[xy-1] + '.tif')
        print('Renaming file: ', file, ' to ', 'MAX_' + labels_list[xy-1] + '.tif')
    

    # replace xy with the corresponding row in labels_list + 1


def rename_files(args):
    maxproj_path = quick_dir(args.input_path + '../', 'maxproj')
    import scipy.io as sio
    import re
    labels_list = sio.loadmat(args.pos_matrix)['labellist']
    labels_list = [x[0][0] for x in labels_list]

    print(labels_list)
    # for each file in maxproj folder, find the corresponding xy in labels_list
    # rename the file to MAX_xy.tif
    files = list_files(maxproj_path)
    for file in files:
        segment = re.search('MAX_(.*)\.tif', file)
        xy = segment.group(1)
        xy = int(xy)
        os.rename(maxproj_path + file, maxproj_path + 'MAX_' + labels_list[xy-1] + '.tif')
        print('Renaming file: ', file, ' to ', 'MAX_' + labels_list[xy-1] + '.tif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create max projections')
    parser.add_argument('--input_path', default='', type=str, help='Input path')
    parser.add_argument('--pos_matrix', default='D:\barseq\CSWB1\Posinfo_regoffsetgeneseq04.mat', type=str, help='Input path')

    #parser.add_argument('--maxproj_path', default='/Users/soitu/Desktop/datasets/CSWB1/', type=str, help='maxproj path')

    #parser.add_argument('--channels_order', type=str2list, default=['G', 'T', 'A', 'C', 'DIC'], help='Channels order')
    parser.add_argument('--channels_order', type=str2list, default=['G', 'R', 'DIC'], help='Channels order')

    args = parser.parse_args()
    create_maxproj(args)
    #rename_files(args)
