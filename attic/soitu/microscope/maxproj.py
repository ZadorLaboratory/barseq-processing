#!/usr/bin/env python
import argparse
import numpy as np
import tifffile as tif
import os


def str2list(v):
    v = v.replace('[', '')
    v = v.replace(']', '')
    v = v.replace("'", '')
    v = v.replace('"', '')
    v = v.replace(' ', '')
    v = v.split(',')
    return v

def create_maxproj(args):
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










if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create max projections')
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('--channels_order', type=str2list, default=['G', 'T', 'A', 'C', 'DIC'], help='Channels order')

    args = parser.parse_args()
    create_maxproj(args)
