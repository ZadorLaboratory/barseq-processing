#!/usr/bin/env python


import os
import argparse
import re

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


def rename_tiles(folder, pos_offset):
    tiles = list_files(folder)
    for file in tiles:
        match = re.search(r'(xy\d+)(.*)(z\d+c\d+)', file)
        if match:
            prefix = match.group(1)
            suffix = match.group(3)
            new_number = int(prefix[2:]) + pos_offset
            new_prefix = f'xy{new_number:02}'
            new_filename = new_prefix + match.group(2) + suffix + '.tif'
            os.rename(os.path.join(folder, file), os.path.join(folder, new_filename))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename microscope image tiles')
    parser.add_argument('--folder', type=str, help='Folder with image tiles')
    parser.add_argument('--config', type=str, default='channels_config.py', help='Channel configuration file')

    args = parser.parse_args()


# load tiles_pos.csv

def main():
    pass
    #rename_tiles(args.folder, pos_offset=-1976)

main()