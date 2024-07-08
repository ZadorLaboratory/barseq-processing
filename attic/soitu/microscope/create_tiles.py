#!/usr/bin/env python

import argparse
import csv
import json 
import os
import numpy as np

def create_tiles(input_file, overlap, xtiles, ytiles, tile_size):
    #Read positions from input file

    parent_folder = os.path.dirname(os.path.abspath(input_file))
    filename = os.path.basename(input_file)

    positions = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            x, y, z = map(float, row)
            positions.append((x, y, z))


    # Initialize position counter
    posn = 0

    # Loop over positions and create tiles
    tiles = {}
    tile_data = []
    pos_dict = {}
    order_data = np.zeros((xtiles, ytiles), dtype=object)
    coords_data = np.zeros((xtiles, ytiles), dtype=object)
    line_number = 0

    for i, (x, y, z) in enumerate(positions):
        # Compute top left corner of tile
        x0 = x + ((xtiles - 1) / 2) * tile_size
        y0 = y - ((ytiles - 1) / 2) * tile_size
        tile = []
        # Loop over tiles and write coordinates to CSV files
        for xi in range (xtiles):
            for yi in range(ytiles):
                px = x0 - xi * tile_size
                py = y0 + yi * tile_size
                pos_dict[str(line_number)] = f'Pos{posn:01d}_{xi:03d}_{yi:03d}'
                line_number += 1
                tile.append([px, py, z])
                if i == 0:
                    order_data[ytiles-yi-1, xtiles-xi-1] = f'({(xi):03d}, {(yi):03d})'
                    coords_data[ytiles-yi-1, xtiles-xi-1] = (px, py)
#                    order_row.append(f'({(xtiles-xi-1):03d}, {(ytiles-yi-1):03d})')
#                    coords_row.append((px, py))


                
        tile_data.extend(tile)
        posn += 1

    order_file = 'tiles_order.csv'
    coords_file = 'tiles_coords.csv'

    with open(os.path.join(parent_folder, order_file), 'w', newline='') as f:
        writer = csv.writer(f)
        for i, row in enumerate(order_data):
            writer.writerow(row)

    with open(os.path.join(parent_folder, coords_file), 'w', newline='') as f:
        writer = csv.writer(f)
        for i, row in enumerate(coords_data):
            writer.writerow(row)
            

    #save pos_dict to txt file
    with open(os.path.join(parent_folder, 'dict_' + filename + '.txt'), 'w') as f:
        f.write(json.dumps(pos_dict))

    #save tile_data to csv file
    with open(os.path.join(parent_folder, 'tiles_' + filename), 'w', newline='') as f:
        writer = csv.writer(f)
        for i, row in enumerate(tile_data):
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tiles from input file')
    parser.add_argument('input_file', type=str, help='Input file with positions')
    parser.add_argument('--overlap', type=float, default=0.15, help='Overlap between tiles')
    parser.add_argument('--xtiles', type=int, default=3, help='Number of tiles in the xdirection')
    parser.add_argument('--ytiles', type=int, default=3, help='Number of tiles in the y direction')
    parser.add_argument('--tile_size', type=float, default=0.82, help='size of each tile')

    args = parser.parse_args()

    create_tiles(args.input_file, args.overlap, args.xtiles, args.ytiles, args.tile_size)