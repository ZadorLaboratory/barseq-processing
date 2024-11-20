import os, glob, argparse
import numpy as np
import tifffile
from PIL import Image, ImageSequence
import shutil
import matplotlib.pyplot as plt
import imagej
ij=imagej.init('/***/Fiji.app/') # to your Fiji app location

parser = argparse.ArgumentParser()
parser.add_argument("--filedir", help="directory name where the new files will be saved") 
parser.add_argument("--folder", help="directory name (of where the original files are")
parser.add_argument("--ch", help="channels to stitch")



def main(folder,ch=3):
    for f in glob.glob(folder):
        filename = [s for s in f.split('/') if 'MAX' in s][0].split('.tif')[0]
        ch1_img = tifffile.imread(f)[ch]
        savefilename = f'{filedir}/{filename}_ch.tif'  # _ch indicates its for one channel
        tifffile.imwrite(savefilename, ch1_img)
        print(f'saved file {savefilename}')
            
    grid_size_x = 7
    grid_size_y = 1
    overlap=23
    total_tiles = grid_size_x * grid_size_y
    file_pattern = "MAX_Pos12_00{i}_001_ch.tif"
    
    args = {
        'type': "Grid: row-by-row",
        'order': "Left & Down",       
        'grid_size_x': grid_size_x,
        'grid_size_y': grid_size_y,
        'tile_overlap': overlap,  # note this is the percentage LOL
        'first_file_index_i': 0,
        'directory': filedir,
        'file_names': file_pattern,
        'output_textfile_name': "TileConfiguration.txt",
        'fusion_method': "Linear Blending",
        'regression_threshold': 0.30,
        'max/avg_displacement_threshold': 2.50,
        'absolute_displacement_threshold': 3.50,
        'compute_overlap': True,
        'computation_parameters': "Save memory (but be slower) ",
        'image_output': "Fuse and display",
    }
    
    
    plugin = "Grid/Collection stitching"
    ij.py.run_plugin(plugin, args=args)
    stitched_image = ij.py.active_imageplus()
    stitched_image_array = np.array(ij.py.to_xarray(stitched_image)).astype('float16')

    
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
        