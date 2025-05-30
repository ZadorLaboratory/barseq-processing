#
#  Initial library to abstract out image handling. 
#  May allow substitution of file formats and tiff/imageio versions.
#  Will only work if version compatibility with the various tool environments is tolerant. 
#
import logging
import os

import tifffile
from tifffile import imread, imwrite, TiffFile, TiffWriter

import imageio.v3 as iio

def read_image(infile):
    np_array = iio.imread(infile)
    logging.debug(f'read image shape={np_array.shape} from {infile}')
    return np_array


def write_image(outfile, np_array):
    iio.imwrite( outfile, np_array, photometric='minisblack' )
    logging.debug(f'wrote image shape={np_array.shape} to {outfile}')
    
    
def write_mosaic(mosaic, outfile):
    #for ci, channel in enumerate(self.mosaic.channels):
    #channel = 0
    if self.verbose:
        logging.info(f"Assembling channel {channel}:")
    img = self.mosaic.assemble_channel(channel)
    img = uint16m(img)
    images.append(img)
    img = None
    #(dirpath, base, ext) = split_path(self.outpath)
    #outfile = os.path.join(dirpath, f'{base}.{channel}.{ext}')
    logging.debug(f'Added channel {channel} to image list.')

    fullimage = np.dstack(images)
    logging.debug(f'dstack() -> {fullimage.shape}')
    # produces e.g. shape = ( 3200,3200,5)
    fullimage = np.rollaxis(fullimage, -1)
    logging.debug(f'rollaxis() -> {fullimage.shape}')
    # produces e.g. shape = ( 5, 3200, 3200)  