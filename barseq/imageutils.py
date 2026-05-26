#
#  Initial library to abstract out image handling. 
#  May allow substitution of file formats and tiff/imageio versions.
#  Will only work if version compatibility with the various tool environments is tolerant. 
#
#
#  Our standard is CHANNEL-FIRST   ( 5, 3200, 3200 )
#  channel, row, column    CXY
#
#
import logging
import os

import tifffile
from tifffile import imread, imwrite, TiffFile, TiffWriter

import imageio.v3 as iio
import numpy as np


def channel_names_index_map(select_channels, image_channels):
    '''

    Given ordered channel name list and list of desired channels, 
    return index list to pull from input image. 

    image_channels = ['GFP', 'YFP', 'TxRed', 'Cy5', 'DAPI', 'BF']
    select_channels = ['GFP', 'YFP', 'TxRed', 'Cy5']
    ->
    [0,1,2,3]

    select_channels = ['TxRed', 'GFP', 'YFP', 'Cy5']
    -> [2,0,1,3]
    '''
    idx_list = []
    for selname in select_channels:
        try:
            ch_idx = image_channels.index(selname)
            idx_list.append(ch_idx)
        except ValueError as ve:
            logging.error('Channel name {selname} not in image_channels={image_channels}. Check config. ')
            raise
    return idx_list


def read_image(infile, channels=None):
    '''
    BARseq standard image interface. 
    Intended to abstract out underlying formats and libraries. 
    image is numpy.ndarray, where shape = (channel, y|height , x|width )
    channels is list of integer *indexes*, starting at 0. 

    Caller is responsible for converting Channel numbers to indexes (i.e. C - 1)
    '''
    np_array = iio.imread(infile)
    #logging.debug(f'read image shape={np_array.shape} from {infile}')
    if channels is not None:
        if len(channels) == 1:
            new_array = np_array[channels[0]]
        else:
            new_array = np.ndarray( ( len(channels), np_array.shape[1], np_array.shape[2] ) ) 
            for i, channel in enumerate( channels ):
                new_array[i] = np_array[channel]
                logging.debug(f'reading channel idx={channel} shape={np_array.shape}')
        np_array = new_array
    return np_array

def write_image(outfile, np_array, photometric='minisblack'):
    '''
    BARseq standard image interface. 
    Intended to abstract out underlying formats and libraries. 
    image is numpy.ndarray, where shape = (channel, y|height , x|width )
    Assuming tif plugin, photometric = [ 'minisblack' | 'miniswhite'| 'rgb' ]
    '''
    # iio.v3 syntax. pass args as-is to underlying plugin
    iio.imwrite( outfile, np_array, photometric=photometric )
    
    # iio v2 syntax explicitly create plugin_kwargs dict. 
    #iio.imwrite( outfile, np_array, plugin_kwargs={"photometric": photometric})
    logging.debug(f'wrote image shape={np_array.shape} photometric={photometric} to {outfile}')
 
 
#   
# Ashlar-specific image handling.
#
    
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
    

#    
#  Bardensr specific image handling.
#
def bd_read_images(infiles, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    might be useful elsewhere...
    
    '''
    I = []
    for infile in infiles:
        for j in range(C):
            I.append( np.expand_dims( read_image( infile, channels=[j]), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[ :, :, round(ny*cropf):round(ny*(1-cropf)), round(nx*cropf):round(nx*(1-cropf)) ]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:, :, trim:-trim, trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    logging.debug(f'created image stack dimensions={I.shape}')
    return I


def bd_read_image_single(infile, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    might be useful elsewhere...
    
    '''
    I = []
    for i in range(1, R+1):
        for j in range(C):
            I.append( np.expand_dims( read_image( infile, channels=[j]), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[ :, :, round(ny*cropf):round(ny*(1-cropf)), round(nx*cropf):round(nx*(1-cropf)) ]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:, :, trim:-trim, trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    return I