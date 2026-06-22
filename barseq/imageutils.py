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


MAX_CHANNELS = 10
MIN_PIXELS = 100
SIM_THRESH = .95 

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
                logging.debug(f'reading channel idx={channel}')
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


def do_compare_images(infile1, infile2):
    '''
    Filename input. 
    '''
    a = read_image(infile1)
    b = read_image(infile2)
    images_identical, s , min_similarity = compare_images(a,b)
    return images_identical, s , min_similarity

def compare_image_matrix(im1, im2, chidx=0):

    s = ''
    images_identical = True
    images_similar = True
    min_similarity = 1.0

    # min()
    amin = int(im1.min())
    bmin = int(im2.min())
    dp = calc_proportion(amin, bmin)
    if dp < min_similarity:
        min_similarity = dp
    if dp >= SIM_THRESH:
        msg = f'   [{chidx}]: np.min() {dp} similar.\n'
        s += msg
        logging.info(msg)
    else :
        msg = f'   [{chidx}]: np.min() {dp} NOT similar.\n'
        s += msg
        images_similar = False

    # max()
    amax = int(im1.max())
    bmax = int(im2.max())
    dp = calc_proportion(amax, bmax)
    if dp < min_similarity:
        min_similarity = dp

    if dp >= SIM_THRESH:
        msg = f'   [{chidx}]: np.max() {dp} similar.\n'
        s += msg
        logging.info(msg)
    else :
        msg = f'   [{chidx}]: np.max() {dp} NOT similar.\n'
        s += msg
        images_similar = False

    # mean()
    amean = int(im1.mean())
    bmean = int(im2.mean())
    dp = calc_proportion(amean, bmean)
    if dp < min_similarity:
        min_similarity = dp

    if dp >= SIM_THRESH:
        msg =f'   [{chidx}]: np.mean() {dp} similar.\n'
        s += msg
        logging.info(msg)
    else :
        msg = f'   [{chidx}]: np.mean() {dp} NOT similar.\n'
        images_similar = False
        s += msg
        logging.info(msg)

    return images_similar, s, min_similarity


def compare_images( a, b):
    '''
    Direct image array input.
    Characterize the differences between two images. 
    Return True if they are *substantially* the same, False otherwise. 
    '''

    images_identical = True
    min_similarity = 1.0

    s = ''
    # type
    ta = type(a)
    tb = type(b)
    if ta != tb:
        images_identical = False
        msg = f'Data type differs: {ta} != {tb}\n'
        logging.info(msg)
        s += msg
        return images_identical, s , min_similarity
    else:
        msg = f'Data type same: {ta}\n'
        s += msg
        logging.debug(msg)
    
    # shape
    sa = a.shape
    sb = b.shape
    if len(sa) != len(sb):
        images_identical = False
        msg = f'Shape length differs: {sa} != {sb}\n'
        logging.info(msg)
        s += msg
        return images_identical, s , min_similarity
    else:
        msg = f'Same shape length: len({sa}) == len({sb}) -> {len(sa)}\n'
        s += msg
        logging.debug(msg) 

    # shape dimension values
    for i,d in enumerate(sa):
        da = sa[i]
        db = sb[i]
        if da != db:
            images_identical = False
            msg = f'shape[{i}] value differs: {da} != {db}\n'
            logging.info(msg)
            s += msg
            return images_identical, s , min_similarity
    msg = f'Same shape dimension values: {sa}\n'
    s += msg
    logging.debug(msg)

    # channels, axes sanity check. 
    if len(sa) == 3: 
        if sa[0] > MAX_CHANNELS:
            images_identical = False
            msg = f'First dimension too large for channels {sa[0]} > {MAX_CHANNELS}\n'
            s += msg
            logging.info(msg)
            return images_identical, s , min_similarity
        else:
            msg = f'First dimension consistent with channels: {sa[0]}\n'
            s += msg
            logging.debug(msg)
        if (sa[1] < MIN_PIXELS ) or (sa[2] < MIN_PIXELS):
            images_identical = False
            msg = f'Second or third dimension to small for pixels: {sa[1]} {sa[2]}\n'
            s += msg
            logging.info(msg)
            return images_identical, s , min_similarity
        else:
            msg = f'Dimensions 2,3 consistent w/ axes: {sa[1]} x {sa[2]}\n'
            s += msg
            logging.debug(msg)

        # Compare matrix statistics. min(), max(), mean(), median()
        # Measure is 90% similar.  
        # axis = 1 is rows, axis = 0 is columns
        images_similar = True
        for chidx in range(0, a.shape[0]):
            msg = f'checking channel {chidx}...\n'
            s += msg
            logging.debug(msg)
            ia = a[chidx]
            ib = b[chidx]

            images_similar, s , min_similarity = compare_image_matrix(ia, ib, chidx)

    elif len(sa) == 2:
        if (sa[0] < MIN_PIXELS ) or (sa[0] < MIN_PIXELS):
            images_identical = False
            msg = f'First and second dimension to small for pixels: {sa[0]} {sa[1]}\n'
            s += msg
            logging.info(msg)
            return images_identical, s , min_similarity 
        else:
            msg = f'Dimensions 1,2 consistent w/ axes: {sa[0]} x {sa[1]}\n'
            s += msg
            logging.debug(msg)                   

        images_similar, s , min_similarity = compare_image_matrix(a, b)

    # if overall sum() is identical, the images are exactly identical. 
    if a.sum() == b.sum():
        msg = f'Same full image sum: {int(a.sum())} Images are precisely identical.\n'
        s += msg
        logging.info(msg)
        return images_identical, s , min_similarity
    else:
        dp = calc_proportion(a.sum(), b.sum())
        if dp < min_similarity:
            min_similarity = dp
        images_identical = False
        msg = f'np.sum() {dp} similar-> detailed comparison...\n'
        s += msg
        logging.info(msg)




    # Summary
    if images_similar:
        msg = f'Images similar at {SIM_THRESH} level. min_similarity={min_similarity}'
        s += msg
        logging.info(msg)
    else:
        msg = f'Images NOT similar at {SIM_THRESH} level. min_similarity={min_similarity}'
        s += msg
        logging.info(msg)

    return images_identical, s , min_similarity


def calc_proportion(a, b):
    '''
        Calc proportion difference between a and b, round to sig figs. 
    '''
    a = int(a)
    b = int(b)
    if a == b:
        return 1.0
    elif (a == 0) or (b == 0):
        return calc_proportion(a+1, b+1)
    if a > b:
        return round(b / a , 3)
    else:
        return round( a / b, 3 )
 
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


def bd_read_image_set(infiles, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    assumes input is set of multiple cycles images for single tile

    '''
    I = []
    #for i in range(1, R+1):
    for infile in infiles:
        # logging.debug(f'reading {infile}')
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