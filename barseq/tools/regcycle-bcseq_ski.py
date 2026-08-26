#!/usr/bin/env python
#
#
import argparse
import logging
import os
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np
import skimage
from skimage.exposure import match_histograms
from skimage.filters import median, difference_of_gaussians
from skimage.registration import phase_cross_correlation as pcc


from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def regcycle_bcseq_ski(infiles, outfiles, template=None, stage=None, cp=None ):
    '''
    @arg infiles    tiles across 1 or more cycles
    @arg outdir     TOP-LEVEL out directory
    @arg template   optional file to use as template against infiles, 
                    otherwise register to first. 
    @arg cp         ConfigParser object
    @arg stage      stage label in cp
    
    template will be geneseq01/ single file. 

    '''
    if cp is None:
        cp = get_default_config()
    
    if stage is None:
        stage = 'regcycle-bcseq'

    # dapi_shift=[], dapi_ch_num=5, num_c_hyb=6, num_c=4, is_affine=0, radius=31, ch_i=2
    # 
    image_type = cp.get(stage, 'image_type')
    channel_names =  get_config_list(cp, image_type, 'channels')
    reg_channels = get_config_list(cp, stage, 'reg_channels')
    reg_indexes = channel_names_index_map(reg_channels, channel_names)

    select_channels = get_config_list(cp, stage, 'select_channels')
    select_indexes = channel_names_index_map(select_channels, channel_names)
    
    template_image_type = cp.get( stage, 'template_image_type')
    template_channel_names = get_config_list(cp, template_image_type , 'channels')
    template_select_channels = get_config_list(cp, stage, 'template_select_channels')
    template_select_indexes = channel_names_index_map(  template_select_channels, 
                                                        template_channel_names)

    template_reg_channels = get_config_list(cp, stage, 'template_reg_channels')
    template_reg_indexes = channel_names_index_map(template_reg_channels, template_channel_names)

    logging.info(f' stage={stage} template={template}')
    logging.debug(f'select_channels={select_channels} select_indexes={select_indexes}') 
    logging.debug(f'template_select_channels={template_select_channels} template_select_indexes={template_select_indexes}')

    upsample_factor = cp.getint(stage, 'upsample_factor')
    logging.debug(f'upsample_factor={upsample_factor}')

    # Determine template(fixed) file name
    if template is None:
        template_file = infiles[0]
    else:
        template_file = template
    logging.debug(f'fixed_file = {template_file}')

    logging.debug(f'Reading template file: {template_file} channels={template_reg_indexes} ')
    template_image = read_image( template_file, template_reg_indexes)

    # Calculate transform using first image.
    logging.debug(f'Reading moving file: {infiles[0]} channels={reg_indexes} ') 
    moving = read_image(infiles[0], reg_indexes )
    moving = moving.copy()
    moving = difference_of_gaussians(moving, 2, 40)   
    moving_norm = np.divide(moving,np.max(moving,axis=None))
    moving = moving_norm.copy()
    moving = np.uint8(np.clip(moving*255,0,255))

    fixed = template_image.copy()
    fixed = difference_of_gaussians(fixed, 2, 40) 
    fixed_norm = np.divide(fixed,np.max(fixed,axis=None))
    fixed = fixed_norm.copy()
    fixed = np.uint8(np.clip(fixed*255,0,255))

    hmatched_moving = match_histograms(moving, fixed)

    shift_values,_,_ = pcc(fixed, hmatched_moving, upsample_factor=upsample_factor)
    tform_bc = skimage.transform.SimilarityTransform(translation=(-shift_values[1],-shift_values[0]))
    logging.debug(f'Calculated transform for {infiles[0]} to {template_file}: {tform_bc}')

    for j, infile in enumerate( infiles ):
        outfile = outfiles[j]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')

        logging.debug(f'Applying transform: {infile} -> {outfile}')
        bc_image = read_image(infile)
        logging.debug(f'Read {infile}. shape={bc_image.shape}')
        bc_aligned = np.zeros_like(bc_image)
        for i in range(bc_image.shape[0]):
            bc_aligned[i,:,:] =  skimage.transform.warp( 
                                    np.squeeze( bc_image[i,:,:]), 
                                    tform_bc, 
                                    preserve_range = True,
                                    output_shape = (fixed.shape[0],fixed.shape[1])
                                )
        logging.info(f'writing to {outfile}')
        write_image(outfile, bc_aligned)
        logging.debug(f'done writing {outfile}')



# Notebook code
def align_bc_to_gene(pthb,bcfname='n2vbcseq', genefname='n2vgene',bcidx=4,geneidx=4,ch_bc=4,ch_bc_init=5):
    """
    Preprocessing function:
    calls single_slign_bc_to_gene to align bcseq0 to geneseq0 for all tiles
    writes transformation matrix per tile
    """
    os.makedirs(os.path.join(pthb,'processed','BCtoGeneAlignementComparison'),exist_ok=True)
    # ALSO, I AM ONLY ALIGNING RAW FILES HERE-NOT N2V-DOESN'T MATTER BECAUSE BF IS NOT CHANGED-WHAT HAPPENS WHEN WE SWITCH TO DAPI IS ANOTHER STORY
    #organize_files(pthb,cyclename='bcseq')
    folders,_,_,_=get_folders(pthb)
    for i in range(len(folders)):
        pthl=os.path.join(pthb,'processed',folders[i])
        [shutil.move(f1,os.path.join(pthl,f1.split('/')[-1])) for f1 in glob.glob(os.path.join(pthl,'original','n2v*.tif'))]
        [os.remove(f) for f in glob.glob(os.path.join(pthl,'*reg*.tif'))]
        tform_bc=single_align_bc_to_gene(pthl,bcfname,genefname,bcidx,geneidx,ch_bc,ch_bc_init)
        dump(tform_bc,os.path.join(pthl,'tforms_bc_to_gene.joblib'))
        [shutil.move(f1,os.path.join(pthb,'processed','BCtoGeneAlignementComparison',bcfname+'2'+genefname+'_'+folders[i]+'.tif')) for f1 in glob.glob(os.path.join(pthl,'comp.tif'))]

def single_align_bc_to_gene(pth,bcfname='bcseq', genefname='gene',bcidx=4,geneidx=4,ch_bc=4,ch_bc_init=5):
    """
    Preprocessing function:
    Aligns bcseq0 to geneseq0, reference channels-brightfield at this point, using phase correlation 
    based registration, writes merged file in ??? 
    [NEED TO CHECK WHERE IS COMP FILE WRITTEN]-operates on one tile
    Writes aligned file
    """
    bcfiles=sorted(glob.glob(os.path.join(pth,'*'+bcfname+'*.tif')))
    genefiles=sorted(glob.glob(os.path.join(pth,'*'+genefname+'*.tif')))
    filename=bcfiles.copy()
    
    geneim=tfl.imread(genefiles[0],key=geneidx)
    bcim=tfl.imread(bcfiles[0],key=bcidx)
       
    moving=bcim.copy()
    moving= difference_of_gaussians(moving, 2, 40) 
    moving_norm=np.divide(moving,np.max(moving,axis=None))
    moving=moving_norm.copy()
    fixed=geneim.copy()
    fixed= difference_of_gaussians(fixed, 2, 40) 
    fixed_norm=np.divide(fixed,np.max(fixed,axis=None))
    fixed=fixed_norm.copy()
    moving=np.uint8(np.clip(moving*255,0,255))
    fixed=np.uint8(np.clip(fixed*255,0,255))
    
    hmatched_moving = match_histograms(moving, fixed)
    
    shift_values,_,_ = pcc(fixed, hmatched_moving, upsample_factor=100)
    
    tform_bc=skimage.transform.SimilarityTransform(translation=(-shift_values[1],-shift_values[0]))
    aligned=np.zeros((3,fixed.shape[0],fixed.shape[1]),dtype=np.double)
    # aligned[0,:,:]=uint16m(geneim)
    # aligned[1,:,:]=uint16m(skimage.transform.warp(bcim,tform_bc,preserve_range=True,output_shape=(fixed.shape[0],fixed.shape[1])))
    # tfl.imwrite(os.path.join(pth,'comp.tif'),aligned,photometric='minisblack')
    aligned[0,:,:]=np.double(fixed)*0.5
    aligned[1,:,:]=np.double(skimage.transform.warp(moving,tform_bc,preserve_range=True,output_shape=(fixed.shape[0],fixed.shape[1])))*0.5
     

    aligned=np.uint8(np.clip(aligned,0,255))
    tfl.imwrite(os.path.join(pth,'comp.tif'),np.transpose(aligned,axes=(1,2,0)),photometric='rgb')

    for j in range(len(bcfiles)):
        if j==0:
            ch_r=ch_bc_init
        else:
            ch_r=ch_bc
        Ibc=tfl.imread(bcfiles[j],key=range(0,ch_r,1))
        Ibc_aligned=np.zeros_like(Ibc)
        for i in range(Ibc.shape[0]):
            Ibc_aligned[i,:,:]=skimage.transform.warp(np.squeeze(Ibc[i,:,:]),tform_bc,preserve_range=True,output_shape=(fixed.shape[0],fixed.shape[1]))
        tfl.imwrite(os.path.join(pth,'reg'+filename[j].split('/')[-1]),Ibc_aligned,photometric='minisblack')
    return tform_bc

    
if __name__ == '__main__':
    FORMAT='%(asctime)s (UTC) [ %(levelname)s ] %(filename)s:%(lineno)d %(name)s.%(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-d', '--debug', 
                        action="store_true", 
                        dest='debug', 
                        help='debug logging')

    parser.add_argument('-v', '--verbose', 
                        action="store_true", 
                        dest='verbose', 
                        help='verbose logging')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')
    
    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')

    parser.add_argument('-t','--template', 
                    metavar='template',
                    default=None,
                    required=False, 
                    type=str, 
                    help='label for this stage config')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='outfile. ')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        loglevel = 'debug'
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   
        loglevel = 'info'
    
    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    logging.debug(f'Running with config={args.config}:\n{cdict}')

    (outdir, file) = os.path.split(args.outfiles[0])
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")
    regcycle_bcseq_ski( infiles=args.infiles,  
                  outfiles=args.outfiles,
                  template=args.template, 
                  stage=args.stage, 
                  cp=cp )
    
    logging.info(f'done processing output to {outdir}')