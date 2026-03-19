#!/usr/bin/env python
#
# Do basecalling on batches of images.
# Intrinsically consumes multiple cycles, to output file is single for multiple
# inputs. So --outfile is only arg. 
#

import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

import numpy as np

import bardensr
import bardensr.plotting

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def basecall_bardensr( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    load codebook, 
    run bardensr, 
    output evidence tensor dataframe to <outdir>/<mode>/<prefix>.brdnsr.tsv   
    arity is single. 
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-geneseq'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    
    noisefloor_final = cp.getfloat(stage, 'noisefloor_final')
    intensity_thresh = cp.getfloat(stage, 'intensity_thresh')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    logging.debug(f'noisefloor_final={noisefloor_final} intensity_thresh={intensity_thresh} trim={trim} cropf={cropf}')

    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = cp.get(stage, 'codebook_bases').split(',')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_codebook_file(cfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles)
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook, codebook_bases, n_cycles=n_cycles)

    # CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    logging.debug(f'calculating max_per_RC...')
    max_per_RC=[ bd_read_image_single(infile, R, C, cropf=cropf).max(axis=(1,2,3)) for infile in infiles ]
    #max_per_RC=bd_read_images(infiles, R, C, cropf=cropf).max(axis=(1,2,3)) 
    #logging.debug(f'max_per_RC len={len(max_per_RC)} item len={len(max_per_RC[0])}')

    # Expected to be 28 values. channels * cycles. 
    # first max(), then median of those max() per cycle. 
    s = pprint.pformat(max_per_RC, indent=4)
    logging.debug(f'max per RC = {s}')
    median_max=np.median(max_per_RC, axis=0)
    logging.debug(f'median_max=\n{median_max}')
    #s = pprint.pformat(median_max, indent=4)
    #logging.debug(f'median_max = {s}')

    img_norm = bd_read_images(infiles, R, C, trim=trim ) / median_max[ :, None, None, None]
    logging.debug(f'img_norm shape={img_norm.shape}\ncodeflat={codeflat}\nnoisefloor_final={noisefloor_final}')
    et = bardensr.spot_calling.estimate_density_singleshot( img_norm, codeflat, noisefloor_final)
    logging.debug(f'estimated_density et = {et}')
    spots = bardensr.spot_calling.find_peaks( et, intensity_thresh, use_tqdm_notebook=False)
    spots.loc[:,'m1'] = spots.loc[:,'m1'] + trim
    spots.loc[:,'m2'] = spots.loc[:,'m2'] + trim            
    spots.to_csv(outfile, index=False)   
    logging.debug(f'wrote spots to outfile={outfile}')

   


# NOTEBOOK CODE

# R = 7 trim = 160 C = 4
def image_reader_trimmed(infile, trim, R, C):
    """
    Helper function for basecalling function for geneseq:
    Returns a border trimmed in xy image stack (with cycles and channels intact)
    """
    I=[]
    for i in range(1,R+1):
        for j in range(C):
            I.append(np.expand_dims(tfl.imread(infile,key=j),axis=0))
    I=np.array(I)
    return(I[:,:,trim:-trim,trim:-trim])

def bardensr_call(pth,config_pth,num_channels=4,codebook_name='codebookM1all.mat',fdrthresh=0.05,trim=160,cropf=0.4,noisefloor_ini=0.01,noisefloor_final=0.05):
    """
    Basecalling function for geneseq:
    1. Calls the function that creates binarized codebook as per the number of geneseq cycles in the experiment
    2. Calculates normalization factors, intensity thresholds for spot detection based on allowed fdr
    3. Calls Bardensr spot calling and peak detection to identify and decode rolonies per tile
    4. Saves basecalling results per tile
    """ 
    make_codebook_bin(pth,num_channels,codebook_name)

    controlidx=np.array(range(25,35))
    [folders,_,_,_]=get_folders(pth)
    
    codebook=scipy.io.loadmat(os.path.join(config_pth,codebook_name))['codebook']
    genes=np.array([str(x[0][0]) for x in codebook],dtype=str)
    cb=load(os.path.join(pth,'processed','codebookforbardensr.joblib'))
    cb=np.transpose(cb,axes=(1,2,0))
    R,C,J=cb.shape
    codeflat=np.reshape(cb,(-1,J))
    pos_unused_codes=np.where(np.char.startswith(genes,'unused'))
    err_codes=genes[pos_unused_codes]

    # SELECTING 10 FOLDERS AS CONTROL
    if len(folders)>=36:
        control_folders=[folders[k] for k in controlidx]
    else:
        control_folders=folders

    # NORMALIZATION-PREPROCESSING--CALCULATING MAX OF EACH CYCLE AND EACH CHANNEL ACROSS ALL CONTROL FOVS
    max_per_RC=[image_reader_cropped(os.path.join(pth,'processed',i,'aligned'),R,C,cropf).max(axis=(1,2,3)) for i in control_folders]
    median_max=np.median(max_per_RC,axis=0)

    # ESTABLISHING BASE THRESHOLD AT THE MEDIAN OF MAXIMUM ERROR READOUT 

    err_max=[]
    evidence_tensors=[]
    for folder in control_folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_trimmed(os.path.join(pth,'processed',folder,'aligned'),trim,R,C)/median_max[:,None,None,None],codeflat,noisefloor_ini)
        err_max.append(et[:,:,:,pos_unused_codes].max(axis=(0,1,2)))
    err_max=np.array(err_max)
    thresh=np.median(np.median(err_max,axis=1))
    print(thresh)

    # FINDING OPTIMUM TRHESHOLD WITH LOWEST FDR ON CONTROL FOLDERS

    err_c_all=[]
    total_c_all=[]
    for folder in control_folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_cropped(os.path.join(pth,'processed',folder,'aligned'),R,C,cropf)/median_max[:,None,None,None],codeflat,noisefloor_final)
        for thresh1 in np.linspace(thresh-0.1,thresh+0.1,10):
            spots=bardensr.spot_calling.find_peaks(et,thresh1,use_tqdm_notebook=False)
            spots.to_csv(os.path.join(pth,'processed',folder,'aligned','spots.csv'),index=False)
            print(f"in {folder} found {len(spots)} spots")
            err_c=0
            for err_idx in pos_unused_codes[0]:
                err_c=err_c+(spots.j==err_idx).to_numpy().sum()
            err_c_all.append(err_c)
            total_c_all.append(len(spots)-err_c)      
    #calculate fdr        
    err_c_all1=np.reshape(err_c_all,[len(control_folders),10])
    total_c_all1=np.reshape(total_c_all,[len(control_folders),10])+1
    fdr=err_c_all1/len(pos_unused_codes[0])*(len(genes)-len(pos_unused_codes[0]))/(total_c_all1)
    fdrmean=err_c_all1.mean(axis=0)/len(pos_unused_codes[0])*(len(genes)-len(pos_unused_codes[0]))/(total_c_all1.mean(axis=0))
    thresh_refined=np.linspace(thresh-0.1,thresh+0.1,10)[(fdrmean<fdrthresh).nonzero()[0][0]]#this is the new threshold optimized by targeted fdr value
    print(thresh_refined)  
    with open(os.path.join(pth,'processed','thresh_refined.txt'),'w') as f:
        f.write(str(thresh_refined))
    with open(os.path.join(pth,'processed','noise_floors.txt'),'w') as f:
        f.write(str(noisefloor_final))
    print(f"threshold {thresh_refined} with noise floor {noisefloor_final}")

    # MAIN BASE-CALLING ON ALL FOLDERS-TRIMMED IMAGES WITH FINALIZED THRESHOLD
    # # spot call each fov, using the thresholds we decided on, and the normalization we decided on
    for folder in folders:
        et=bardensr.spot_calling.estimate_density_singleshot(image_reader_trimmed(os.path.join(pth,'processed',folder,'aligned'),trim,R,C)/median_max[:,None,None,None],codeflat,noisefloor_final)
        spots=bardensr.spot_calling.find_peaks(et,thresh_refined,use_tqdm_notebook=False)
        spots.loc[:,'m1']=spots.loc[:,'m1']+trim
        spots.loc[:,'m2']=spots.loc[:,'m2']+trim
        spots.to_csv(os.path.join(pth,'processed',folder,'aligned','bardensrresult.csv'),index=False)
        print(f"in {folder} found {len(spots)} spots")


def import_bardensr_results(pth, 
                            fname='bardensrresult.csv',
                            prev_codebook_len=0,
                            codebook_name='codebook.joblib',
                            is_optseq=0,
                            codebook_opt_name='codebook.mat'):
    """
    Basecalling function for geneseq:
    1. Calls an accumulator function to aggregate bardensr results per tile
    2. Calls function to compute fdr for all tiles combined
    3. Writes aggregated geneseq basecall results
    """
    lroi_x=[] 
    lroi_y=[]
    gene_id=[]

    [lroi_x,lroi_y,gene_id]=accumulate_bardensr_results(pth, codebook_name, fname, 0)
    fdr=get_fdr(pth,codebook_name,gene_id)
    print(f'Finished importing bardensr results. FPR is {fdr}')

    if is_optseq:
        print('Has optseq')
        codebook_path=os.path.join(pth,'processed',codebook_opt_name)
        codebook=scipy.io.loadmat(codebook_path)['codebook']
        [lroi_x_opt,lroi_y_opt,gene_id_opt]=accumulate_bardensr_results(pth,codebook_opt_name,fname_opt,len(codebook))
        fdr_o=get_fdr(pth,codebook_opt_name,gene_id_opt)
        gene_id.append(gene_id_opt)
        lroi_x.append(lroi_x_opt)
        lroi_y.append(lroi_y_opt)

    # let's check if this works --exchanging x and y
    dump({"lroi_x":lroi_y, "lroi_y":lroi_x, "gene_id":gene_id}, os.path.join(pth,'processed','basecalls'+'.joblib')) 


def accumulate_bardensr_results(pth, codebook_name, fname, prev_codebook_len):
    """
    Basecalling function for geneseq:
    1. Accumulator function that combines bardensr results per tile
    2. Returns combined basecall results to the calling function
    """
    # codebook_path=os.path.join(pth,'processed',codebook_name)
    # codebook=scipy.io.loadmat(codebook_path)['codebook']
    [folders,_,_,_]=get_folders(pth)
    lroi_x=[] 
    lroi_y=[]
    gene_id=[]

    for i,folder in enumerate(folders):
        try:
            T=pd.read_csv(os.path.join(pth,'processed',folder,'aligned', fname),header=0)
            lroi_x.append(T.m2)
            lroi_y.append(T.m1)
            gene_id.append(T.j+prev_codebook_len)
        except:
            lroi_x.append([])
            lroi_y.append([])
            gene_id.append([])
            print(f'No geneseq rolonies found for tile {folder}\n')
    return(lroi_x,lroi_y,gene_id)



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
                    help='stage to use as template')
    
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
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    basecall_bardensr( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

