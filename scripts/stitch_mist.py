#!/usr/bin/env python
#
#
#   https://forum.image.sc/t/microscope-image-stitching-package-in-python/55721
#   Ashlar   https://github.com/labsyspharm/ashlar
#   M2 Stitch
#

# Script to use MIST (via ImageJ/FIJI with MIST plugins to stitch images.
#
#    1.     --filenamePattern <string containing filename pattern>
#    2.     --filenamePatternType <ROWCOL/SEQUENTIAL>
#    3a.    --gridOrigin <UL/UR/LL/LR>  -- Required only for ROWCOL or SEQUENTIAL
#    3b.    --numberingPattern <VERTICALCOMBING/VERTICALCONTINUOUS/HORIZONTALCOMBING/HORIZONTALCONTINUOUS> 
#                -- Required only for SEQUENTIAL
#    4.     --gridHeight <#>
#    5.     --gridWidth <#>
#    6.     --imageDir <PathToImageDir>
#    7a.     --startCol <#> -- Required only for ROWCOL
#    7b.     --startRow <#> -- Required only for ROWCOL
#    7c.     --startTile <R> -- Required only for SEQUENTIAL
#    8.     --programType <AUTO/JAVA/FFTW> -- Highly recommend using FFTW
#    9.     --fftwLibraryFilename libfftw3f.dll -- Required for FFTW program type
#    9a.     --fftwLibraryName libfftw3f -- Required for FFTW program type
#    9b.     --fftwLibraryPath <path/to/library> -- Required for FFTW program type

#    Example execution:
#    java.exe -jar MIST_-2.1-jar-with-dependencies.jar --filenamePattern img_r{rrr}_c{ccc}.tif --filenamePatternType ROWCOL --gridHeight 5 --gridWidth 5 --gridOrigin UR --imageDir C:\Users\user\Downloads\Small_Fluorescent_Test_Dataset\image-tiles --startCol 1 --startRow 1 --programType FFTW --fftwLibraryFilename libfftw3f.dll --fftwLibraryName libfftw3f --fftwLibraryPath C:\Users\user\apps\Fiji.app\lib\fftw
#
#
import argparse
import logging
import os
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *


import imagej


def run_cmd_shell(cmd):
    """
    maybe subprocess.run(" ".join(cmd), shell=True)
    cmd should be standard list of tokens...  ['cmd','arg1','arg2'] with cmd on shell PATH.
    
    """
    cmdstr = " ".join(cmd)
    logging.debug(f"running command: {cmdstr} ")
    start = dt.datetime.now()
    cp = subprocess.run(" ".join(cmd), 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT)

    end = dt.datetime.now()
    elapsed =  end - start
    logging.debug(f"ran cmd='{cmdstr}' return={cp.returncode} {elapsed.seconds} seconds.")
    
    if cp.stderr is not None:
        logging.warn(f"got stderr: {cp.stderr}")
        pass
    if cp.stdout is not None:
        #logging.debug(f"got stdout: {cp.stdout}")
        pass
    if str(cp.returncode) == '0':
        #logging.debug(f'successfully ran {cmdstr}')
        logging.debug(f'got rc={cp.returncode} command= {cmdstr}')
    else:
        logging.warn(f'got rc={cp.returncode} command= {cmdstr}')
        raise NonZeroReturnException(f'For cmd {cmdstr}')
    return cp


def stitch_fijimist( infiles, outdir, stage=None, cp=None):
    '''
    image_type = [ geneseq | bcseq | hyb ]
    
    stitch images from the Small_Flourescent_Test_Dataset that comes with MIST
    
    '''
    if cp is None:
        cp = get_default_config()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
        
    output_dtype = cp.get('stitch','output_dtype')

    fiji_path = os.path.abspath( os.path.expanduser( cp.get('mist','fiji_path')))
    logging.debug(f'fiji_path = {fiji_path}')
    # config with imageJ


    # unlikely to ever change    
    static_macro_args = [ 'debuglevel=NONE',
                          'loglevel=MANDATORY', 
                          'assemblefrommetadata=false',
                          'assemblenooverlap=false' , 
                          'compressionmode=UNCOMPRESSED',
                          'displaystitching=false', 
                          'globalpositionsfile=[]',
                          'headless=true', 
                          'isusedoubleprecision=false', 
                          'isusebioformats=false', 
                          'issuppressmodelwarningdialog=false', 
                          'istimeslicesenabled=false', 
                          'isenablecudaexceptions=false', 
                          'outputfullimage=true', 
                          'outputmeta=true',
                          'outputimgpyramid=false', 
                           'stagerepeatability=0',                          
                           'overlapuncertainty=NaN',
                          'timeslices=0',
                          'unit=MICROMETER', 
                          'unitx=1.0', 
                          'unity=1.0',
                        ]

    # defaults for CSHL/current protocol
    # image processing/algorithm parameters
    default_macro_args = [ 'filenamepatterntype=ROWCOL',
                           'gridorigin=UR',
                           'starttilerow=0', 
                           'starttilecol=0',
                           'numberingpattern=HORIZONTALCOMBING',
                           'programtype=JAVA',
                           'startrow=0',
                           'startcol=0',
                           'blendingmode=AVERAGE', 
                           'blendingalpha=NaN',
                          
                           'translationrefinementmethod=SINGLE_HILL_CLIMB', 
                           'numtranslationrefinementstartpoints=16',
                          ]
    # experiment/run-specific settings.
    # directories/filenames/patterns
    maxproj_regex=cp.get('maxproj','maxproj_regex')
    (dirpath, base, ext) = split_path(infiles[0])
    filename = f'{base}.{ext}'
    input_dir = dirpath
    m = re.search(maxproj_regex, filename)
    if m is not None:
        pos = m.group(1)
        x = m.group(2)
        y = m.group(3)
        x = int(x)
        y = int(y)
        logging.debug(f'pos={pos} x={x} y={y} type(pos)={type(pos)}')
    else:
        logging.error(f'unable to parse input filename {filename} via regex.')
        sys.exit(1)
    filenamePattern='Max_Pos'+pos+'_\{ccc\}_\{rrr\}.tif'    
    logging.debug(f'filenamePattern = {filenamePattern}')
    outfileprefix = f'Max_Pos{pos}'
    
    # image properties
    vertical_overlap=cp.get('tile','vertical_overlap') 
    horizontal_overlap=cp.get('tile','horizontal_overlap') 
    n_jobs = cp.get('mist', 'n_jobs')
    
    var_macro_args = [  f'imagedir={input_dir}',
                        f'filenamepattern={filenamePattern}',
                        f'outfileprefix={outfileprefix}',
                        f'outputpath={outdir}', 
                        f'horizontaloverlap={horizontal_overlap}', 
                        f'verticaloverlap={vertical_overlap}',         
                        f'extentwidth=4',
                        f'extentheight=5', 
                        f'gridwidth=4', 
                        f'gridheight=5',
                        f'numcputhreads={n_jobs}',
                        'numfftpeaks=0',
                        'loadfftwplan=false',       
                        'savefftwplan=false', 
                        ]
    
                        #'fftwplantype=MEASURE', 
                        #'fftwlibraryname=libfftw3',
                        #'fftwlibraryfilename=libfftw3.dll 
                        #'planpath=C:\\\\Fiji.app\\\\lib\\\\fftw\\\\fftPlans \
                        #'fftwlibrarypath=C:\\\\Fiji.app\\\\lib\\\\fftw

    all_args = [ x for xs in [var_macro_args, default_macro_args,static_macro_args ] for x in xs  ]
    macro_args = ' '.join(all_args)
                        
    macro = f'''run("MIST", "{macro_args}");'''
    logging.debug(f'macro={macro}')

    logging.info(f'initializing fiji at {fiji_path}...')
    ij=imagej.init(fiji_path)
    logging.debug(f'fiji version = {ij.getVersion()}')



def stitch_javamist(infiles, outdir, stage=None, cp=None):
    '''
    
        infiles:    list of images for a single position to be stitched a single image. 
        e.g. MAX_Pos1_000_000.denoised.tif MAX_Pos1_000_001.denoised.tif ...
        output:    MAX_Pos1_stitched.tif 

    --assembleFromMetadata <arg>                  Input param
                                                  assembleFromMetadata
    --assembleNoOverlap <arg>                     Input param
                                                  assembleNoOverlap
    --blendingAlpha <arg>                         Output param
                                                  blendingAlpha
    --blendingMode <arg>                          Output param
                                                  blendingMode
    --compressionMode <arg>                       Output param
                                                  compressionMode
    --cudaDevice <arg>                            Advanced param
                                                  cudaDevice
    --debugLevel <arg>                            Logging param debugLevel
    --displayStitching <arg>                      Output param
                                                  displayStitching
    --extentHeight <arg>                          Input param extentHeight
    --extentWidth <arg>                           Input param extentWidth
    --fftwLibraryFilename <arg>                   Advanced param
                                                  fftwLibraryFilename
    --fftwLibraryName <arg>                       Advanced param
                                                  fftwLibraryName
    --fftwLibraryPath <arg>                       Advanced param
                                                  fftwLibraryPath
    --fftwPlanType <arg>                          Advanced param
                                                  fftwPlanType
    --filenamePattern <arg>                       Input param
                                                  filenamePattern
    --filenamePatternType <arg>                   Input param
                                                  filenamePatternType
    --globalPositionsFile <arg>                   Input param
                                                  globalPositionsFile
    --gridHeight <arg>                            Input param gridHeight
    --gridOrigin <arg>                            Input param gridOrigin
    --gridWidth <arg>                             Input param gridWidth
     -h,--help                                        Display this help
                                                  message and exit.
    --headless <arg>                              Advanced param headless
    --horizontalOverlap <arg>                     Advanced param
                                                  horizontalOverlap
    --imageDir <arg>                              Input param imageDir
    --isEnableCudaExceptions <arg>                Advanced param
                                                  isEnableCudaExceptions
    --isSuppressModelWarningDialog <arg>          Advanced param
                                                  isSuppressModelWarningDi
                                                  alog
    --isTimeSlicesEnabled <arg>                   Input param
                                                  isTimeSlicesEnabled
    --isUseBioFormats <arg>                       Advanced param
                                                  isUseBioFormats
    --isUseDoublePrecision <arg>                  Advanced param
                                                  isUseDoublePrecision
    --loadFFTWPlan <arg>                          Advanced param
                                                  loadFFTWPlan
    --logLevel <arg>                              Logging param logLevel
    --numberingPattern <arg>                      Input param
                                                  numberingPattern
    --numCPUThreads <arg>                         Advanced param
                                                  numCPUThreads
    --numFFTPeaks <arg>                           Advanced param
                                                  numFFTPeaks
    --numTranslationRefinementStartPoints <arg>   Advanced param
                                                  numTranslationRefinement
                                                  StartPoints
    --outFilePrefix <arg>                         Output param
                                                  outFilePrefix
    --outputFullImage <arg>                       Output param
                                                  outputFullImage
    --outputImgPyramid <arg>                      Output param
                                                  outputImgPyramid
    --outputMeta <arg>                            Output param outputMeta
    --outputPath <arg>                            Output param outputPath
    --overlapUncertainty <arg>                    Advanced param
                                                  overlapUncertainty
    --planPath <arg>                              Advanced param planPath
    --programType <arg>                           Advanced param
                                                  programType
    --saveFFTWPlan <arg>                          Advanced param
                                                  saveFFTWPlan
    --stageRepeatability <arg>                    Advanced param
                                                  stageRepeatability
    --startCol <arg>                              Input param startCol
    --startRow <arg>                              Input param startRow
    --startTile <arg>                             Input param startTile
    --startTileCol <arg>                          Input param startTileCol
    --startTileRow <arg>                          Input param startTileRow
    --timeSlices <arg>                            Input param timeSlices
    --translationRefinementMethod <arg>           Advanced param
                                                  translationRefinementMet
                                                  hod
    --unit <arg>                                  Output param unit
    --unitX <arg>                                 Output param unitX
    --unitY <arg>                                 Output param unitY
    --verticalOverlap <arg>                       Advanced param
                                                  verticalOverlap

INITAL...
    --filenamePattern MAX_Pos1_{ccc}_{rrr}.tif
    --filenamePatternType ROWCOL 
    --gridHeight 5 
    --gridWidth 4 
    --gridOrigin UR 
    --imageDir <indir> 
    --startCol 0 
    --startRow 0 
    --programType FFTW 
    --fftwLibraryFilename libfftw3f.dll 
    --fftwLibraryName libfftw3f 
    --fftwLibraryPath /Applications/Fiji.app/lib/fftw
     --outputPath <arg> 
   --horizontalOverlap <arg>   # in PERCENT
   --verticalOverlap <arg>     # in PERCENT
   
   --numberingPattern "HORIZONTALCOMBING"
   
       <VERTICALCOMBING/ 
       VERTICALCONTINUOUS/
       HORIZONTALCOMBING/
       HORIZONTALCONTINUOUS> -- Required only for SEQUENTIAL
   
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'stitch'
 
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    output_dtype = cp.get('stitch','output_dtype')
    logging.debug(f'handling stage={stage} to outdir={outdir} output_dtype={output_dtype}')    
    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]}')
    
    current_env = os.environ['CONDA_DEFAULT_ENV']
    conda_prefix = os.environ['CONDA_PREFIX']
    java_home = os.environ['JAVA_HOME']
    
    java_exe = os.path.join(java_home, 'bin', 'java')
    mist_jar = f'{conda_prefix}/libexec/MIST_-2.1-jar-with-dependencies.jar'
    
    fftw_path = f'{conda_prefix}/lib/'
    fftw_name = 'libfftw3f'
    fftw_filename = 'libfftw3.dylib'

    #fftw_path = '/Applications/Fiji.app/lib/fftw'
    #fftw_name = 'libfftw3f'
    #fftw_filename = 'libfftw3f.dll '
    
    
    cmdlist = [java_exe, '-jar',  mist_jar,
           '--headless', 'true',
           '--programType' , 'JAVA',
           '--fftwLibraryPath',  fftw_path ,
           '--fftwLibraryFilename', fftw_filename ,
           '--fftwLibraryName',  fftw_name ,
                      

           ]
    ''' 
           '--fftwLibraryPath',  fftw_path ,
           '--fftwLibraryFilename', fftw_filename ,
           '--fftwLibraryName',  fftw_name ,
    '''
    cmdlist.append( '--version')
    
    cmdstr = ' '.join(cmdlist)
    logging.info(f'running {cmdstr}')
    logging.debug(f'got command: {cmdlist}')
    run_command_shell(cmdlist)
    logging.debug(f'completed command: {cmdlist}')
    logging.info(f'completed {cmdstr} ')                           

                            
    ''' 
                               '--config' , runconfig , ]
            cmd.append('--stage')
            cmd.append(f'{stage}')
            cmd.append( '--outdir ' )
            cmd.append( f'{sub_outdir}')                
            for rname in flist:
                infile = f'{indir}/{rname}'
                outfile = f'{outdir}/{rname}'
    '''


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
    
    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    default=None, 
                    type=str, 
                    help='outdir. output base dir if not given.')
    
    parser.add_argument('infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 
       
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
      
    outdir = os.path.abspath('./')
    if args.outdir is not None:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    stitch_fijimist( infiles=args.infiles, 
                     outdir=outdir, 
                     cp=cp )
    
    logging.info(f'done processing output to {outdir}')