#!/usr/bin/env python
# 
# Wrapper for CONDA-based Java version of MIST using compiled executable JAR. 
# Per Github docs. 
# 
#
#
#
import argparse
import logging
import os 
import subprocess
import sys


import datetime as dt

class NonZeroReturnException(Exception):
    """
    Thrown when a command has non-zero return code. 
    """
    
def run_command_shell(cmd):
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



if __name__ == '__main__':
    '''
      $CONDA_PREFIX/lib/jvm/bin/java 
    -jar $CONDA_PREFIX/libexec/MIST_-2.1-jar-with-dependencies.jar 
    --programType FFTW 
    --fftwLibraryFilename libfftw3.so 
    --fftwLibraryName libfftw3 
    --fftwLibraryPath 
    $CONDA_PREFIX/lib --help
    
    '''
    
    try:
        CONDA_PREFIX = os.environ['CONDA_PREFIX']
    
    except KeyError:
        logging.error('no CONDA_PREFIX.')
    
    
    
    


    
    #logging.warning(f'args={args}')
    
    cmd = [ f'{CONDA_PREFIX}/lib/jvm/bin/java' ,
           '-jar', f'{CONDA_PREFIX}/libexec/MIST_-2.1-jar-with-dependencies.jar',
           '--programType' , 'FFTW',
           '--fftwLibraryFilename' , 'libfftw3.so', 
           '--fftwLibraryName', 'libfftw3', 
           '--fftwLibraryPath', f'{CONDA_PREFIX}/lib',
           args
           ]
    
    print(f'cmd={cmd}')
                       
            
    
    
  