import logging
import os
import pprint
import subprocess
import tempfile
import threading

import datetime as dt

import pandas as pd
import numpy as np


def format_config(cp):
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    s = pprint.pformat(cdict, indent=4)
    return s

def get_boolean(s):
    TRUES = [ 1 , '1', 'true']
    FALSES = [ 0 , '0', 'false']
    a = None
    if s in TRUES:
        a = True
    elif s in FALSES:
        a = False
    return a

def pprint_dict(d):
    s = pprint.pformat(d, indent=4)
    return s

def split_path(filepath):
    '''
    dir, base, ext = split_path(filepath)
    
    '''
    filepath = os.path.abspath(filepath)
    dirpath = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    ext = ext[1:] # remove dot
    return (dirpath, base, ext)


def load_df(filepath, as_array=False, dtype='float64'):
    """
    Convenience method to load DF consistently across modules.
    
    @arg as_array  return the contents as a standard numpy ndarray 
    @arg dtype     specify dtype
    """
    logging.debug(f'loading {filepath}')
    filepath = os.path.abspath( os.path.expanduser(filepath) )
    df = pd.read_csv(filepath, sep='\t', index_col=0, keep_default_na=False, dtype="string[pyarrow]", comment="#")
    logging.debug(f'initial load done. converting types...')
    df = df.convert_dtypes(convert_integer=False)
    for col in df.columns:
        #logging.debug(f'trying column {col}')
        try:
            df[col] = df[col].astype('uint32')
        except ValueError:
            pass
            #logging.debug(f'column {col} not int')
    logging.debug(f'dtypes = \n{df.dtypes}')
    if as_array:
        return df.to_numpy(dtype=dtype)
    return df


def merge_dfs(dflist):
    newdf = None
    for df in dflist:
        if newdf is None:
            newdf = df
        else:
            newdf = pd.concat([df, newdf], ignore_index=True, copy=False)
    logging.debug(f'merged {len(dflist)} dataframes newdf len={len(newdf)}')
    return newdf

def write_df(newdf, filepath,  mode=0o644):
    """
    Writes df in standard format.  
    """
    logging.debug(f'inbound df:\n{newdf}')
    try:
        df = newdf.reset_index(drop=True)
        rootpath = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        df.to_csv(filepath, sep='\t')
        os.chmod(filepath, mode)
        logging.debug(f"wrote df to {filepath}")

    except Exception as ex:
        logging.error(traceback.format_exc(None))
        raise ex


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

# Multiprocessing  using explicity command running. 
#            jstack = JobStack()
#            cmd = ['program','-a','arg1','-b','arg2','arg3']
#            jstack.addjob(cmd)
#            jset = JobSet(max_processes = threads, jobstack = jstack)
#            jset.runjobs()
#
#            will block until all jobs in jobstack are done, using <max_processes> jobrunners that
#            pull from the stack.
#

class JobSet(object):
    def __init__(self, max_processes, jobstack):
        self.max_processes = max_processes
        self.jobstack = jobstack
        self.threadlist = []
        
        for x in range(0, self.max_processes):
            jr = JobRunner(self.jobstack, label=f'{x}')
            self.threadlist.append(jr)
        logging.debug(f'made {len(self.threadlist)} jobrunners. ')


    def runjobs(self):
        logging.debug(f'starting {len(self.threadlist)} threads...')
        for th in self.threadlist:
            th.start()
            
        logging.debug(f'joining threads...')    
        for th in self.threadlist:
            th.join()
            
        logging.debug(f'all threads joined. returning...')


class JobStack(object):
    def __init__(self):
        self.stack = []

    def addjob(self, cmdlist):
        '''
        List is tokens appropriate for 
        e.g. cmd list :  [ '/usr/bin/x','-x','xarg','-y','yarg']
        '''
        self.stack.append(cmdlist)
        
    def setlist(self, job_cmdlist):
        '''
        Allows explictly setting a list (of cmd lists) created in bulk.  
        '''
        self.stack = job_cmdlist
    
    def pop(self):
        return self.stack.pop()


class JobRunner(threading.Thread):

    def __init__(self, jobstack, label=None):
        super(JobRunner, self).__init__()
        self.jobstack = jobstack
        self.label = label
        
    def run(self):
        while True:
            try:
                cmdlist = self.jobstack.pop()
                cmdstr = ' '.join(cmdlist)
                logging.info(f'[{self.label}] running {cmdstr}')
                logging.debug(f'[{self.label}] got command: {cmdlist}')
                run_command_shell(cmdlist)
                logging.debug(f'[{self.label}] completed command: {cmdlist}')
                logging.info(f'[{self.label}] completed {cmdstr} ')
            
            except NonZeroReturnException:
                logging.warning(f'[{self.label}] NonZeroReturn Exception job: {cmdlist}') 
            
            except IndexError:
                logging.info(f'[{self.label}] Command list empty. Ending...')
                break
                
def uint16m(x):
    y=np.uint16(np.clip(np.round(x),0,65535))
    return y

def write_config(config, filename, timestamp=True, datestring=None):
    '''
    writes config file to relevant name,
    if timestamp=True, puts date/time code dot-separated before extension. e.g.
    filename = /path/to/some.file.string.txt  ->  /path/to/some.file.string.202303081433.txt
    date is YEAR/MO/DY/HR/MI
    if datestring is not None, uses that timestamp
    
    '''
    filepath = os.path.abspath(filename)    
    dirname = os.path.dirname(filepath)
    basefilename = os.path.basename(filepath)
    (base, ext) = os.path.splitext(basefilename) 
    
    if timestamp:
        if datestring is None:
            datestr = dt.datetime.now().strftime("%Y%m%d%H%M")
        else:
            datestr = datestring
        filename = f'{dirname}/{base}.{datestr}{ext}'

    os.makedirs(dirname, exist_ok=True)
        
    with open(filename, 'w') as configfile:
        config.write(configfile)
    logging.debug(f'wrote current config to {filename}')
    
    return os.path.abspath(filename)


    
