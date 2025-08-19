import itertools
import logging
import math
import os
import pprint
import subprocess
import tempfile
import threading

import datetime as dt

from collections import defaultdict

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

def flatten_nested_lists(lol):
    '''
    use pandas core flatten.
    '''
    flattened = list( pd.core.common.flatten(lol) )
    return flattened


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




class SimpleMatrix:
    '''
    Create simple expandable matrix with lax data types. 
    Accessed by  matrix[ m , n ] syntax.  m = row, n = column
    display is list of lists, expanded to handle largest index on each axis.  
    list of lists suitable for Pandas dataframe or Numpy ndarray creation. 
    
    m = SimpleMatrix()
    m[0,2] =  'Fred'
    m[1,1] = 'Ginger'
    m
    [[ '', '', 'Fred' ],
     [ '', 'Ginger', '']
                       ]
    
    Inspired by incredibly fussy Scipy and Numpy sparse matrix classes, which 
    must support large scaling. 
        
    '''
    def __init__(self, dtype=str):
        self.matrix = defaultdict(lambda: defaultdict(dtype))

    def __getitem__(self, key ):
        try:
            (r,c) = key
            r = int(r)
            c = int(c)
            val = self.matrix[r][c]
            logging.debug(f'get key is {key} r={r} c={c} val={val}')
            return val
        except:
            logging.warning(f'unable to parse key={key} e.g. [ 2,5]')
        
    def __setitem__(self, key, value ):
        try:
            (r,c) = key
            r = int(r)
            c = int(c)
            logging.debug(f'set key is {key} r={r} c={c}')
            self.matrix[r][c] = value
        except:
            logging.warning(f'unable to parse key={key} e.g. [ 2,5]')

    def __str__(self):
        return str(self.matrix )


    def __repr__(self):
        return str(self)
    
    
    def to_ndarray(self):
        '''
        Return ndarray of minimum dimensions to include all values in matrix. 
        
        '''
        rowvals = list( self.matrix.keys() )
        rowvals.sort()
        if len(rowvals) > 0:
            rmax = rowvals[-1]
        else:
            rmax = 0
        gmax = 0 
        for r in range(0,rmax):
            colvals = list( self.matrix[r].keys())
            colvals.sort()
            if len(colvals) > 0:
                cmax = colvals[-1]
            else:
                cmax = 0
            gmax = max(cmax, gmax )
        logging.debug(f'making ndarray with (row,col) = ({rmax}, {gmax})')            
        ndout = np.empty( (rmax +1 ,gmax + 1), dtype='U128'  )
        rkeys = list( self.matrix.keys())
        logging.debug(f'rkeys= {rkeys}')
        for rkey in rkeys:
            ckeys = list( self.matrix[rkey].keys() )
            logging.debug(f'rkey={rkey} ckeys={ckeys}')
            for ckey in ckeys:
                ndout[rkey,ckey] = self.matrix[rkey][ckey]
        return ndout
        
        
        
    def as_lol(self):
        '''
        Return list of lists, row-major
        '''
        pass



#    
#        Bardensr specific utils
#

def load_codebook_file(infile):
    df = pd.read_csv(infile, sep='\t', index_col=0)
    return df

def make_codebook_object(codebook, codebook_bases, n_cycles=7):
    '''
    Create binary codebook object for Bardensr from simple codebook dataframe.  
    
    
    '''
    # make codebook array to match explicit number of cycles.
    # it is possible that there are fewer cycles than codebook sequence lengths?
    num_channels = len(codebook_bases)
    genes = np.reshape( np.array( codebook['gene'], dtype='<U8'), (np.size(codebook,0),-1) )
    codebook_char = np.zeros((len(codebook), n_cycles), dtype=str)
    logging.debug(f'made empty array shape={codebook_char.shape} filling... ')
    codebook_seq = codebook['sequence']
    for i in range(len(codebook)):
        for j in range(n_cycles):       
            codebook_char[i,j] = codebook_seq.iloc[i][j]
    logging.debug(f'made sequence array {codebook_char}. making binary array.')
        
    codebook_bin=np.ones(np.shape(codebook_char), dtype=np.double)    
    bmax = math.pow(2, len(codebook_bases) - 1)
    rmap = {}
    for bchar in codebook_bases:
        rmap[bchar] = bmax
        bmax = bmax / 2
    logging.debug(f'made binary mappings for chars: {rmap}')
    
    codebook_bin=np.reshape( np.array([ rmap[x] for y in codebook_char for x in y]), np.shape(codebook_char))
    logging.debug(f'binary codebook = {codebook_bin}')
    #codebook_bin=np.reshape( np.array([float( x.replace('G','8').replace('T','4').replace('A','2').replace('C','1')) for y in codebook_char for x in y]), np.shape(codebook_char))
    codebook_bin=np.matmul(np.uint8(codebook_bin), 2**np.transpose(np.array((np.arange(4 * n_cycles -4, -1, -4)))))
    codebook_bin=np.array([bin(i)[2:].zfill(n_cycles * num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook_char, 0), n_cycles * num_channels))
    logging.debug(f'reshaped codebook_bin={codebook_bin}')

    co=[[genes[i],codebook_bin[j,:]] for i in range(np.size(genes, 0))]
    co=[codebook,co]  
    codebook_bin=np.reshape(codebook_bin,(np.size(codebook_bin, 0), -1, num_channels))
    logging.debug(f'final codebook_bin={codebook_bin}')
    
    cb = np.transpose(codebook_bin, axes=(1,2,0))
    R,C,J=cb.shape
    pos_unused_codes = np.where(np.char.startswith( genes,'unused'))
    logging.debug(f' R={R} C={C} J={j} pos_unused_codes={pos_unused_codes}')
    codeflat=np.reshape(cb,( -1, J))
    return (codeflat, R, C, J, genes, pos_unused_codes)
