#!/usr/bin/env python
# take matlab codebook and convert to Pandas/TSV 
#
import argparse
import joblib
import logging
import os
import pprint
import sys
import traceback

from configparser import ConfigParser

import numpy as np
import pandas as pd
import scipy

from scipy.io import loadmat

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.utils import *

def format_config(cp):
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    s = pprint.pformat(cdict, indent=4)
    return s

def dump_bardensr_codebook(infile, outfile, n_cycles=7):
      
    infile = os.path.abspath(os.path.expanduser(infile))
    outfile = os.path.abspath(os.path.expanduser(outfile))
    (odirpath, obase, olabel, oext) = split_path(os.path.abspath(outfile))

    logging.debug(f'infile={infile} outfile={outfile}')    
    num_channels = 4

    mcodebook = scipy.io.loadmat(infile)['codebook']
    genes=np.array([str(x[0][0]) for x in mcodebook], dtype=str)
    logging.debug(f'mcodebook={mcodebook}')

    mcblist = list(mcodebook)
    lol = []
    for e in mcblist:
        elist = list(e)
        gene = str(elist[0][0])
        seq = str(elist[1][0])
        lol.append([gene,seq])
    df = pd.DataFrame(lol, columns=['gene','sequence'])

    codebook1=np.zeros((np.size(mcodebook,0), n_cycles ), dtype=str)
    for i in range(np.size(mcodebook,0)):
        for j in range(n_cycles):
            codebook1[i,j]=mcodebook[i][1][0][j]

    codebook_bin=np.ones(np.shape(codebook1),dtype=np.double)
    codebook_bin=np.reshape(np.array([float(x.replace('G','8').replace('T','4').replace('A','2').replace('C','1')) for y in codebook1 for x in y]), np.shape(codebook1))
    codebook_bin=np.matmul(np.uint8(codebook_bin),2**np.transpose(np.array((np.arange(4*n_cycles-4,-1,-4)))))
    codebook_bin=np.array([bin(i)[2:].zfill( n_cycles * num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook1,0),n_cycles*num_channels))
    co=[[genes[i],codebook_bin[j,:]] for i in range(np.size(genes,0))]
    co=[mcodebook,co]
    codebook_bin1=np.reshape(codebook_bin, ( np.size(codebook_bin, 0), -1, num_channels) )
        
    # os.path.join(pth,'processed','codebook.joblib')
    of = os.path.join(odirpath, f'{obase}.codebook.joblib' )
    joblib.dump(co, of )


    # os.path.join(pth,'processed','codebookforbardensr.joblib')
    of = os.path.join(odirpath, f'{obase}.codebookforbardensr.joblib' )
    joblib.dump(codebook_bin1, of )

    df.to_csv(outfile, sep='\t')
    logging.debug(f'got codebook len={len(df)} e.g. {df.iloc[0]}')


def load_bardensr_codebook(codebook_tsv, n_cycles=7, codebook_bases=['G','T','A','C']):
    '''
    Load TSV/Pandas DataFrame codebook and create appropriate binary input for
    bardensr. 
                gene	sequence
        0	Calb1	AGTTCGG
        1	Rasgrf2	CTTCGTT
        2	Tafa1	CGAGTGG
        3	Enpp2	ACCGTAG
        4	Col19a1	TAACGCG
        5	Rorb	GCTAGAG
        6	Slc24a3	GTCGAAC
        7	Galntl6	TTGTTAG


    '''
    cbdf = load_codebook_file(codebook_tsv)
    num_channels = len(codebook_bases)
    genes = np.reshape(  np.array( cbdf['gene'],  dtype='<U8'), (np.size(cbdf,0), -1) )
    pos_unused_codes=np.where(np.char.startswith(genes,'unused'))
    err_codes=genes[pos_unused_codes]

    codebook_char = np.zeros((len(cbdf), n_cycles), dtype=str)
    codebook_seq = cbdf['sequence']
    for i in range(len(cbdf)):
        for j in range(n_cycles):       
            codebook_char[i,j] = codebook_seq.iloc[i][j]
    codebook_bin=np.ones(np.shape(codebook_char), dtype=np.double)
    codebook_bin=np.reshape( np.array([ rmap[x] for y in codebook_char for x in y]), np.shape(codebook_char))
    codebook_bin=np.matmul(np.uint8(codebook_bin), 2**np.transpose(np.array((np.arange(4 * n_cycles -4, -1, -4)))))    
    codebook_bin=np.array([bin(i)[2:].zfill(n_cycles * num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook_char, 0), n_cycles * num_channels))

    codebook_bardensr=np.reshape(codebook_bin,(np.size(codebook_bin,0),-1,num_channels))
    R,C,J=codebook_bardensr.shape
    codeflat=np.reshape(codebook_bardensr,(-1,J))




def load_codebook_file(infile):
    df = pd.read_csv(infile, sep='\t', index_col=0)
    return df


def make_codebook_object(codebook, codebook_bases, n_cycles=7):
    '''
    Create binary codebook object for Bardensr from simple codebook dataframe.  
    codebook_bases = G,T,A,C
    code
    
    @arg codebook           Pandas dataframe. 
    @arg codebook_bases     list -> ['G','T','A','C']  
    @arg n_cycles           int -> number of cycles

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
    logging.debug(f'made sequence array len= {len(codebook_char)}. making binary array.')
        
    codebook_bin=np.ones(np.shape(codebook_char), dtype=np.double)    
    bmax = math.pow(2, len(codebook_bases) - 1)
    rmap = {}
    for bchar in codebook_bases:
        rmap[bchar] = bmax
        bmax = bmax / 2
    logging.debug(f'made binary mappings for chars: {rmap}')
    
    codebook_bin=np.reshape( np.array([ rmap[x] for y in codebook_char for x in y]), np.shape(codebook_char))
    logging.debug(f'binary codebook shape= = {codebook_bin.shape}')
    #codebook_bin=np.reshape( np.array([float( x.replace('G','8').replace('T','4').replace('A','2').replace('C','1')) for y in codebook_char for x in y]), np.shape(codebook_char))
    codebook_bin=np.matmul(np.uint8(codebook_bin), 2**np.transpose(np.array((np.arange(4 * n_cycles -4, -1, -4)))))
    codebook_bin=np.array([bin(i)[2:].zfill(n_cycles * num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook_char, 0), n_cycles * num_channels))
    logging.debug(f'reshaped codebook_bin shape={codebook_bin.shape}')

    co=[[genes[i],codebook_bin[j,:]] for i in range(np.size(genes, 0))]
    co=[codebook,co]  
    codebook_bin=np.reshape(codebook_bin,(np.size(codebook_bin, 0), -1, num_channels))
    logging.debug(f'final codebook_bin shape={codebook_bin.shape}')
    
    cb = np.transpose(codebook_bin, axes=(1,2,0))
    R,C,J=cb.shape
    pos_unused_codes = np.where(np.char.startswith( genes,'unused'))
    logging.debug(f' R={R} C={C} J={j} pos_unused_codes={pos_unused_codes}')
    codeflat=np.reshape(cb,( -1, J))
    return (codeflat, R, C, J, genes, pos_unused_codes)



   
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

    parser.add_argument('-t', '--test', 
                        action="store_true", 
                        default=False,
                        dest='test', 
                        help='test TSV loading and conversion.')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')    

    parser.add_argument('-n','--n_cycles', 
                        metavar='n_cycles',
                        required=False,
                        default=7,
                        type=int, 
                        help='Number of cycles to make codebook.')    



    parser.add_argument('-o','--outfile', 
                    metavar='outfile',
                    required=True,
                    type=str,
                    )     

    parser.add_argument('infile',
                        metavar='infile',
                        type=str,
                        help='Single BARseq bardenser .mat file.')
       

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)    
    logging.debug(f'Running with config. {args.config}: {cdict}')
    logging.debug(f'infile={args.infile} outfile={args.outfile}')
       
    dump_bardensr_codebook(args.infile, args.outfile, args.n_cycles)

    load_bardensr_codebook(args.outfile)
    