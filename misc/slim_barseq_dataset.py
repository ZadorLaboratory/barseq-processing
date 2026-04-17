#!/usr/bin/env python
#
# For given position, rowstart, rowend, colstart, colend
# copies relevant files from input root to parallel tree under outdir root. 
# Assumes MAX_<position>_<column>_<row> 
# Assumer row/col numbers have width=3 with leading 0
# Assumes axis starts in UPPER RIGHT. 
# Assumes a single layer of dirs below input/output roots. 
# Starts and Ends are inclusive. 
# 
import argparse
import logging
import os
import pprint
import shutil

def make_index_range(start, end, shift=False):
    '''
        E.g. 
        start = 002 and end = 005. ->
           [ 002, 003, 004, 005  ]
    f"{number:05d}"
    0: character to pad with
    5: width of final
    d: input integer

    shift=True -> move start to 000 and others accordingly.

    '''
    start_s = str(start)
    end_s = str(end)
    start_i = int(start_s)
    end_i = int(end_s)
    outlist = []
    if shift:
        logging.debug(f'shifting by {start_i} ...')
        for i in range( 0, end_i - start_i + 1 ):
            outlist.append(f'{i:03d}')
    else:
        for i in range(start_i, end_i +1):
            outlist.append(f'{i:03d}')
    logging.debug(f'outlist={outlist}')
    return outlist

def make_file_list(position, col_range, row_range):
    '''
    
    '''
    flist = []
    for col_s in col_range:
        for row_s in row_range:
            fname = f'MAX_{position}_{col_s}_{row_s}.tif'
            flist.append(fname)
    return flist

def make_copy_map(indir, outdir, col_start, col_end, row_start, row_end , position):
    logging.info(f'indir={indir} outdir={outdir}  position={position} ')
    logging.info(f'row_start={row_start}  row_end={row_end} col_start={col_start} col_end={col_end}')

    all_infiles = []
    all_outfiles = []

    indir = os.path.abspath(indir)
    outdir = os.path.abspath(outdir)
    subdirs = os.listdir(indir)
    logging.debug(f'subdirs={subdirs}')
    for subdir in subdirs:
        insub = os.path.join(indir, subdir)
        logging.debug(f'handling subdir={insub} ...')
        col_range = make_index_range(col_start, col_end, shift=False)
        row_range = make_index_range(row_start, row_end, shift=False)
        infiles = make_file_list(position, col_range, row_range)
        infiles = [ os.path.join(indir, subdir, infile) for infile in infiles  ]
        logging.debug(f'infiles = {infiles }')
        for infile in infiles:
            all_infiles.append(infile) 
        
        col_range = make_index_range(col_start, col_end, shift=True)
        row_range = make_index_range(row_start, row_end, shift=True)
        outfiles = make_file_list(position, col_range, row_range)
        outfiles = [ os.path.join(outdir, subdir, outfile) for outfile in outfiles ]
        logging.debug(f'outfiles = {outfiles}')
        for outfile in outfiles:
            all_outfiles.append(outfile)
    logging.info(f'made list of {len(all_infiles)} infiles and {len(all_outfiles)} outfiles')
    copy_map = list( zip(all_infiles, all_outfiles))
    logging.debug(f'copy map len={len(copy_map )}, e.g.: {copy_map[0]}')
    return copy_map

def copy_transform_position_tiles(indir, outdir, 
                                  col_start, col_end, 
                                  row_start, row_end, 
                                  position, dry_run=False):
    copy_map = make_copy_map(indir, outdir, col_start, col_end, row_start, row_end , position)
    copy_map_s = pprint.pformat(copy_map)
    logging.info(f'copy_map = {copy_map}')
    if not dry_run:
        for (infile, outfile) in copy_map:
            if os.path.exists(infile):
                dirpath, filename = os.path.split(outfile)
                os.makedirs(dirpath, exist_ok=True)
                shutil.copyfile(infile, outfile)
                logging.debug(f'{infile} -> {outfile}')
            else:
                logging.warning(f'{infile} does not exist!')
        logging.info(f'done slimming from {indir} to {outdir}')
    else:
        logging.info(f'dry run requested. no copying. ')
        print( copy_map_s)

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

    parser.add_argument('-n', '--dry_run', 
                        action="store_true",
                        default=False,  
                        dest='dry_run', 
                        help='do not perform copy, just print map.')

    parser.add_argument('-I','--indir', 
                    metavar='indir',
                    required=True, 
                    type=str, 
                    help='input directory base. ')

    parser.add_argument('-O','--outdir', 
                    metavar='outdir',
                    required=True, 
                    type=str, 
                    help='output directory base')

    parser.add_argument('-p','--position', 
                    metavar='position',
                    required=True, 
                    type=str, 
                    help='position')

    parser.add_argument('-S','--col_start', 
                    metavar='col_start',
                    required=True, 
                    type=str, 
                    help='col_start index')

    parser.add_argument('-E','--col_end', 
                    metavar='col_end',
                    required=True, 
                    type=str, 
                    help='col_end index')

    parser.add_argument('-s','--row_start', 
                    metavar='row_start',
                    required=True, 
                    type=str, 
                    help='row_start index')

    parser.add_argument('-e','--row_end', 
                    metavar='row_end',
                    required=True, 
                    type=str, 
                    help='row_end index')           

    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   

       
    copy_transform_position_tiles(args.indir, 
                                  args.outdir,
                                  args.col_start,
                                  args.col_end,
                                  args.row_start,
                                  args.row_end,
                                  args.position,
                                  args.dry_run)
    