
import logging
import os
import pprint

import pandas as pd

def format_config(cp):
    cdict = {section: dict(cp[section]) for section in cp.sections()}
    s = pprint.pformat(cdict, indent=4)
    return s


def split_path(filepath):
    '''
    dir, base, ext = split_path(filepath)
    
    '''
    filepath = os.path.abspath(filepath)
    dirpath = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    return (dirpath, base, ext)


def merge_dfs(dflist):
    newdf = None
    for df in dflist:
        if newdf is None:
            newdf = df
        else:
            newdf = pd.concat([df, newdf], ignore_index=True, copy=False)
    logging.debug(f'merged {len(dflist)} dataframes newdf len={len(newdf)}')
    return newdf