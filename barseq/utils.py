
import os
import pprint

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