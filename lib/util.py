import os
from collections import Iterable

def flatten(l):
    # Flatten an irregular list of lists
    # Source: https://stackoverflow.com/questions/2158395
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

# Check path exist otherwise creat new
def check_and_create(out_dir):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
        
def global2local(data_mms, mm, nn):
    # global reduce to local coordinates
    data_mms[:,:3] = data_mms[:,:3] - [mm,nn,0]
    return data_mms #[:,:3] - [mm,nn,0]

def local2global(data_mms, mm, nn):
    # local back to global
    data_mms[:,:3] = data_mms[:,:3] + [mm,nn,0]
    return data_mms #[:,:3] + [mm,nn,0]