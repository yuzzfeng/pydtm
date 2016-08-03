import numpy as np
import struct

from asc import read_asc

#######################################################################
# From the local to global coordinates system
# Author: Yu Feng
#######################################################################

### Kaffeezimmer IKG
##x_offset = 548495
##y_offset = 5804458
##r = 25


#######################################################################
# get the coordinates of the left down corder from file name
#######################################################################

def Hexa2Decimal(x):
    tile = struct.unpack('>i', x.decode('hex'))
    return tile[0]

def coord(m,n,r,x_offset,y_offset):
    return r*Hexa2Decimal(m)+x_offset, r*Hexa2Decimal(n)+y_offset

def read_cellname(fn,r,x_offset,y_offset):
    m,n = fn.split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def read_fn(fn,r,x_offset,y_offset):
    m,n,runid = fn.split('.')[0].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def read_fn_runid(fn,r,x_offset,y_offset):
    m,n,runid = fn.split('.')[0].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn, runid

#######################################################################
# From the coordinates to the cell name
#######################################################################

def int2hex(num):
    if num >=0:
        return '{:08x}'.format(num)
    else:
        return '{:08x}'.format(4294967296 + num)
    
def coord_fn(x,y,runid):

    hexx = int2hex((x - x_offset)/r)
    hexy = int2hex((y - y_offset)/r)

    return '_'.join([hexx,hexy,runid])
    

def coord_fn_from_cell_index(x,y,runid):

    hexx = int2hex(x)
    hexy = int2hex(y)
    return '_'.join([hexx,hexy,runid]), '_'.join([hexx,hexy])

   
def max_runid_range(M, N, minM, minN, runid):

    new_fn = []
    for i in xrange(M/r):
        for j in xrange(N/r):
            new_fn.append( coord_fn(minM + i * r, minN + j * r, runid) + '.ply' )

    return new_fn


# First check of the size of the current runid
def runid_size(fn_list):

    # iterate each cell
    mn = [read_fn(fn) for fn in fn_list]

    minM, minN = np.min(mn, axis=0)
    maxM, maxN = np.max(mn, axis=0)

    M = maxM - minM + r
    N = maxN - minN + r

    return M, N, minM, minN


# Merge the diff raster(asc) in one runid
def merge_all_cells_in_one_runid(fn_list, M, N, minM, minN, scale, nonvalue, path):

    diff_img = np.zeros((M * scale, N * scale))

    for fn in fn_list:

        mm,nn = read_fn(fn)
        mm,nn = mm-minM, nn-minN
        data = read_asc(path + fn)

        diff_img[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale] = data

    np.place(diff_img, diff_img == nonvalue, 0)
    return diff_img 
