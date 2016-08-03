import numpy as np

#########################################################################################
# Write and read asc file (ASCII Raster)
# Author: Yu Feng
#########################################################################################

asc_header = '''ncols %d
nrows %d
xllcorner %d
yllcorner %d
cellsize %f
nodata_value %f'''


def write_asc(fn, mm, nn, data, res):

    ncols = 50
    nrows = 50
    cellsize = res
    nodata_value = -999.00

    np.savetxt(fn + '.asc', data, delimiter=' ', header = asc_header % (ncols, nrows, mm, nn, cellsize, nodata_value),
               fmt='%f', comments='')

def read_asc(fn):

    return np.loadtxt(fn,  delimiter=' ', skiprows= 6, dtype = np.float)
