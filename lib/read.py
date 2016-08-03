import numpy as np
import math
import struct
import random

floor= math.floor
join = str.join

def selectbyindex(data, index):
    return [data[ind] for ind in index]


# insert and update keys and values
def gen_str(coords, i):

    num = [str(int(floor(float(coord)/eps))) for coord in coords]
    string = join('+',num)
    
    if d.has_key(string) == True:
        app = d[string].append # This is a method
        app(i)
    else:    
        d[string] = [i]


# iterate each line in txt or ply data
def rasterize(data, e, dim = 3):

    global eps, d
    eps = e
    d = dict()
    i = 0

    if dim == 2:
        for point in data:
            gen_str(point[0:2], i)
            i = i + 1

    if dim == 3:
        for point in data:
            gen_str(point, i)
            i = i + 1   

    return d


def downSampling(data, e):

    d = rasterize(data, e, 3)
    
    resampled = []
    for key, value in d.iteritems():
        resampled.append(random.choice(value))

    return resampled



def read_bin_resmapling(path, skiprows, args):

    global eps, d
    height_min,height_max,eps = args

    xyz = []
    d = dict()
    i = 0

    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(4*3)
            point = struct.unpack("fff", binary)

            z_tol = point[2]
            if z_tol < height_max and z_tol > height_min and point[0]<25.0 and point[1]<25.0:
                xyz.append(point)
                gen_str(point, i)
                i = i + 1

    resampled = []
    for key, value in d.iteritems():
        resampled.append(random.choice(value))
                
    return selectbyindex(xyz, resampled)


# Read ascii pointcloud - iterate each line in txt or ply data
def read_ascii_xyz(filename, delimiter, skiprows, dtype, e, dim):

    global eps, d
    eps = e
    d = dict()
    
    def iter_func():    
        i = 0
        with open(filename, 'r') as infile:
            for _ in xrange(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                point = line[0:3]
                gen_str(point[0:dim], i)
                i = i + 1
                for item in point:
                    yield dtype(item)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, 3))
    return data, d
