import numpy as np
import struct

from tool import gen_str

#####################################################################
# Write binary pointcloud
#####################################################################

def write_points(points, path):
    write_points_header(len(points), path)
    expand_points(points, path)

def write_points_header(count, path):
    with open(path, "w+") as file_:
        file_.write("""ply
format binary_little_endian 1.0
element vertex {0}
property float x
property float y
property float z
end_header\n""".format(count))

def expand_points(points, path):
    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("fff", *p)
            file_.write(txt)



def write_points_double(points, path):
    write_points_header_double(len(points), path)
    expand_points_double(points, path)

def write_points_header_double(count, path):
    with open(path, "w+") as file_:
        file_.write("""ply
format binary_little_endian 1.0
element vertex {0}
property double x
property double y
property double z
end_header\n""".format(count))

def expand_points_double(points, path):
    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("ddd", *p)
            file_.write(txt)


#####################################################################
# Read binary pointcloud
#####################################################################

# Read float binary pointcloud
def read_bin(path, skiprows):

    xyz = []
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
            xyz.append(point)

    return xyz

# Read double binary pointcloud
def read_bin_double(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*3)
            point = struct.unpack("ddd", binary)
            xyz.append(point)

    return xyz


# Read Claus's binary pointcloud
def read_bin_xyz_norm_scale(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*3 + 4*4)
            point = struct.unpack("dddffff", binary)
            xyz.append(point)

    return xyz
    