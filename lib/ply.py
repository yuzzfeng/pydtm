import numpy as np
import struct

from tool import gen_str

#####################################################################
# Write binary pointcloud
#####################################################################

dict_pattern = {'double': 'd',
                'float': 'f',
                'short': 'h',
                'int': 'i',
                }

def write_points_bin(points, path, fields, dtypes):
    
    count = len(points)
  
    pattern = ""
    header = """ply\nformat binary_little_endian 1.0\nelement vertex {0}\n""".format(count)
    for f,d in zip(fields, dtypes):
        header += " ".join(["property", d, f, '\n'])
        pattern += dict_pattern[d]
    header += """end_header\n"""
    
    with open(path, "w+") as file_:
        file_.write(header)
        
    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack(pattern, *p)
            file_.write(txt)

def write_points_ddd(points, path):
    fields = ["x", "y", "z"]
    dtypes = ['double', 'double', 'double']
    write_points_bin(points, path, fields, dtypes)

def write_points_ddfi(points, path):
    fields = ["x", "y", "z", "run-id"]
    dtypes = ['double', 'double', 'float', 'int']
    write_points_bin(points, path, fields, dtypes)


#####################################################################
# Write binary pointcloud - old way - tobedelate
#####################################################################
def write_points_global(points, path):
    
    count = len(points)
    
    with open(path, "w+") as file_:
        file_.write("""ply
                    format binary_little_endian 1.0
                    element vertex {0}
                    property double x
                    property double y
                    property float z
                    end_header\n""".format(count))
    
    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("ddf", *p)
            file_.write(txt)

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

    return np.array(xyz)

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

    return np.array(xyz)

# Read double binary pointcloud
def read_bin_ddf(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*2 + 4)
            point = struct.unpack("ddf", binary)
            xyz.append(point)
    return np.array(xyz)


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

    return np.array(xyz)

# Read Claus's 40m binary pointcloud: double-double-float-int
def read_bin_xyz_scale(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*2 + 2*4)
            point = struct.unpack("ddfi", binary)
            xyz.append(point)

    return np.array(xyz)

# Read Claus's 25m binary pointcloud: double-double-float-short
def read_bin_xyzr_scale(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*2 + 1*4 + 2)
            point = struct.unpack("ddfh", binary)
            xyz.append(point)

    return np.array(xyz)

# Read Claus's 25m binary pointcloud: double-double-float-short
def read_bin_xyzrid_scale(path, skiprows):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in xrange(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*2 + 4 + 2 + 4)
            point = struct.unpack("ddfh", binary[:22])
            runid = struct.unpack("i", binary[22:])
            xyz.append(point + runid)

    return np.array(xyz)