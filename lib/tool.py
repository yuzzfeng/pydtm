import numpy as np
import math
import struct

from scipy import spatial


#########################################################################################################
# Header of PLY
#########################################################################################################

ply_xyzrgb = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

ply_xyznormal = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property float n1
property float n2
property float n3
end_header
'''

ply_xyz = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

ply_xyzi = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property float scalar_Scalar_field
end_header
'''

floor= math.floor
join = str.join

#########################################################################################################
# given index for a list
#########################################################################################################

def selectbyindex(data, index):

##    return [data[ind] for ind in range(len(data)) if ind in index]

    return [data[ind] for ind in index]


#########################################################################################################
# Flatten irregular list
#########################################################################################################

def flatten(l):
    from collections import Iterable
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

#########################################################################################################
# Neighbour mask
#########################################################################################################

# 2n+1*2n+1*2n+1 window
def gen_mat(win_size_x,win_size_y,win_size_z):

    mat = []
    for i in xrange(-win_size_x,win_size_x+1):
        for j in xrange(-win_size_y,win_size_y+1):
            for k in xrange(-win_size_z,win_size_z+1):
                mat.append([i,j,k])
    return mat

#########################################################################################################
# Neighbour mask
#########################################################################################################

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]


#########################################################################################################
# Math
#########################################################################################################

def deg2rad(angle):
    return angle * math.pi/180

def Hexa2Decimal(x):
    tile = struct.unpack('>i', x.decode('hex'))
    return tile[0]

def int2hex(num):
    if num >=0:
        return '{:08x}'.format(num)
    else:
        return '{:08x}'.format(4294967296 + num)

#########################################################################################################
# Tile Tools
#########################################################################################################

def coord(m,n,r,x_offset,y_offset):
    return r*Hexa2Decimal(m)+x_offset, r*Hexa2Decimal(n)+y_offset


#########################################################################################################
# Merge Pointclouds
#########################################################################################################

def mergecloud(data1,data2):
    return np.vstack((data1,data2))


#########################################################################################################
# Rasterize *.ply data
#########################################################################################################

# insert nand update keys and values
def gen_str(coord,i):
    
    num = [str(int(floor(float(coord[0])/eps))),
           str(int(floor(float(coord[1])/eps))),
           str(int(floor(float(coord[2])/eps)))]
    string = join('+',num)
    
    if d.has_key(string) == True:
        app = d[string].append # This is a method
        app(i)
    else:    
        d[string] = [i]


# iterate each line in txt or ply data
def iter_loadraster(filename, delimiter, skiprows, dtype, args):

    global d, eps
    d = dict()
    height_min,height_max,eps = args
    
    def iter_func():
        i = 0
        with open(filename, 'r') as infile:
            for _ in xrange(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                z_tol = float(line[2])
                if z_tol<height_max and z_tol>height_min:
                    v = line[0:3]
                    gen_str(v,i)
                    i = i + 1
                    for item in v:
                        yield dtype(item)

    
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, 3))
    return data,d


def read_bin_xyz(path, delimiter, skiprows, dtype, args):

    global d,eps
    d = dict()
    height_min,height_max,eps = args
    
    xyz = []
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
            if z_tol<height_max and z_tol>height_min:
                xyz.append(point)
                gen_str(point,i)
                i = i + 1
    xyz = np.array(xyz)
    return xyz, d


def read_bin_xyz_return_list(path, delimiter, skiprows, dtype, args):

    global d,eps
    d = dict()
    height_min,height_max,eps = args
    
    xyz = []
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
            if z_tol<height_max and z_tol>height_min:
                xyz.append(point)
                gen_str(point,i)
                i = i + 1
    return xyz, d

# insert nand update keys and values
def gen_str_2d(coord,i):
    
    num = [str(int(floor(float(coord[0])/eps))),
           str(int(floor(float(coord[1])/eps)))]
    string = join('+',num)
    
    if d.has_key(string) == True:
        app = d[string].append # This is a method
        app(i)
    else:    
        d[string] = [i]


# iterate each line in txt or ply data
def rasterize(data, e):

    global d, eps
    d = dict()
    eps = e

    i = 0
    for point in data:
        gen_str_2d(point,i)
        i = i + 1    

    return d

def rasterize3d(data, e):

    global d, eps
    d = dict()
    eps = e

    i = 0
    for point in data:
        gen_str(point,i)
        i = i + 1    

    return d


# iterate each line in txt or ply data
def read_ascii_xyz(filename, delimiter, skiprows, dtype, e, dim):

    global d, eps
    d = dict()
    eps = e
    
    def iter_func():
        i = 0
        with open(filename, 'r') as infile:
            for _ in xrange(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                v = line[0:3]
                gen_str_2d(v,i)
                i = i + 1
                for item in v:
                    yield dtype(item)

    
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, 3))
    return data,d


#########################################################################################################
# Read and write *.ply data
#########################################################################################################

## Read PLY file binary


##def read_bin(path, skiprows):
##
##    xyz = []
##    with open(path, "rb") as file_:
##        skip = 0
##        for _ in xrange(skiprows):  
##            skip = skip + len(next(file_))
##            
##        file_.seek(0, 2)
##        size = file_.tell()
##        file_.seek(skip, 0)
##            
##        while file_.tell() < size:
##            binary = file_.read(4*3)
##            point = struct.unpack("fff", binary)
##            xyz.append(point)
##
##    return np.array(xyz)

## Read PLY file (Quicker than np.genfromtxt)
def iter_loadtxt(filename, delimiter, skiprows, dtype):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in xrange(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def read_ply(filename, sk):
    data = iter_loadtxt(filename, delimiter=' ', skiprows=sk, dtype=np.float)
    return data

## Write PLY file
def write_xyz(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'w') as f:
        f.write(ply_xyz % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_xyzrgb % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def write_xyzi(fn, verts, intensity):
    verts = verts.reshape(-1, 3)
    intensity = intensity.reshape(-1, 1)
    verts = np.hstack([verts, intensity])
    with open(fn, 'w') as f:
        f.write(ply_xyzi % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %f')


def save_color(fn,data,color):

    if color == 'red':
        c = [255,0,0]
    if color == 'green':
        c = [0,255,0]
    if color == 'blue':
        c = [0,0,255]

    d = np.ones((len(data),1))
    write_ply(fn,np.array(data),c*d)

#########################################################################################################
# Read and write *.csv data
#########################################################################################################

def read_csv(filename, sk, dt): # read data from csv file

    data = iter_loadtxt(filename, delimiter=',', skiprows=sk, dtype=dt)
    return data

def write_csv(name,data):  # Save the result as csv data   

    import csv
    csvFile = open(name, 'wb') 
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(data)              
    csvFile.close()

#########################################################################################################
# Load camera position data
#########################################################################################################

##def load_positions(indexstr, x0, y0, z0, dir_pos):
##
##    indexstr = '{0:04}'.format(index)
##    cam3 = read_csv("%s/Camera 3/140401_143424.csv" % dir_pos, sk=1, dt=str)
##    cam4 = read_csv("%s/Camera 4/140401_143424.csv" % dir_pos, sk=1, dt=str)
##
##    # Pose of the cameras measured by IMU
##    [roll3,pitch3,yaw3] = cam3[index-1,11:14].astype(np.float)
##    [roll4,pitch4,yaw4] = cam4[index-1,11:14].astype(np.float)


#########################################################################################################
# Image Processing
#########################################################################################################

#### Smooth the image with median filter and closing operation
##def img_smooth(img,maxKernal, morph_size):
##    import cv2
##    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2*morph_size+1,2*morph_size+1))
##    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
##    blur = cv2.medianBlur(closing,2*maxKernal+1)
##    return blur


#########################################################################################################
# Orientation
#########################################################################################################

##def vector2euler(vdir,vup):
##
##    x = np.array([0,1,0])
##    z = np.array([0,0,1])
##    
##    pitch = math.acos(np.dot(vdir,z)/ np.linalg.norm(vdir)) - math.pi/2
##
##    pj = vdir
##    pj[2] = 0
##    pj = pj/np.linalg.norm(pj)
##
##    yaw = math.acos(np.dot(pj,x)/ np.linalg.norm(pj))
##
##    cp = np.cross(z,vdir)
##    roll = math.acos(np.dot(cp,vup)/(np.linalg.norm(cp)*np.linalg.norm(vup)))
##    
##    return np.array([roll,pitch,yaw])

