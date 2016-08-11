import os.path
import math
import time
import random


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import libTool as tool
from libTool import flatten
import libCurb as Curb



# local functions
npsum = np.sum
npwhere = np.where
dot = np.dot
floor= math.floor
join = str.join


def filterNeigh(fn, featurePoint, intensity,  X0):

    from scipy import spatial
    tree = spatial.KDTree(zip(featurePoint[:,0], featurePoint[:,1], featurePoint[:,2]))
    t = []
    s = []
    for i in tree.data:
        ind = tree.query_ball_point(i,0.5)
        neigh = tree.data[ind]
        inten = intensity[ind]
        t.append(len(ind))
        s.append(np.var(inten))

    s = np.array(s)
    t = np.array(t)

    ind = (t>10) #*(s<4)  #0.5 radius number>10 var<1

##    tool.write_xyzi(fn, featurePoint[ind,0:3] + X0, featurePoint[ind,3])
    tool.write_xyzi(fn, featurePoint[ind,0:3], featurePoint[ind,3])


def curbdetection(fn):


    t = time.time()

    [fx,fy,runid] = fn.split('_')
    x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
    X0 = [x0,y0,0]
    
##    data,d = tool.iter_loadraster(inputdir+'\\'+fn, delimiter=' ', skiprows=9, dtype=np.float, args = args)
    
    data,d = tool.read_bin_xyz(inputdir+'\\'+fn, delimiter=' ', skiprows=7, dtype=np.float, args = args)

    if len(d)>0:
        print 'Number of rasters: ',len(d), ',Loading time: ',time.time()-t

        # Neighbourhood mask
        m = tool.gen_mat(*c)
        result = Curb.curb(d,data,m,threshold,6)


        result = np.array(result)
        list_points = list(flatten(result[:,0]))
        feature = np.array(list(flatten(result[:,1]))).reshape(-1,3)
        intensity = np.array(list(flatten(result[:,2])))


        featurePoint = np.hstack((feature, intensity[:, None]))
        featurePoint = tool.unique(featurePoint)

        
##        tool.write_xyzi('test.ply', featurePoint[:,0:3] + X0, featurePoint[:,3])
        filterNeigh(outputdir+'curb_'+fn, featurePoint, intensity, X0)

        print time.time()-t
        return 0

    else:
        print 'No proper data'
        return 0


if __name__ == '__main__':

    # Kaffeezimmer IKG
    x_offset = 548495
    y_offset = 5804458
    r = 25

    # Height range of the point cloud
    height_min = 101 #99.5
    height_max = 103 #101
    # Raster size
    eps = 0.1
    # cell_radius
    c = [2,2,2]

    args = [height_min,height_max,eps]

    # min_z, max_z, min_v, max_v, min_delta, max_delta
    threshold = [0.55, 0.90, 0.0008, 0.0032, 0.1, 0.2]

    inputdir = "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_3_Kacheln - ply\\00000000_ffffff40"
    outputdir = "C:\\_EVUS_DGM\\Curb\\"
    directory = [ x for x in os.listdir(inputdir) if x[-4:] == ".ply" ]

    map(curbdetection, directory)



