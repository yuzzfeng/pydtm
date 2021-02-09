import os.path
import math
import time
import numpy as np
from scipy import spatial

from environmentSetting import * 

import lib.tool as tool
from lib.flatten import flatten
import lib.curb as Curb
from lib.read import rasterize
from lib.tool import read_bin_auto, read_bin_args, read_bin_resmapling


# local functions
npsum = np.sum
npwhere = np.where
dot = np.dot
floor= math.floor
join = str.join


def filterNeigh(fn, featurePoint, intensity,  X0):

    
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
    
    if len(ind)>0:
        tool.write_xyzi(fn, featurePoint[ind,0:3] + X0, featurePoint[ind,3])
#    tool.write_xyzi(fn, featurePoint[ind,0:3], featurePoint[ind,3])
    
    return featurePoint[ind,0:3]


def curbdetection(fn):

#    t = time.time()
#    
#    if 1:
#        gf = read_bin_auto(gf_out_dir + fn)
#        height_min = min(np.array(gf)[:,2]) + global_offset - height_range
#        height_max = max(np.array(gf)[:,2]) + global_offset + height_range
#        args = [height_min, height_max, 0.01]
#    
#        [fx,fy] = fn[:-4].split('_')
#        x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
#        X0 = [x0,y0,0]
#    
#        data = read_bin_resmapling(inputdir+'\\'+fn, args, r, X0)
#        d = rasterize(data, eps, dim = 3) 
#    
##    data,d = read_bin_args(inputdir+'\\'+fn, [height_min, height_max, eps])
##    data = data[:,:3] - X0
#    
#    if len(d)>0:
#        print 'Number of rasters: ',len(d), ',Loading time: ',time.time()-t
#
#        # Neighbourhood mask
#        m = tool.gen_mat(*c)
#        
##        result = Curb.curb(d,data,m,threshold,6)
#        result = Curb.curb_single_core(d,data,m,threshold)
#
#        result = np.array(result)
##        list_points = list(flatten(result[:,0]))
#        feature = np.array(list(flatten(result[:,1]))).reshape(-1,3)
#        intensity = np.array(list(flatten(result[:,2])))
#
#
#        featurePoint = np.hstack((feature, intensity[:, None]))
#        featurePoint = tool.unique(featurePoint)
#
#        
#        filtered = filterNeigh(outputdir+'curb_'+fn, featurePoint, intensity, X0)
#
#        print time.time()-t
#        return filtered
#
#    else:
#        print 'No proper data'
#        return 0

        t = time.time()
        
        gf = read_bin_auto(gf_out_dir + fn)
        height_min = min(np.array(gf)[:,2]) + global_offset - height_range
        height_max = max(np.array(gf)[:,2]) + global_offset + height_range
        args = [height_min, height_max, eps]
    
        [fx,fy] = fn[:-4].split('_')
        x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
        X0 = [x0,y0,0]
    
        data, d = read_bin_resmapling(inputdir+'\\'+fn, args, r, X0)
        
        if len(d)>0 and len(data)>5000:
            
            print fn, 'Number of rasters: ',len(d), ',Loading time: ',time.time()-t
    
            # Neighbourhood mask
            m = tool.gen_mat(*c)
            
#            result = Curb.curb(d,data,m,threshold,6)
            result = Curb.curb_single_core(d,data,m,threshold)
            
            result = np.array(result)
            feature = np.array(list(flatten(result[:,1]))).reshape(-1,3)
            intensity = np.array(list(flatten(result[:,2])))
            
            featurePoint = np.hstack((feature, intensity[:, None]))
            featurePoint = tool.unique(featurePoint)
            
            if len(featurePoint)>0:
                tool.write_xyzi(outputdir+'curb_'+fn, featurePoint[:,0:3] + X0, featurePoint[:,3])
    
#            filtered = filterNeigh(outputdir+'curb_'+fn, featurePoint, intensity, X0)
            
            idx = np.array(list(flatten(result[:,0])))
            if len(idx)>0:
                tool.write_xyz(outputdir+'curbxxxx_'+fn, data[idx]+ X0)
            
            print 'processing finished'
        else:
            print fn, 'no sufficient data'

        print time.time()-t


def reduce_range(directory):
    
    Xmin = 549164.545
    Xmax = 549499.120
    
    Ymin = 5799792.721
    Ymax = 5800017.458
    
    subset = []
    for fn in directory:    
        [fx,fy] = fn[:-4].split('_')
        x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
        if x0 > Xmin and x0 <Xmax and y0 > Ymin and y0 <Ymax:
            subset.append(fn)
    
    return subset
    
    

if __name__ == '__main__':
    

    outputdir = "C:\\SVN\\datasets\\lidar\\ricklingen_adjusted_curb_ply\\"   
    gf_out_dir = out_path + 'aligned\\'
    
    inputdir = "C:\\SVN\\datasets\\lidar\\ricklingen_adjusted_cloud_40m_ply\\"   
    directory = [ x for x in os.listdir(inputdir) if x[-4:] == ".ply" ]
    directory = reduce_range(directory)
    
    global_offset = 42.9317864988 
    height_range = 0.05

    # Raster size
    eps = 0.1
    # cell_radius
    c = [1,1,2]

    # min_z, max_z, min_v, max_v, min_delta, max_delta, hori_dist
    threshold = [0.55, 0.90, 0.0004, 0.0030, 0.03, 0.18, 0.05]
    
    fn = "0000002d_fffffec7.ply"
    featurePoint = curbdetection(fn)
    
#    directory = directory[144:146]
#    map(curbdetection, directory)
#    print time.time()
    
    
#    featurePoint = curbdetection(fn)
##    map(curbdetection, [directory[-2]])
