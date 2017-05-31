import os.path
import math
import time
import numpy as np
from scipy import spatial

import lib.tool as tool
from lib.flatten import flatten
import lib.curb as Curb


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

##    tool.write_xyzi(fn, featurePoint[ind,0:3] + X0, featurePoint[ind,3])
    tool.write_xyzi(fn, featurePoint[ind,0:3], featurePoint[ind,3])
    
    return featurePoint[ind,0:3]


def curbdetection(fn):


    t = time.time()

    [fx,fy] = fn[:-4].split('_')
    x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
    X0 = [x0,y0,0]
    
    data, d = tool.read_ascii_xyz(inputdir + fn, delimiter=' ', skiprows=7, dtype=np.float, e=eps, dim=3)
    
##    data,d = tool.iter_loadraster(inputdir+'\\'+fn, delimiter=' ', skiprows=9, dtype=np.float, args = args)
    
#    data,d = tool.read_bin_xyz(inputdir+'\\'+fn, delimiter=' ', skiprows=7, dtype=np.float, args = args)

    if len(d)>0:
        print 'Number of rasters: ',len(d), ',Loading time: ',time.time()-t

        # Neighbourhood mask
        m = tool.gen_mat(*c)
        result = Curb.curb(d,data,m,threshold,6)
#        result = Curb.curb_single_core(d,data,m,threshold)


        result = np.array(result)
        list_points = list(flatten(result[:,0]))
        feature = np.array(list(flatten(result[:,1]))).reshape(-1,3)
        intensity = np.array(list(flatten(result[:,2])))


        featurePoint = np.hstack((feature, intensity[:, None]))
        featurePoint = tool.unique(featurePoint)

        
##        tool.write_xyzi('test.ply', featurePoint[:,0:3] + X0, featurePoint[:,3])
        filtered = filterNeigh(outputdir+'curb_'+fn, featurePoint, intensity, X0)

        print time.time()-t
        return filtered

    else:
        print 'No proper data'
        return 0


if __name__ == '__main__':

    # Kaffeezimmer IKG
    x_offset = 548495
    y_offset = 5804458
    r = 25
    
    global_offset = 42.9317864988 
    height_range = 0.3

    # Height range of the point cloud
    #height_min = 101 #99.5
    #height_max = 103 #101
    # Raster size
    eps = 0.1
    # cell_radius
    c = [1,1,1]

    # min_z, max_z, min_v, max_v, min_delta, max_delta
    threshold = [0.55, 0.90, 0.0004, 0.0032, 0.05, 0.2, 0.05] # default
    
    threshold = [0.55, 0.90, 0.0004, 0.0016, 0.05, 0.25, 0.1] 

    
#    kn = "00000000_ffffff40"
#    inputdir = "..\\pydtm\\testData\\" + kn
    outputdir = "..\\pydtm\\testData\\Curb\\"
#    refdir = "..\\pydtm\\testData\\REF_"+kn+".ply"
#    gfdir = "..\\pydtm\\testData\\GF_"+kn+".ply"
#    directory = [ x for x in os.listdir(inputdir) if x[-4:] == ".ply" ]
#    gf, _ = tool.read_bin_xyz_return_list(gfdir, delimiter=' ', skiprows=7, dtype=np.float, args = [0,200,0.1])
    
    fn = "0000003e_fffffee1.ply"
    inputdir = "..\\pydtm\\"
    gfdir = "..\\pydtm\\"
    
    gf,_ = tool.read_ascii_xyz(gfdir + fn, delimiter=' ', skiprows=7, dtype=np.float, e=eps, dim=3)

    height_min = min(np.array(gf)[:,2]) + global_offset - height_range
    height_max = max(np.array(gf)[:,2]) + global_offset + height_range
    args = [height_min,height_max,eps]
    
    #ref,_ = tool.read_ascii_xyz(refdir, delimiter=' ', skiprows=7, dtype=np.float, args = args)
    
    
    t = time.time()

    [fx,fy] = fn[:-4].split('_')
    x0,y0 = tool.coord(fx,fy,r,x_offset,y_offset)
    X0 = [x0,y0,0]
    
    data, d = tool.read_ascii_xyz(inputdir + fn, delimiter=' ', skiprows=7, dtype=np.float, e=eps, dim=3)

    if len(d)>0:
        print 'Number of rasters: ',len(d), ',Loading time: ',time.time()-t

        # Neighbourhood mask
        m = tool.gen_mat(*c)
#        result = Curb.curb(d,data,m,threshold,6)
        result = Curb.curb_single_core(d,data,m,threshold)


        result = np.array(result)
        list_points = list(flatten(result[:,0]))
        feature = np.array(list(flatten(result[:,1]))).reshape(-1,3)
        intensity = np.array(list(flatten(result[:,2])))


        featurePoint = np.hstack((feature, intensity[:, None]))
        featurePoint = tool.unique(featurePoint)

        
#        tool.write_xyzi('test.ply', featurePoint[:,0:3] + X0, featurePoint[:,3])
        filtered = filterNeigh(outputdir+'curb_'+fn, featurePoint, intensity, X0)

        print time.time()-t
        
        tool.write_xyz('jumps.ply',data[list_points])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###################################################################

#    map(curbdetection, [directory[-2]])

#    for fn in [directory[-2]]:
#    featurePoint = curbdetection(fn)
    
    
#    import fiona
#    import pandas as pd
#    import geopandas as gp
#    import matplotlib.pyplot as plt
#    from shapely.geometry import LineString
#    from sklearn.cluster import DBSCAN
#    
#    if 1:
#    
#        X = featurePoint[:,:2]
#        
#        db = DBSCAN(eps=0.20, min_samples=5).fit(X)
#        labels = db.labels_
#        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#        
#        lines = []
#        # Black removed and is used for noise instead.
#        unique_labels = set(labels)
#        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#        angles = X[:,1] / X[:,0]
#        
#        for k, col in zip(unique_labels, colors):
#            if k != -1:
#                class_member_mask = (labels == k)
#                plt.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o', markerfacecolor=col,
#                         markeredgecolor='k', markersize=3)
#                         
#                angles[class_member_mask]
#                         
#                line = LineString(X[class_member_mask][np.argsort(angles[class_member_mask])])
#                tolerance = 0.025
#                simplified_line = line.simplify(tolerance, preserve_topology=False)
#                lines.append(simplified_line)
#        
#        g = gp.GeoSeries(lines)
#        g.to_file("lines.shp")
#
#
#
