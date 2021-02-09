import numpy as np
import os.path

from environmentSetting import * 
from lib.ply import read_bin_xyz_norm_scale

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

# Lidar Points Path
pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'
outputdir = "C:\\SVN\\datasets\\lidar\\ricklingen_adjusted_wall_ply\\"

directory = [ x for x in os.listdir(pointcloud_path) if x[-4:] == ".ply" ]
directory = reduce_range(directory)

for fn in directory:
    
    points = np.array(read_bin_xyz_norm_scale(pointcloud_path + fn, 13))
    normz = points[:, 5] < 0.2
    points = points[normz]
    
    if len(points)>0:
        tool.write_xyz(outputdir+'wall_'+fn, points[:, 0:3])
        print fn

