import os 
import numpy as np

from lib.read import rasterize
from lib.ground_filtering import ground_filter
from lib.ply import read_bin_xyz_norm_scale
from lib.cell2world import coord, coord_fn_from_cell_index
from lib.ply import write_points, write_points_double, read_bin, read_bin_double
from lib.shift import  shiftvalue, reject_outliers


from lib.assemble import split_ref_to_tiles
from lib.produceDTM import local_to_UTM


x_offset = 548495 + 5
y_offset = 5804458 + 42
r = 15

ref_out_dir = 'C:\\temp\\aligned_ref\\'
##ref_path = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_ply\\1in4\\'
##split_ref_to_tiles(ref_path, ref_out_dir, r, x_offset, y_offset)


pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'
ground_filtering_out_dir = 'C:\\temp\\aligned_GF\\'
geo_ground_filtering_out_dir = 'C:\\temp\\aligned_GF_georefrenced\\'

list_pointcloud = os.listdir(pointcloud_path)
##for fn in list_pointcloud[493:]:
##
##    m,n = fn[:17].split('_')
##    [mm,nn] = coord(m, n, r, x_offset, y_offset)
##    
##    data = read_bin_xyz_norm_scale(pointcloud_path + fn, 13)
##    data_mms = np.array(data)[:,0:3] - [mm,nn,0]
##
##    if len(data_mms)>200:
##        radius = 3
##        res_list = [1.0, 0.5, 0.25, 0.1, 0.05]
##        reduced  = ground_filter(data_mms, radius, res_list, 15)
##        if len(reduced)>100:
##            write_points(reduced,  ground_filtering_out_dir + fn)
##
####            shift_value, shift_img = shiftvalue(reduced, res_ref, data_ref, d_ref, 1., r)
##            
##            print ground_filtering_out_dir + fn
##        else:
##            print fn
##    else:
##        print fn





##list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)
##list_pointcloud_ref = os.listdir(ref_out_dir)
##
##list_shift_value = []
##for fn in list_pointcloud_filtered:
##    if fn in list_pointcloud_ref:
##        data_mms = read_bin(ground_filtering_out_dir + fn, 7)
##        data_ref = read_bin(ref_out_dir + fn, 7)
##
##        d_ref = rasterize(data_ref, 0.5, dim = 2)
##        
##        shift_value, shift_img = shiftvalue(np.array(data_mms), 0.5, np.array(data_ref), d_ref, 1., 15)
##
##        list_shift_value.append(shift_value)
##    else:
##        print fn
##
##
##if 1:
##    ind = reject_outliers(list_shift_value, 5)
##
##    list_shift_value = np.array(list_shift_value)
##    hist, bins = np.histogram(list_shift_value[ind], bins=100)
##    import matplotlib.pyplot as plt
##    shift = np.median(list_shift_value[ind])
##    plt.plot(bins[:-1], hist)
##    plt.show()
##
##local_to_UTM(ground_filtering_out_dir, geo_ground_filtering_out_dir, shift, r, x_offset, y_offset)
