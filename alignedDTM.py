import os 
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from environmentSetting import *

from lib.checkRunids import check_and_create
from lib.diff import calc_diff
from lib.report import generate_report, plot_img
from lib.update import update_dtm
from lib.produceDTM import local_to_UTM, local_to_UTM_update_ref, local_to_UTM_rest_ref

##from lib.read import rasterize
##from lib.cell2world import Hexa2Decimal, int2hex, coord_fn_from_cell_index
##from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
##from lib.shift import  shiftvalue, reject_outliers
##from lib.load import load_aligned_data
##from lib.boundaries import apply_gaussian
##
##def search_index(list_pointcloud, name):
##    list_pointcloud = np.array(list_pointcloud)
##    return np.where(list_pointcloud==name)
##
##def calc_difference_mms_ref(fn, args):
##
##    list_pointcloud_ref, ground_filtering_out_dir, ref_out_dir = args
##
##    if fn in list_pointcloud_ref:
##        data_mms = read_bin(ground_filtering_out_dir + fn, 7)
##        data_ref = read_bin(ref_out_dir + fn, 7)
##
##        d_ref = rasterize(data_ref, 0.5, dim = 2)
##            
##        shift_value, shift_img = shiftvalue(np.array(data_mms), res_ref, np.array(data_ref), d_ref, 1., r)
##            
##        return fn, shift_value, shift_img
##
##list_pointcloud = os.listdir(pointcloud_path)



if __name__ == "__main__":

    list_pointcloud_ref = os.listdir(ref_out_dir)
    list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)
    
    list_shift_value, list_shift_img = calc_diff(list_pointcloud_filtered, ground_filtering_out_dir, ref_out_dir, res_ref, r)
    print 'difference calculated'
    
    shift = generate_report(list_shift_value, list_shift_img, out_path, r, x_offset,y_offset)
    print 'report generated'
    
    raster_size = 30
    radius = 1

    update_dtm(list_shift_img, raster_size, radius, ref_cut_dir, ref_update_dir,
               shift, res_ref, list_pointcloud_ref, ref_out_dir)

    check_and_create(geo_ground_filtering_out_dir)
    check_and_create(final_dir)
    check_and_create(rest_dir)
    
##    shift = 42.9317700613 #42.9316920581

    # Process the mms and update of ref together 
    local_to_UTM(ground_filtering_out_dir, geo_ground_filtering_out_dir, ref_cut_dir, shift, r, x_offset, y_offset)

    # Process the update ref and combine the duplicated ones
    local_to_UTM_update_ref(ref_update_dir, final_dir, r, x_offset, y_offset)

    # Process the rest of tiles into UTM32 global coordinate system
    list_rest = list(set(list_pointcloud_ref) - set(os.listdir(final_dir)) - set(os.listdir(geo_ground_filtering_out_dir)))
    local_to_UTM_rest_ref(list_rest, ref_out_dir, rest_dir, r, x_offset, y_offset)

    print 'Process finished'
