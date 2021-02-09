# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

# TODO: To be deleted
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

import os 
import numpy as np
from itertools import product

from environmentSetting import *

from lib.util import check_and_create
from lib.diff import calc_diff, calc_diff_core
from lib.report import generate_report, plot_img
from lib.update import update_dtm
from lib.produceDTM import local_to_UTM, local_to_UTM_update_ref, local_to_UTM_rest_ref

if __name__ == "__main__":
    
    is_calc = True
    
    list_pointcloud_ref = [fn for fn in os.listdir(ref_dir) if '.ply' in fn] 
    
#    # Many of the gf files are processed
#    if len(os.listdir(ground_filtering_adp_out_dir)) > len(os.listdir(gfilter_out_dir))*2/3:
#        list_pointcloud_filtered = os.listdir(ground_filtering_adp_out_dir)
#        mms_dir = ground_filtering_adp_out_dir
#    else:
    if 1:
        list_pointcloud_filtered = os.listdir(gfilter_out_dir)
        mms_dir = gfilter_out_dir
        
    #args = mms_dir, ref_out_dir, res_ref, r
    args = mms_dir, ref_dir, res_ref, r, x_offset, y_offset, geoid, sigma_geoid
    
    if is_calc:
        
        list_shift_value, list_shift_img, dict_shift_value = calc_diff(list_pointcloud_filtered, args)
        print 'difference calculated'
        
        # Save the dict for shift values
        np.save(project_name + "_shiftx.npy", [dict_shift_value, 0])   
        np.save(project_name + "_shift_imgx.npy", [list_shift_img, 0])    
        np.save(project_name + "_shift_valuex.npy", [list_shift_value, 0])    
    
    else:
        
        dict_shift_value = np.load(project_name + "_shiftx.npy")[0]
        list_shift_img = np.load(project_name + "_shift_imgx.npy")[0]
        list_shift_value = np.load(project_name + "_shift_valuex.npy")[0]
        
    list_shift_value = np.array(list_shift_value)[np.array(list_shift_value)!=None]
    
    # Find the invalid tiles
    invalid = []
    for key,value in dict_shift_value.iteritems():
        if not value:
            invalid.append(key)
        else:
            if np.isnan(value):
                invalid.append(key)
    
    # Remove the invalid from dictionary
    for key in invalid:
        list_shift_img.pop(key)
        dict_shift_value.pop(key)

    shift = generate_report(list_shift_value, list_shift_img, dict_shift_value, 
                            out_path, r, x_offset,y_offset, res_ref)
    print 'report generated'

    raster_size = r/res_ref #30
    radius = 1
    
    check_and_create(geo_ground_filtering_out_dir)
    check_and_create(final_dir)
    check_and_create(rest_dir)
    
if 0:    
    update_dtm(list_shift_img, raster_size, radius, ref_cut_dir, ref_update_dir,
               shift, res_ref, list_pointcloud_ref, ref_dir)
    print 'updates generated'

   
    #shift = 42.9317700613 # Hannover - 42.9316920581
    #shift = 43.5341477257 # Hildesheim - After reject outlier 
    #shift = 43.5346042674 # Hildesheim - Before reject outlier 
    #shift = 43.4262356488 # SEhi 
    # Process the mms and update of ref together 
    local_to_UTM(mms_dir, dict_shift_value.keys(), geo_ground_filtering_out_dir, 
                 ref_cut_dir, shift, r, x_offset, y_offset)

    # Process the update ref and combine the duplicated ones
    local_to_UTM_update_ref(ref_update_dir, final_dir, r, x_offset, y_offset)

    # Process the rest of tiles into UTM32 global coordinate system
    list_rest = list(set(list_pointcloud_ref) - set(os.listdir(final_dir)) - set(os.listdir(geo_ground_filtering_out_dir)))
    local_to_UTM_rest_ref(list_rest, ref_dir, rest_dir, r, x_offset, y_offset)
        
    print 'Process finished'
