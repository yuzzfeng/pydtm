import numpy as np

# Multiprocessing
from multiprocessing import Pool
#from itertools import izip
from itertools import repeat

from lib.ply import read_bin
from lib.read import rasterize
from lib.shift import shiftvalue
from lib.load import load_mms

############################################################################


def calc_diff_core(fn, args):
    
    #mms_dir, ref_dir, res_ref, r, x_offset, y_offset, geoid, sigma_geoid  = args
    mms_dir, ref_dir, res_ref, r, x_offset, y_offset  = args
    
    data_mms = load_mms(fn, args)
    #data_mms = read_bin(mms_dir + fn, 7)
    data_ref = read_bin(ref_dir + fn, 7)

    d_ref = rasterize(data_ref, res_ref, dim = 2)
    
    shift_value, shift_img = shiftvalue(np.array(data_mms), res_ref, np.array(data_ref), d_ref, 1., r)
    
    return fn, shift_value, shift_img


def calc_diff_star(a_b):
    return calc_diff_core(*a_b)

def calc_diff(list_pointcloud_filtered, args):       

    #args = mms_dir, ref_dir, res_ref, r
    
    ## 6 Cores
    p = Pool(6)
    results = p.map(calc_diff_star, zip(list_pointcloud_filtered, repeat(args)))
    p.close()
    p.join()
    
#    results = []
#    for i in range(len(list_pointcloud_filtered)):
#        
#        result = calc_diff_core(list_pointcloud_filtered[i], args)
#        results.append(result)
#        
#        if i%20==0:
#            print(i, len(list_pointcloud_filtered))

    list_shift_value = []
    dict_shift_value = dict()
    list_shift_img = dict()

    for fn, shift_value, shift_img in results:
        
        list_shift_value.append(shift_value)
        list_shift_img[fn] = shift_img
        dict_shift_value[fn] = shift_value

    del results

    return list_shift_value, list_shift_img, dict_shift_value

def calc_diff_tqdm(list_pointcloud_filtered, args):       
    
    from tqdm import tqdm

    list_shift_value = []
    dict_shift_value = dict()
    list_shift_img = dict()
    
    for fn in tqdm(list_pointcloud_filtered):
        fn, shift_value, shift_img = calc_diff_core(fn, args)
        list_shift_value.append(shift_value)
        list_shift_img[fn] = shift_img
        dict_shift_value[fn] = shift_value

    return list_shift_value, list_shift_img, dict_shift_value

####################################################################################

from lib.tool import read_ascii_xyz

def grid_difference(fn_ref, res_ref, data_mms, nonvalue, raster_size):

    d_mms = rasterize(data_mms, res_ref, dim=2)
    
    # read reference
    data_ref,d_ref = read_ascii_xyz(fn_ref, delimiter=' ', skiprows=7, dtype=np.float, e=res_ref, dim=2)

    # calculate the difference
    difference = nonvalue * np.ones((raster_size,raster_size))

    for key, value in d_mms.iteritems():
        m,n = np.int0(key.split('+'))
        z_mms = data_mms[value,2]
        mean_mms = np.mean(z_mms)

        mean_ref = data_ref[d_ref[key],2]
        difference[m,n] = mean_mms - mean_ref
        
    return difference


def grid_difference_filter_based(fn_ref, res_ref, data_mms, nonvalue, raster_size, threshold_height_diff):

    d_mms = rasterize(data_mms, res_ref, dim=2)
    
    # read reference
    data_ref,d_ref = read_ascii_xyz(fn_ref, delimiter=' ', skiprows=7, dtype=np.float, e=res_ref, dim=2)

    # calculate the difference
    difference = nonvalue * np.ones((raster_size,raster_size))

    reduce_list = []

    for key, value in d_mms.iteritems():
        m,n = np.int0(key.split('+'))
        z_mms = data_mms[value,2]
        mean_mms = np.mean(z_mms)

        mean_ref = data_ref[d_ref[key],2]
        delta = mean_mms - mean_ref

        if delta > threshold_height_diff:
            reduce_list.extend(value)
        else:
            difference[m,n] = mean_mms - mean_ref
        
    return difference, reduce_list

# downsampling the points
def down_sampling(merged, res_output, method):

    d_merged = rasterize(merged, res_output, dim=3)
    reduced_cloud = [method(merged[value],axis=0) for key, value in d_merged.iteritems()]
    
    return np.array(reduced_cloud)
