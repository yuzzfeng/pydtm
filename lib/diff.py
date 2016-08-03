import numpy as np
from tool import read_ascii_xyz
from read import rasterize

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
