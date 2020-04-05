import numpy as np
import matplotlib.pyplot as plt

from lib.ply import read_bin
from lib.read import rasterize
    
def filter_by_reference(data_mms, fn, geoid, sigma_geoid):
    
    # Load reference dtm
    data_ref = read_bin(fn, 7)
    
    # Height range of the point cloud
    height_max, height_min = max(data_ref[:,2]) + geoid + sigma_geoid, min(data_ref[:,2]) + geoid - sigma_geoid
    
    # Remove points based on reference dtm
    data_mms = data_mms[(data_mms[:,2] > height_min) * (data_mms[:,2]< height_max)]
    
    return data_mms


def calc_difference_mms_ref(data_mms, data_ref, res_diff, r, shift,
                            is_plot = False):
    # Calculate a difference image between mms and ref
    difference = np.zeros((int(r/res_diff),int(r/res_diff)))
    
    d_mms_sub = rasterize(data_mms, res_diff, dim = 2)
    d_ref_sub = rasterize(data_ref, res_diff, dim = 2)
    
    for subkey, subvalue in d_mms_sub.iteritems():
        m,n = np.int0(subkey.split('+'))
        z_mms = np.median(data_mms[subvalue,2])
        z_ref = data_ref[d_ref_sub[subkey], 2][0] + shift
        difference[m,n] = z_mms - z_ref
    
    if is_plot:
        plt.figure()    
        plt.imshow(difference, vmin=-0.3, vmax=0.3)
        plt.colorbar()
        plt.figure()
        plt.hist(difference, bins='auto')
        plt.show()
    
    return difference

