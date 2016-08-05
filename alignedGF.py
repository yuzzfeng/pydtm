import os 
import numpy as np

from lib.load import load_aligned_data
from lib.ground_filtering import ground_filter
from lib.cell2world import coord, coord_fn_from_cell_index
from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale

ref_out_dir = 'C:\\temp\\aligned_ref\\'
list_pointcloud_ref = os.listdir(ref_out_dir)
    
pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'
list_pointcloud = os.listdir(pointcloud_path)

def ground_filter_aligned_data(fn, args):

    [pointcloud_path, ground_filtering_out_dir, r, x_offset, y_offset, geoid, sigma_geoid] = args

    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)

    if fn in list_pointcloud_ref:
        data_mms = load_aligned_data(fn, pointcloud_path, ref_out_dir, mm, nn, geoid, sigma_geoid)  
        if len(data_mms)>200:
            radius = 3
            res_list = [1.0, 0.5, 0.25, 0.1, 0.05]
            reduced  = ground_filter(data_mms, radius, res_list, r)
            if len(reduced)>200:
                write_points(reduced,  ground_filtering_out_dir + fn)
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def ground_filter_aligned_data_star(a_b):
    return ground_filter_aligned_data(*a_b)

def ground_filter_multicore(list_pointcloud, args):

    # Multiprocessing
    from multiprocessing import Pool
    from itertools import izip
    from itertools import repeat

    ## 6 Cores
    p = Pool(6)
    result = p.map(ground_filter_aligned_data_star, zip(list_pointcloud, repeat(args)))
    p.close()
    p.join()

    return result



if __name__ == "__main__":

    geoid = 42.9664
    sigma_geoid = 0.5

    x_offset = 548495 + 5
    y_offset = 5804458 + 42
    r = 15

    ground_filtering_out_dir = 'C:\\temp\\aligned_GF_04082016\\'
    args = [pointcloud_path, ground_filtering_out_dir, r, x_offset, y_offset, geoid, sigma_geoid]
    result = ground_filter_multicore(list_pointcloud, args)
