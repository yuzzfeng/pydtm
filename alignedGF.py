# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import os
import numpy as np
from environmentSetting import *
from tqdm import tqdm    

from lib.cell2world import coord
from lib.util import check_and_create
from lib.reference import filter_by_reference
from lib.ground_filtering import ground_filter
from lib.util import global2local, local2global
from lib.ply import write_points_double, read_bin_xyzrid_scale

def ground_filter_aligned_data(fn, args):
    
    pts_dir, ref_dir, tmp_dir, out_dir, r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list = args
    
    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)

    if fn not in os.listdir(ref_dir):
        return False
    
    # Load mms
    data_mms = read_bin_xyzrid_scale(pts_dir + fn, 12)
    
    # Rough filter based on given DTM
    data_mms = filter_by_reference(data_mms, ref_dir + fn, geoid, sigma_geoid)
    
    if len(data_mms) < 200:
        return False

    data_mms = global2local(data_mms, mm, nn)
    reduced, _  = ground_filter(data_mms, radius, res_list, r, is_plot = False)
    
    if len(reduced) < 200:
        return False

    write_points_double(local2global(reduced, mm, nn)[:,:3], 
                        out_dir + fn)
    return True


if __name__ == "__main__":
    
    ##radius = 3
    ##res_list = [1.0, 0.5, 0.25, 0.1, 0.05]
            
    args = [pts_dir, ref_dir, tmp_dir, gfilter_out_dir, 
            r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list]
    
    list_pointcloud = os.listdir(pts_dir)
    
    check_and_create(gfilter_out_dir)
    print("Total number of tiles:", len(list_pointcloud))
    
    list_tiles = sorted(set(list_pointcloud) - set(os.listdir(gfilter_out_dir)))
    print("Number of tiles still need filtering:", len(list_tiles))

    list_pointcloud = ["0000000f_ffffffde.ply"]
    #list_pointcloud = ["fffffff9_fffffff0.ply", "0000000c_fffffff5.ply", "0000002b_ffffffdc.ply"]
    
    result = []
    for fn in tqdm(list_pointcloud):
        print('processing:', fn)
        r = ground_filter_aligned_data(fn, args)
        result.append(r)
    print('process finished', sum(result), '/', len(result))