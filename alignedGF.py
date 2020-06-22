# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import os, random
import numpy as np
from environmentSetting import *

from lib.cell2world import coord
from lib.util import check_and_create
from lib.reference import filter_by_reference
from lib.ground_filtering import ground_filter
from lib.util import global2local, local2global
from lib.ply import write_points_double, read_bin_xyzrid_scale

def ground_filter_aligned_data(fn, args):
    #[pointcloud_path, ground_filtering_out_dir, r, x_offset, y_offset, geoid, sigma_geoid] = args
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

def ground_filter_aligned_data_star(a_b):
    return ground_filter_aligned_data(*a_b)

def ground_filter_multicore(list_pointcloud, args):
    # Multiprocessing
    from multiprocessing import Pool
    from itertools import repeat
    #from itertools import izip
    
    ## 6 Cores
    p = Pool(6)
    result = p.map(ground_filter_aligned_data_star, zip(list_pointcloud, repeat(args)))
    p.close()
    p.join()
    return result


if __name__ == "__main__":
    
    radius = 3
    res_list = [1.0, 0.5, 0.25, 0.1, 0.05]
            
    #args = [pointcloud_path, ground_filtering_out_dir, 
    #        r, x_offset, y_offset, geoid, sigma_geoid]
    args = [pts_dir, ref_dir, tmp_dir, gfilter_out_dir, 
            r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list]
    
    list_pointcloud = os.listdir(pts_dir)
    
    check_and_create(gfilter_out_dir)
    print("Total number of tiles:", len(list_pointcloud))
    
    list_tiles = sorted(set(list_pointcloud) - set(os.listdir(gfilter_out_dir)))
    print("Number of tiles still need filtering:", len(list_tiles))

    ## Process all with 6 cores
    #random.shuffle(list_tiles) 
    #result = ground_filter_multicore(list_tiles, args)
    #print('process finished', sum(result), '/', len(result))
    
    from tqdm import tqdm    
    result = []
    for fn in tqdm(list_pointcloud):
        r = ground_filter_aligned_data(fn, args)
        result.append(r)
    print('process finished', sum(result), '/', len(result))
    
    ## Rickling bugs
    #list_pointcloud = ['00000017_fffffebc.ply', '00000018_fffffeac.ply']
    #
    #list_pointcloud = ["00000032_fffffff1.ply", "0000001b_fffffff4.ply", 
    #                   "00000028_fffffff1.ply", "00000015_fffffff6.ply"]
    #list_pointcloud = ["00000019_fffffff5.ply", "00000019_fffffff6.ply", 
    #                   "00000019_fffffff7.ply", "0000001a_fffffff5.ply",
    #                   "0000001a_fffffff6.ply", "0000001a_fffffff7.ply"]
    ## Test with single core
    #for fn in list_pointcloud:
    #    ground_filter_aligned_data(fn, args)