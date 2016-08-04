import os 
import numpy as np

from lib.read import rasterize
from lib.ground_filtering import ground_filter
from lib.cell2world import coord, coord_fn_from_cell_index
from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
from lib.shift import  shiftvalue, reject_outliers

from lib.load import load_aligned_data

from lib.assemble import split_ref_to_tiles
from lib.produceDTM import local_to_UTM

def search_index(list_pointcloud, name):
    list_pointcloud = np.array(list_pointcloud)
    return np.where(list_pointcloud==name)

import matplotlib.pyplot as plt
def plot_img(img):  
    plt.figure()
    plt.imshow(img)
    plt.colorbar()

def read_fn(fn,r,x_offset,y_offset):
    m,n = fn.split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def get_range_from_fn_list(fn_list, r,x_offset,y_offset):
    mn = [read_fn(fn[:17],r,x_offset,y_offset) for fn in list_shift_img.keys()]
    minM, minN = np.min(mn, axis=0)
    maxM, maxN = np.max(mn, axis=0)

    M = maxM - minM + r
    N = maxN - minN + r

    return minM, minN, M/r, N/r

geoid = 42.9664
sigma_geoid = 0.4

x_offset = 548495 + 5
y_offset = 5804458 + 42
r = 15


ref_out_dir = 'C:\\temp\\aligned_ref\\'
##ref_path = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_ply\\1in4\\'
##split_ref_to_tiles(ref_path, ref_out_dir, r, x_offset, y_offset)
list_pointcloud_ref = os.listdir(ref_out_dir)

pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'

ground_filtering_out_dir = 'C:\\temp\\aligned_GF\\'


geo_ground_filtering_out_dir = 'C:\\temp\\aligned_GF_georefrenced\\'


list_pointcloud = os.listdir(pointcloud_path)
list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)





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

##    for fn in list_pointcloud:
##        print ground_filter_aligned_data(fn, args)

if __name__ == "__main__":

    print search_index(list_pointcloud, '0000004f_fffffec2.ply'), len(list_pointcloud)

    
##    ground_filtering_out_dir = 'C:\\temp\\aligned_GF_04082016\\'
##    args = [pointcloud_path, ground_filtering_out_dir, r, x_offset, y_offset, geoid, sigma_geoid]
##    ground_filter_multicore(list_pointcloud[:20], args)


    list_shift_value = []
    list_shift_img = dict()

    for fn in list_pointcloud_filtered:
        if fn in list_pointcloud_ref:
            data_mms = read_bin(ground_filtering_out_dir + fn, 7)
            data_ref = read_bin(ref_out_dir + fn, 7)

            d_ref = rasterize(data_ref, 0.5, dim = 2)
            
            shift_value, shift_img = shiftvalue(np.array(data_mms), 0.5, np.array(data_ref), d_ref, 1., 15)

    ##        plot_img(shift_img - shift_value)
    ##        plt.show()
            
            list_shift_value.append(shift_value)
            list_shift_img[fn] = shift_img
        else:
            print fn


    if 1:
        ind = reject_outliers(list_shift_value, 5)

        list_shift_value = np.array(list_shift_value)
        hist, bins = np.histogram(list_shift_value[ind], bins=100)
        
        shift = np.median(list_shift_value[ind])
        plt.plot(bins[:-1], hist)
        plt.show()

    print shift
    minM, minN, lenM, lenN = get_range_from_fn_list(list_shift_img.keys(), r,x_offset,y_offset)

    if 1:
        img = np.zeros((lenN, lenM))
        i = 0
        for fn in list_shift_img.keys():
            if ind[i]: 
                x,y = read_fn(fn[:17], r, x_offset, y_offset)
                img[lenN -1  - (y-minN)/r, (x-minM)/r] = list_shift_value[i] - shift
            i = i + 1

        plot_img(img)
        plt.show()

    print np.std(list_shift_value[ind] - shift), np.max(list_shift_value[ind] - shift), np.min(list_shift_value[ind] - shift)

##    for fn in list_shift_img.keys()[5:15]:
##
##        if 1:
##            img = list_shift_img[fn] - shift
##            radius = 1
##            single_len = img.shape[0]
##            new_size = (2*radius + 1) * img.shape[0]
##            neighbour = np.zeros((new_size, new_size))
##            
##            from lib.cell2world import Hexa2Decimal, int2hex, coord_fn_from_cell_index
##            from itertools import product
##            from lib.boundaries import apply_gaussian
##
##            
##            m,n = fn[:17].split('_')
##
##            int_m = Hexa2Decimal(m)
##            int_n = Hexa2Decimal(n)
##            combi = np.array(list((product(range(-radius,radius+1), range(-radius,radius+1)))))
##            combi_global = combi + [int_m, int_n]
##
##            neigh_list = [coord_fn_from_cell_index(m,n,'')[1]+'.ply' for m,n in combi_global]
##
##            
##            for neigh, loc in zip(neigh_list,combi):
##                if neigh in list_shift_img.keys():
##                    a,b = (loc + radius) * single_len
##                    neighbour[a:a+single_len, b:b+single_len] =  list_shift_img[neigh] - shift
##
##            # nonvalue
##            nonvalue = -999.0
##
##            img = neighbour
##            plot_img(img)
##            img = np.nan_to_num(img)
##            filtered, boundbuffer, mask = apply_gaussian(img, 0, 0, nonvalue, 'linear')
##            boundbuffer = np.nan_to_num(boundbuffer)
##            plot_img(boundbuffer)
##            plot_img(filtered)
##            plot_img(mask)
##            plt.show()
    
    local_to_UTM(ground_filtering_out_dir, geo_ground_filtering_out_dir, shift, r, x_offset, y_offset)
