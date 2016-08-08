import os 
import numpy as np

from lib.read import rasterize
##from lib.ground_filtering import ground_filter
from lib.cell2world import coord, coord_fn_from_cell_index
from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
from lib.shift import  shiftvalue, reject_outliers

from lib.checkRunids import check_and_create

from lib.load import load_aligned_data

from lib.assemble import split_ref_to_tiles
from lib.produceDTM import local_to_UTM, local_to_UTM_update_ref, local_to_UTM_rest_ref


from lib.cell2world import Hexa2Decimal, int2hex, coord_fn_from_cell_index
from lib.boundaries import apply_gaussian
from itertools import product

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

res_ref = 0.5

geoid = 42.9664
sigma_geoid = 0.4

x_offset = 548495 + 5
y_offset = 5804458 + 42
r = 15


ref_out_dir = 'C:\\temp\\aligned_ref\\'
##ref_path = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_ply\\1in4\\'
##split_ref_to_tiles(ref_path, ref_out_dir, r, x_offset, y_offset)
list_pointcloud_ref = os.listdir(ref_out_dir)

ref_update_dir = 'C:\\temp\\aligned_ref_update\\'
ref_cut_dir = 'C:\\temp\\aligned_ref_update_cut\\'

pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'

##ground_filtering_out_dir = 'C:\\temp\\aligned_GF\\'
ground_filtering_out_dir = 'C:\\temp\\aligned_GF_05082016\\'




list_pointcloud = os.listdir(pointcloud_path)
list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)


def calc_difference_mms_ref(fn, args):

    list_pointcloud_ref, ground_filtering_out_dir, ref_out_dir = args

    if fn in list_pointcloud_ref:
        data_mms = read_bin(ground_filtering_out_dir + fn, 7)
        data_ref = read_bin(ref_out_dir + fn, 7)

        d_ref = rasterize(data_ref, 0.5, dim = 2)
            
        shift_value, shift_img = shiftvalue(np.array(data_mms), res_ref, np.array(data_ref), d_ref, 1., r)
            
        return fn, shift_value, shift_img

    

if __name__ == "__main__":

##    print search_index(list_pointcloud, '0000004f_fffffec2.ply'), len(list_pointcloud)

    from lib.diff import calc_diff
    list_shift_value, list_shift_img = calc_diff(list_pointcloud_filtered, ground_filtering_out_dir, ref_out_dir, res_ref, r)

    if 1:
        ind = reject_outliers(list_shift_value, 5)

        list_shift_value = np.array(list_shift_value)
        hist, bins = np.histogram(list_shift_value[ind], bins=100)
        
        shift = np.median(list_shift_value[ind])
        plt.plot(bins[:-1], hist)

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

    # nonvalue
    nonvalue = -999.0
    raster_size = 30
    radius = 1

    
    for fn in list_shift_img.keys():

        img = list_shift_img[fn] - shift
        single_len = img.shape[0]
        new_size = (2*radius + 1) * img.shape[0]
        neighbour = np.zeros((new_size, new_size))
        
        m,n = fn[:17].split('_')

        int_m = Hexa2Decimal(m)
        int_n = Hexa2Decimal(n)
        combi = np.array(list((product(range(-radius,radius+1), range(-radius,radius+1)))))
        combi_global = combi + [int_m, int_n]

        neigh_list = [coord_fn_from_cell_index(m,n,'')[1]+'.ply' for m,n in combi_global]

        not_in_list = []
        for neigh, loc in zip(neigh_list,combi):
            if neigh in list_shift_img.keys():
                a,b = (loc + radius) * single_len
                neighbour[a:a+single_len, b:b+single_len] =  list_shift_img[neigh] - shift
            else:
                if neigh in list_pointcloud_ref:
                    a,b = (loc + radius) * single_len
                    not_in_list.append([neigh,(a,b)])

        print fn, not_in_list

                
        

        img = neighbour
        
        img = np.nan_to_num(img)
        filtered, boundbuffer, mask = apply_gaussian(img, 0, 0, nonvalue, 'linear')
        boundbuffer = np.nan_to_num(boundbuffer)

        a,b = (np.array([0,0]) + radius) * single_len
        update = boundbuffer[a:a+single_len, b:b+single_len]
        upmask = mask[a:a+single_len, b:b+single_len]
        data_ref = read_bin(ref_out_dir + fn, 7)
        d_ref = rasterize(data_ref, res_ref, dim=2)
        data_ref = np.array(data_ref)

        raster_size = single_len
        data_output = []
        for i in xrange(0,raster_size):
            for j in xrange(0,raster_size):
                string = str.join('+',[str(i), str(j)])
                index = d_ref[string]
                if upmask[i,j] == 0:
                    data_output.append(data_ref[index][0] + [0,0,update[i,j]])

        write_points(data_output, ref_cut_dir + fn)
        

        for fn_not, (a,b) in not_in_list:
            
            update = boundbuffer[a:a+single_len, b:b+single_len]
            print np.sum(update)
            if abs(np.sum(update)) > 0.01:
                data_ref = read_bin(ref_out_dir + fn_not, 7)
                d_ref = rasterize(data_ref, res_ref, dim=2)

                data_ref = np.array(data_ref)

                data_output = []
                for i in xrange(0,raster_size):
                    for j in xrange(0,raster_size):
                        string = str.join('+',[str(i), str(j)])
                        index = d_ref[string]
                        data_output.append(data_ref[index][0] + [0,0,update[i,j]])

                check_and_create(ref_update_dir + fn_not)
                write_points(data_output, ref_update_dir + fn_not +'//' +fn_not + '_from_' + fn)

##                plot_img(update)
##        plot_img(boundbuffer)
##        plot_img(filtered)
##        plot_img(img)
##        plt.show()



    
##    shift = 42.9316920581
    geo_ground_filtering_out_dir = 'C:\\temp\\aligned_a\\'
    final_dir = 'C:\\temp\\aligned_b\\'
    rest_dir = 'C:\\temp\\aligned_c\\'

    # Process the mms and update of ref together 
    local_to_UTM(ground_filtering_out_dir, geo_ground_filtering_out_dir, ref_cut_dir, shift, r, x_offset, y_offset)

    # Process the update ref and combine the duplicated ones
    local_to_UTM_update_ref(ref_update_dir, final_dir, r, x_offset, y_offset)

    # Process the rest of tiles into UTM32 global coordinate system
    list_rest = list(set(list_pointcloud_ref) - set(os.listdir(final_dir)) - set(os.listdir(geo_ground_filtering_out_dir)))
    local_to_UTM_rest_ref(list_rest, ref_out_dir, rest_dir, r, x_offset, y_offset)

    print 'Process finished'
