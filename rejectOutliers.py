import os
import numpy as np
from itertools import product
from scipy import interpolate 
from environmentSetting import *
from lib.checkRunids import check_and_create
from lib.read import rasterize
from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
from lib.cell2world import Hexa2Decimal, int2hex, coord_fn_from_cell_index
from lib.shift import  shiftvalue, reject_outliers

from lib.cell2world import coord, read_cellname

import matplotlib.pyplot as plt

is_plot_flag = False
nonvalue = -999.0
shift = 42.9317864988

ref_out_dir = out_path_old + 'ref\\'
ground_filtering_out_dir = out_path_old + 'aligned_GF\\'

ground_removed_out_dir = out_path + 'aligned_GR\\'
check_and_create(ground_removed_out_dir)

mms_dir = ground_filtering_out_dir
ref_dir = ref_out_dir
out_dir = ground_removed_out_dir

list_mms = os.listdir(mms_dir)

dict_shift_value = np.load('x.npy')[0]
#dict_shift_value_ = np.load('x.npy')[0]

res_update = 0.1

# Generate update grid in 0.1m
xnew = np.arange(0.05, 15, res_update)
ynew = np.arange(0.05, 15, res_update)    
xx, yy = np.meshgrid(xnew, ynew)
xxf = xx.flatten()
yyf = yy.flatten()

# Center of the neibouring 9 tiles
c9n = [-7.5, 7.5, 22.5]
coords9n = list(product(c9n, c9n))

### Center of the neibouring 25 tiles
#c25n = [-22.5, -7.5, 7.5, 22.5, 37.5]
#coords25n = list(product(c25n, c25n))
#
### Center of the neibouring 49 tiles
#c49n = [-37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5]
#coords49n = list(product(c49n, c49n))

coords = np.array(coords9n)
xn = coords[:,0]
yn = coords[:,1]

def column(matrix, i):
    return [row[i] for row in matrix]

def search_neigh_tiles(fn, radius = 1):
    # Search the nearest tiles based on grid and radius
    m,n = fn[:17].split('_')
    int_m = Hexa2Decimal(m)
    int_n = Hexa2Decimal(n)
    combi = np.array(list((product(range(-radius,radius+1), range(-radius,radius+1)))))
    combi_global = combi + [int_m, int_n]
    neigh_list = [coord_fn_from_cell_index(mx,nx,'')[1]+'.ply' for mx,nx in combi_global]
    return neigh_list


def interpolationNeigbours(xn, yn, z):
    
    lowleft = [0,1,3,4]
    lowright = [3,4,6,7]
    upperleft = [1,2,4,5]
    upperright = [4,5,7,8]

    img_lowleft = [0, 75, 0, 75]
    img_lowright = [0, 75, 75, 150]
    img_upperleft = [75, 150, 0, 75]
    img_upperright = [75, 150, 75, 150]
    
    zfinal = np.zeros((150,150))
    for pos, img in zip([lowleft, lowright, upperleft, upperright], 
                        [img_lowleft, img_lowright, img_upperleft, img_upperright]):

        f = interpolate.interp2d(xn[pos], yn[pos], z[pos], kind='linear')
        znew = f(xnew, ynew)
        
        sa, sb, sc, sd = img
        zfinal[sa:sb, sc:sd] = znew[sa:sb, sc:sd]
    
#    plt.figure()
#    plt.imshow(zfinal)
    
    return zfinal


def generate_updates_mms(dict_shift_value, neigh_list, 
                         is_plot = is_plot_flag):

#    z = np.array([dict_shift_value_[nfn] - shift if nfn in dict_shift_value_.keys() else 0 for nfn in neigh_list])
#    print(z)  

    z = np.array([dict_shift_value[nfn] - shift if nfn in dict_shift_value.keys() else 0 for nfn in neigh_list])
    print(z)    
    
    max_change = 0.8
    
    zz= filter(lambda a: a != 0 and abs(a) < max_change, z)    
    
    if len(zz) < len(z):
        mz = np.median(zz)
        
        for idx in np.where(np.array(z) == 0)[0]:
            z[idx] = mz
        
        for idx in np.where(np.abs(z) > max_change)[0]:
            z[idx] = mz            
            
        print('changed', z)
    
#    f = interpolate.interp2d(xn, yn, z, kind='linear')
#    znew = f(xnew, ynew)
#    plt.figure()
#    plt.imshow(znew)
    
    znew = interpolationNeigbours(xn, yn, z)    
    
    
    if is_plot:
        plt.figure()    
        plt.imshow(znew)
        plt.colorbar()
        plt.show()
        
    data_update = np.array(zip(xxf, yyf, znew.flatten()))
    return data_update

def calc_difference_mms_ref(data_mms, data_ref, res_diff, r,
                            is_plot = is_plot_flag):
    
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

def find_time_without_value(dict_shift_value, list_mms):
    height_needed = []
    for fn in list_mms:
        neighbours = search_neigh_tiles(fn)
        for nfn in neighbours:
            if nfn not in dict_shift_value or np.isnan(dict_shift_value[nfn]):
                height_needed.append(nfn)
    return list(set(height_needed))


def get_distance_with_padding(dict_shift_value, neighbours):
    return np.array([dict_shift_value[nfn] - shift if nfn in dict_shift_value.keys() else 0 for nfn in neighbours])

def get_distance_without_padding(dict_shift_value, neighbours):
    return np.array([dict_shift_value[nfn] for nfn in neighbours if nfn in dict_shift_value.keys()])

def insert_values_based_on_neigbours(dict_shift_value, height_needed):    
    ndict = dict()
    for fn in height_needed:
        neighbours = search_neigh_tiles(fn, radius = 1)
        z = get_distance_without_padding(dict_shift_value, neighbours) 
        if len(z)>0:
            if not np.isnan(np.median(z)):
                ndict[fn] = np.median(z)  
        
    for key in ndict.keys():
        dict_shift_value[key] = ndict[key]
    return dict_shift_value
        
def reject_single_change(dict_shift_value, thre = 0.6):
    # Delete outliers which heigher than threshold
    for fn in dict_shift_value.keys():
        dz = dict_shift_value[fn] - shift
        if abs(dz) > thre:
            print(fn, dz)
            dict_shift_value.pop(fn)
    return dict_shift_value        

def reject_local_change(dict_shift_value, list_mms, thre = 0.5):
    # Reject local changes over threshold
    tobedeleted = []
    for fn in list_mms:
        neighbours = search_neigh_tiles(fn, radius = 2)
        data = [[nfn, dict_shift_value[nfn] - shift] for nfn in neighbours if nfn in dict_shift_value.keys()]
        
        fns = np.array(column(data, 0))
        z = np.array(column(data, 1))       
        idx = abs(z - np.median(z)) > thre
        
        if (True) in idx:
#            print(list(fns[idx]), list(z[idx]))
#            print(list(z))
            tobedeleted.extend(fns[idx])
    tobedeleted = list(set(tobedeleted))
    
    print(len(tobedeleted))
    
    for fn in tobedeleted:
        dict_shift_value.pop(fn)
    return dict_shift_value,  len(tobedeleted)

if 1:
    
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])
    
    # Delete outliers which heigher than 0.8m
    dict_shift_value = reject_single_change(dict_shift_value)
    
    height_needed = find_time_without_value(dict_shift_value, list_mms)                
    while len(height_needed) > 0:    
        dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
        height_needed = find_time_without_value(dict_shift_value, list_mms)
    
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])

    dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
    
#    while num_change > 3:
#        height_needed = find_time_without_value(dict_shift_value, list_mms)                
#        while len(height_needed) > 0:    
#            dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
#            height_needed = find_time_without_value(dict_shift_value, list_mms)
#        dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
#        print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])

#fn = "00000025_ffffff02.ply"
#fn = "00000025_ffffff05.ply"
##fn = "0000002b_ffffff09.ply"
#fn = "0000002a_ffffff08.ply"
#fn = "00000027_ffffff05.ply"
#fn = "00000020_fffffec3.ply"
#for fn in ["00000025_ffffff02.ply", "00000025_ffffff05.ply", "0000002b_ffffff09.ply", 
#           "0000002a_ffffff08.ply", "00000027_ffffff05.ply", "00000020_fffffec3.ply"]:
#for fn in ["00000025_ffffff02.ply"]:
#    neighbours = search_neigh_tiles(fn)
#    for fn in neighbours:
#        if fn in list_mms:
#            print(fn)

#if 1:
for fn in list_mms:
    
            m,n = fn[:17].split('_')
            [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
            data_mms = read_bin(mms_dir + fn, 7)
            data_ref = read_bin(ref_dir + fn, 7)
            
            d_ref = rasterize(data_ref, res_ref, dim = 2)
            d_mms = rasterize(data_mms, res_ref, dim = 2)
            
            data_mms = np.array(data_mms)
            data_ref = np.array(data_ref)
            
            index_list = []
            
            # Correction for old dtm with bad interpolation
            for key, value in d_mms.iteritems():
                x, y = [int(k) for k in key.split('+')]
                
                z_mms = data_mms[value,2]
                z_ref = data_ref[d_ref[key], 2][0] + shift
                
                dz = z_mms -z_ref
                
                newIndex = list(np.array(value)[dz < 0.5])
                index_list.extend(newIndex)
            
            data_output = data_mms[sorted(index_list)]
            data_mms = data_output
            
            
        #    res_img = 1
        #    d_mms_big = rasterize(data_mms, res_img, dim = 2)
        #    d_ref_big = rasterize(data_ref, res_img, dim = 2)
            
        #    # calculate the difference
        #    #difference = nonvalue * np.ones((int(r/res_img),int(r/res_img)))
        #    difference = np.zeros((int(r/res_img),int(r/res_img)))
        #    
        #    for key, value in d_mms_big.iteritems():
        #        
        #        m,n = np.int0(key.split('+'))
        #        
        #        sub_mms = data_mms[value]
        #        sub_ref = data_ref[d_ref_big[key]] 
        #        
        #        d_mms_sub = rasterize(sub_mms, res_ref, dim = 2)
        #        d_ref_sub = rasterize(sub_ref, res_ref, dim = 2)
        #        
        #        diffs = []
        #        for subkey, subvalue in d_mms_sub.iteritems():
        #            z_mms = np.median(sub_mms[subvalue,2])
        #            z_ref = sub_ref[d_ref_sub[subkey], 2][0] + shift
        #            diffs.append(z_mms - z_ref)
        #        
        #        difference[m,n] = np.mean(diffs)
        #    
        #    plt.figure()
        #    plt.hist(difference, bins='auto')
        #        
        #    plt.figure()
        #    plt.imshow(difference)
            
            _ = calc_difference_mms_ref(data_mms, data_ref, 0.5, r)    
            
            neigh_list = search_neigh_tiles(fn)
            data_update = generate_updates_mms(dict_shift_value, neigh_list)
            
            d_update = rasterize(data_update, res_update, dim = 2)        
            d_mms = rasterize(data_mms, res_update, dim = 2)
            
            data_updated = []
            for key, value in d_mms.iteritems():
                sub_mms = data_mms[value]
                update_value = data_update[d_update[key][0],2]
                data_updated.extend(sub_mms - [0,0,update_value]) # Must be minus here
            data_updated = np.array(data_updated)   
            
            _ = calc_difference_mms_ref(data_updated, data_ref, 0.5, r)
            
            if len(data_output) != len(data_mms):
                print('updated for ', fn, len(data_output) / float(len(data_mms)))
            
            if len(data_output) > 0:
                #write_points(data_output, out_dir + '//_' + fn)
    
                write_points(data_updated, out_dir + '//' + fn) # local output
#                write_points_double(data_updated + [mm, nn, 0], out_dir + '//' + fn) # Global output