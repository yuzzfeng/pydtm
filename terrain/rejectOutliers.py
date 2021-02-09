import os
import numpy as np
from itertools import product
from scipy import interpolate 
import matplotlib.pyplot as plt

from lib.util import check_and_create
from lib.read import rasterize
from lib.ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
from lib.shift import shiftvalue, reject_outliers

from lib.cell2world import coord, read_cellname

from lib.adapt_height import generate_updates_mms
from lib.reference import calc_difference_mms_ref
from lib.neighbour import search_neigh_tiles, get_neighbour_matrix, get_distance_without_padding
        
def column(matrix, i):
    return [row[i] for row in matrix]


def find_time_without_value(dict_shift_value, list_mms):
    height_needed = []
    for fn in list_mms:
        neighbours = search_neigh_tiles(fn)
        for nfn in neighbours:
            if nfn not in dict_shift_value or np.isnan(dict_shift_value[nfn]):
                height_needed.append(nfn)
    return list(set(height_needed))

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


def shifts_cleaning(dict_shift_value, list_mms):
    # Correct height changes
    invalid = []
    for key,value in dict_shift_value.iteritems():
        if not value:
            invalid.append(key)
        else:
            if np.isnan(value):
                invalid.append(key)

    for key in invalid:
        dict_shift_value.pop(key)
        
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])
    
    # Delete outliers which heigher than 0.8m
    dict_shift_value = reject_single_change(dict_shift_value)

    plot_tiles_hdiff(dict_shift_value, args)

    height_needed = find_time_without_value(dict_shift_value, list_mms)                
    while len(height_needed) > 0:    
        dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
        height_needed = find_time_without_value(dict_shift_value, list_mms)
    
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])

    dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
    
    plot_tiles_hdiff(dict_shift_value, args)
    
    while num_change > 3:
        height_needed = find_time_without_value(dict_shift_value, list_mms)                
        while len(height_needed) > 0:    
            dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
            height_needed = find_time_without_value(dict_shift_value, list_mms)
        dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
        print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])
        
    plot_tiles_hdiff(dict_shift_value, args)
    
    return dict_shift_value


if 1:
    
    from lib.draw import plot_tiles_hdiff
    from environmentSetting import project_name, res_ref, res_update, nonvalue
    from environmentSetting import gfilter_out_dir, correct_out_dir 
    from environmentSetting import gupdate_out_dir, ref_dir, tmp_dir
    from environmentSetting import r, x_offset, y_offset, geoid, sigma_geoid
    
    shift = 43.53326697865561 # Report0

    is_plot_flag = True #False

    
    # If points changed based on runid h
    if os.path.isdir(correct_out_dir):
        mms_dir = correct_out_dir
    else:
        mms_dir = gfilter_out_dir
    list_mms = os.listdir(mms_dir)
    
    # tmp foldersfor intermediat result
    check_and_create(gupdate_out_dir)
    out_dir = gupdate_out_dir
    
    args = [mms_dir, ref_dir, tmp_dir, gupdate_out_dir, 
            r, x_offset, y_offset, geoid, sigma_geoid, res_ref, res_update]
    
    # Load the shift values of each tile according to the output of calc_diff
    dict_shift_value = np.load(tmp_dir + project_name + "_shift.npy",
                               allow_pickle=True)[0]
    dict_shift_value = shifts_cleaning(dict_shift_value, list_mms)
    
    
    
    
if 1:
    from tqdm import tqdm
    from lib.util import local2global
    from lib.load import load_mms
    args = mms_dir, ref_dir, res_ref, r, x_offset, y_offset, geoid, sigma_geoid
    
    
    for fn in tqdm(list_mms):
        
        #print(fn)
        
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
        data_ref = read_bin(ref_dir + fn, 7)
        data_mms = load_mms(fn,args)
        #data_mms = read_bin(mms_dir + fn, 7)
        #data_mms = read_bin_double(mms_dir + fn, 7)
        #data_mms = global2local(data_mms, mm, nn)
        
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
            
            dz = z_mms - z_ref
            
            newIndex = list(np.array(value)[dz < 0.5])
            index_list.extend(newIndex)
        
        data_output = data_mms[sorted(index_list)]
        data_mms = data_output
        
        _ = calc_difference_mms_ref(data_mms, data_ref, 0.5, r, shift)    
        
        neigh_list = search_neigh_tiles(fn)
        data_update, z = generate_updates_mms(dict_shift_value, neigh_list, 
                                              r, res_update, shift)
            
        d_update = rasterize(data_update, res_update, dim = 2)        
        d_mms = rasterize(data_mms, res_update, dim = 2)
        
        data_updated = []
        for key, value in d_mms.iteritems():
            sub_mms = data_mms[value]
            update_value = data_update[d_update[key][0],2]
            data_updated.extend(sub_mms - [0,0,update_value]) # Must be minus here
        data_updated = np.array(data_updated)   
        
        _ = calc_difference_mms_ref(data_updated, data_ref, 0.5, r, shift)
        
        if len(data_output) != len(data_mms):
            print('updated for ', fn, len(data_output) / float(len(data_mms)))
        
        if len(data_output) > 0:
            write_points_double(local2global(data_updated, mm, nn), 
                                out_dir + '//' + fn)
      
        
        
        
        
        
        
        
        
        
            
# Trash
            #    # Generate update grid in 0.1m
#    xnew = np.arange(0.05, r, res_update)
#    ynew = np.arange(0.05, r, res_update)    
#    xx, yy = np.meshgrid(xnew, ynew)
#    xxf = xx.flatten()
#    yyf = yy.flatten()
#    
#    coords9n = get_neighbour_matrix(9)
#    coords = np.array(coords9n)
#    xn = coords[:,0]
#    yn = coords[:,1]
            
            #    ref_out_dir = out_path + 'ref\\'
#    ground_filtering_out_dir = out_path + 'aligned_GF\\'
#    ground_removed_out_dir = out_path + 'aligned_GR\\'
#    check_and_create(ground_removed_out_dir)
#    mms_dir = ground_filtering_out_dir
#    ref_dir = ref_out_dir
#    out_dir = ground_removed_out_dir
            
            #fn = "00000025_ffffff02.ply"
#fn = "00000025_ffffff05.ply"
##fn = "0000002b_ffffff09.ply"
#fn = "0000002a_ffffff08.ply"
#fn = "00000027_ffffff05.ply"
#fn = "00000020_fffffec3.ply"
#for fn in ["00000025_ffffff02.ply", "00000025_ffffff05.ply", "0000002b_ffffff09.ply", 
#           "0000002a_ffffff08.ply", "00000027_ffffff05.ply", "00000020_fffffec3.ply"]:
#for fn in ["00000025_ffffff02.ply"]:
            
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