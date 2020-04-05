import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from tqdm import tqdm

from lib.ply import read_bin, read_bin_ddf, read_bin_xyzrid_scale
from lib.ply import write_points_double
from environmentSetting import *
from lib.ground_filtering import ground_filter
from lib.cell2world import coord, search_neigh_tiles


def update_tile_height(collect_data_mms, index, offset):
    if index!=None:
        # Update the index's layer with a fix offset at z direction
        collect_data_mms[index] = collect_data_mms[index] + np.array([0,0, offset])
    return np.vstack(collect_data_mms)

def detect_neigbour_runids(data_runids):
    # Find runids pairs, because they are normally good aligned
    
    runids = np.unique(data_runids)
    
    mask = np.hstack([np.array([False]), np.diff(runids) == 1])
    before = runids[mask] - 1
    after = runids[mask]
    
    # Add the rest without match pair
    pairs = zip(before, after)
    rest = [[runid] for runid in runids if runid not in before and runid not in after]
    
    return pairs + rest

def query_points_by_runids(data_mms, data_runids, grunid):
    # Get points based on grunids
    overall_mask = np.zeros(len(data_mms))
    
    for runid in grunid:
        mask = data_runids == runid
        overall_mask = overall_mask + np.int0(mask)
        
    data_mms_runid = data_mms[np.bool8(overall_mask)]
    return data_mms_runid

def search_index_offset(select_combi, select_offsets):
    
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    index =most_common(list(select_combi.flatten()))
    
    mask_confirm_index = [index in elem for elem in select_combi]
    
    select_combi = select_combi[mask_confirm_index]
    select_offsets = select_offsets[mask_confirm_index]
    
    order_check = [np.where(elem==index)[0][0] for elem in select_combi]
    mask_wrong_order = np.array(order_check)==0
    
    select_offsets[mask_wrong_order] = -select_offsets[mask_wrong_order]
    offset = np.mean(select_offsets)
    return index, offset
    
def find_index_to_move(grunids, save_imgs, is_plot=False):
    
    # Find the layer has significant height changes
    collect_mean = []
    collect_median = []
    combis = list(it.combinations(range(len(grunids)), 2))
    
    for i,j in combis:
        
        diff = save_imgs[i] - save_imgs[j]
        diff = diff.flatten()
        diff = diff[~np.isnan(diff)]
        collect_mean.append(np.mean(diff))
        collect_median.append(np.median(diff))
        
        if is_plot:
            plt.figure()
            plt.imshow(diff)
            plt.figure()
            _ = plt.hist(diff, bins='auto')
            print(np.mean(diff), np.std(diff))
    
    mask = np.abs(collect_mean) > 0.05
    if sum(mask)==0:
        return None, None
    
    select_combi = np.array(combis)[mask]
    select_offsets = np.array(collect_mean)[mask]
    
    index, offset = search_index_offset(select_combi, select_offsets)
    
    if abs(offset) < 0.05:
        return None, None
    
    #a, b = select_combi
    #index = list(set(a).intersection(set(b)))[0]    
    #offset = np.mean(select_offsets)
    return index, offset



    

def load_aligned_data_runid(fn, pointcloud_path, ref_out_dir):
    
    # Get origin from filename
    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)

    # Load reference dtm
    data_ref = np.array(read_bin(ref_out_dir + fn, 7))

    # Height range of the point cloud
    height_max, height_min = max(data_ref[:,2]) + geoid + sigma_geoid, min(data_ref[:,2]) + geoid - sigma_geoid
    
    # Load mms data
    #data_mms = read_bin_xyz_scale(pointcloud_path + fn, 10)         # 40m Ricklingen
    #data_mms = read_bin_xyz_norm_scale(pointcloud_path + fn, 13)    # Broken tiles
    #data_mms = read_bin_xyzr_scale(pointcloud_path + fn, 11)        # 25m Hildesheim
    data_mms = read_bin_xyzrid_scale(pointcloud_path + fn, 12)       # 25m Hildesheim
    data_mms = np.array(data_mms)
    
    # Remove points based on reference dtm
    data_mms = data_mms[(data_mms[:,2] > height_min) * (data_mms[:,2]< height_max)]

    # Get runids and points
    data_runids = data_mms[:,4]
    data_mms = data_mms[:,0:3] - [mm,nn,0]
    
    return data_mms, data_runids










#update_history = dict()
#
#for fn in ['0000002d_ffffffe2.ply']:#os.listdir(pointcloud_path):
#    
#    print("Processing...", fn)
#    
#    if fn in os.listdir(out_path):
#        continue
#    
#    # Get origin from filename
#    m,n = fn[:17].split('_')
#    [mm,nn] = coord(m, n, r, x_offset, y_offset)
#    
#    data_mms, data_runids = load_aligned_data_runid(fn, pointcloud_path, ref_out_dir)
#    
#    if len(data_mms) < 200:
#        continue
#    
#    grunids = detect_neigbour_runids(data_runids)
#    
#    collect_save_imgs = []
#    collect_data_mms = []
#    collect_grunids = []
#    
#    for grunid in grunids:
#        
#        # Query based on runid group
#        data_mms_runid = query_points_by_runids(data_mms, data_runids, grunid)
#        
#        # Filter ground for each runid group
#        reduced_runid, save_img  = ground_filter(data_mms_runid, radius, res_list, r)
#        
#        if type(save_img)!=type(None) and len(reduced_runid)>200:
#            collect_data_mms.append(data_mms_runid)
#            collect_grunids.append(grunid)
#            collect_save_imgs.append(save_img)
#            write_points_global(reduced_runid, out_path + str(grunid[0]) + fn)
#        
#    index, offset = find_index_to_move(collect_grunids, collect_save_imgs, 
#                                       is_plot=False)
#    
#    if index!=None:
#        print(index, offset, collect_grunids[index])
#        new_data_mms = update_tile_height(collect_data_mms, index, offset)
#        
#        reduced, _  = ground_filter(new_data_mms, radius, res_list, r)
#        update_history[fn] = (offset, collect_grunids[index])
#    else:
#        reduced, _  = ground_filter(data_mms, radius, res_list, r)
#        update_history[fn] = (0)
#    
#    if len(reduced)>200:
#        write_points_global(local2global(reduced, mm, nn), out_path + fn)
#    
#    #reduced_old, _  = ground_filter(data_mms, radius, res_list, r)
#    #write_points_global(local2global(reduced_old, mm, nn), out_path + 'prev' + fn)
#
#np.save("update_history.npy", [update_history, 0])
#
#update_history = np.load("update_history.npy", allow_pickle=True)[0]
#
#dict_shift_value = np.load('tmp/20190924_Hildesheim_shiftx.npy', allow_pickle=True)[0]
#
#if 1:
#    fn = '0000002d_ffffffe2.ply'
#    
#    shift = 43.5341477257 #(Hildesheim)
#    
#    # Get origin from filename
#    m,n = fn[:17].split('_')
#    [mm,nn] = coord(m, n, r, x_offset, y_offset)
#
#    # Load reference dtm
#    data_ref = np.array(read_bin(ref_out_dir + fn, 7))
#    data_mms = reduced
#    
#    from lib.read import rasterize
#    d_ref = rasterize(data_ref, res_ref, dim = 2)
#    d_mms = rasterize(data_mms, res_ref, dim = 2)
#    
#    index_list = []
#    
#    # Correction for old dtm with bad interpolation
#    for key, value in d_mms.iteritems():
#        x, y = [int(k) for k in key.split('+')]
#        
#        z_mms = data_mms[value,2]
#        z_ref = data_ref[d_ref[key], 2][0] + shift
#        
#        dz = z_mms -z_ref
#        
#        newIndex = list(np.array(value)[dz < 0.5])
#        index_list.extend(newIndex)
#    
#    data_output = data_mms[sorted(index_list)]
#    data_mms = data_output
#    
#    _ = calc_difference_mms_ref(data_mms, data_ref, 0.5, r)
#    
#    neigh_list = search_neigh_tiles(fn)
#    data_update = generate_updates_mms(dict_shift_value, neigh_list)

