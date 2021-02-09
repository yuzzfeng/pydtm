import os
import numpy as np
from tqdm import tqdm

# Multiprocessing
from multiprocessing import Pool
from itertools import izip
from itertools import repeat

from read import rasterize
from tool import mergecloud, read_ply
from ply import write_points_double, read_bin
from cell2world import coord, read_cellname

from load import load_mms
#######################################################################
# Combine mms, update ref and rest ref together
# Author: Yu Feng
#######################################################################


def local_to_UTM_core(fn, args):

    mms_dir, out_dir, update_dir, list_update, r, x_offset, y_offset, z_offset = args

    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    
    args = mms_dir, 0, 0, r, x_offset, y_offset
    
    #data = read_bin(mms_dir + fn, 7)
    data = load_mms(fn, args)
    data_mms = np.array(data) + [mm, nn, -z_offset]

    if fn in list_update:
        data_update = np.array(read_bin(update_dir + fn, 7))
        if len(data_update)>0:
            data_mms = mergecloud(data_mms, data_update + [mm, nn, 0])

    write_points_double(data_mms,  out_dir + fn)

    return True

def local_to_UTM_star(a_b):
    return local_to_UTM_core(*a_b)

def local_to_UTM(in_dir, list_mms, out_dir, update_dir, z_offset, r, x_offset, y_offset):       
 
    list_update = os.listdir(update_dir)

    args = in_dir, out_dir, update_dir, list_update, r, x_offset, y_offset, z_offset
        
    ## 6 Cores
    p = Pool(6)
    result = p.map(local_to_UTM_star, zip(list_mms, repeat(args)))
    p.close()
    p.join()

    return result

def local_to_UTM_tqdm(in_dir, list_mms, out_dir, update_dir, z_offset, r, x_offset, y_offset):
    
    
    list_update = os.listdir(update_dir)

    args = in_dir, out_dir, update_dir, list_update, r, x_offset, y_offset, z_offset
    
    result = []
    for fn in tqdm(list_mms):
        res = local_to_UTM_core(fn, args)
        result.append(res)
        
    return result
#######################################################################

def local_to_UTM_rest_ref_core(fn, args):

    in_dir, out_dir, r, x_offset, y_offset = args

    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    
    new_pointcloud = np.array(read_bin(in_dir + fn, 7))
    write_points_double(np.array(new_pointcloud)+ [mm, nn, 0], out_dir + fn)

    return True
    
def local_to_UTM_rest_ref_star(a_b):
    return local_to_UTM_rest_ref_core(*a_b)

def local_to_UTM_rest_ref(list_rest, in_dir, out_dir, r, x_offset, y_offset):

    args = in_dir, out_dir, r, x_offset, y_offset

#    ## 6 Cores
#    p = Pool(6)
#    result = p.map(local_to_UTM_rest_ref_star, zip(list_rest, repeat(args)))
#    p.close()
#    p.join()

    result = []
    for fn in tqdm(list_rest):
        res = local_to_UTM_rest_ref_core(fn, args)
        result.append(res)
        
    return result


#######################################################################

def local_to_UTM_update_ref_core(fn, args):

    in_dir, out_dir, r, x_offset, y_offset = args

    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)

    list_duplicate = os.listdir(in_dir + fn)

    if len(list_duplicate) == 1:

        new_pointcloud = np.array(read_bin(in_dir + fn + '\\' + list_duplicate[0] ,7))
        write_points_double(np.array(new_pointcloud)+ [mm, nn, 0], out_dir + fn)

    else:

        pointcloud = np.array([0,0,0])
        for fn_under in list_duplicate:
            pointcloud = mergecloud(pointcloud, np.array(read_bin(in_dir + fn + '//' + fn_under, 7)))

        pointcloud = pointcloud[1:]
        d = rasterize(pointcloud, 0.5, dim=2)

        new_pointcloud = []
        for key in d.keys():
            new_pointcloud.append(np.mean(pointcloud[d[key]], axis = 0))
        
##            if np.std(pointcloud[d[key]][:,2])>0.1:
##                print fn, np.std(pointcloud[d[key]][:,2]), np.mean( pointcloud[d[key]], axis = 0)

        write_points_double(np.array(new_pointcloud)+ [mm, nn, 0], out_dir + fn)

    
def local_to_UTM_update_ref_star(a_b):
    return local_to_UTM_update_ref_core(*a_b)


def local_to_UTM_update_ref(in_dir, out_dir, r, x_offset, y_offset):
    
    list_ref_update = os.listdir(in_dir)

    args = in_dir, out_dir, r, x_offset, y_offset

#    ## 6 Cores
#    p = Pool(6)
#    result = p.map(local_to_UTM_update_ref_star, zip(list_ref_update, repeat(args)))
#    p.close()
#    p.join()
    
    result = []
    for fn in tqdm(list_ref_update):
        res = local_to_UTM_update_ref_core(fn, args)
        result.append(res)
        
    return result


    
    
#######################################################################

def final_dtm(path_final, path_mms, path_ref, in_dir_ref):

    list_mms = os.listdir(path_mms)
    list_ref = os.listdir(path_ref)

    d_mms = dict()
    for mms in list_mms:
        d_mms[mms[:17]]=mms

    d_ref = dict()
    for ref in list_ref:
        d_ref[ref[:17]]=ref

    # Add merged dtm, single mms or single ref into final dtm model
    list_all_cells = list(set(d_mms.keys() + d_ref.keys()))

    for cell in tqdm(list_all_cells):
        mm, nn= read_cellname(cell)
        data = np.array([0,0,0])
        
        if d_ref.has_key(cell):
            fn_ref = path_ref + d_ref[cell]
            data_ref = read_bin(fn_ref, 7)
            data = mergecloud(data, data_ref)

        if d_mms.has_key(cell):
            fn_mms = path_mms + d_mms[cell]
            data_mms = read_bin(fn_mms, 7)
            data = mergecloud(data, data_mms)

        write_points_double(data[1:] + [mm,nn,0], path_final + cell + '.ply')
        print cell
        
    print 'mms+ref saved'

    # Add the not processed ref dtm into the final dtm model
    list_all_ref = os.listdir(in_dir_ref)                           # Reference DTM DTM_2009 0.5m Resolusion
    list_ref_rest = list(set(list_all_ref) - set(list_all_cells))   # Difference with the processed cells
    for cell in tqdm(list_ref_rest):
        mm, nn= read_cellname(cell)
        fn = in_dir_ref + cell + '\\' + cell + '.ply'
        data = read_ply(fn, 7)
        write_points_double(data + [mm,nn,0], path_final + cell + '.ply')
    print 'rest ref saved'
