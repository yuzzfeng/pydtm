import os
import numpy as np

from tool import mergecloud, read_ply
from ply import write_points_double, read_bin
from cell2world import coord, read_cellname

#######################################################################
# Combine mms, update ref and rest ref together
# Author: Yu Feng
#######################################################################



def local_to_UTM(ground_filtering_out_dir, geo_ground_filtering_out_dir, z_offset, r, x_offset, y_offset):       
    list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)

    for fn in list_pointcloud_filtered:

        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)

        data = read_bin(ground_filtering_out_dir + fn, 7)
        data_mms = np.array(data) + [mm, nn, -z_offset]
        write_points_double(data_mms,  geo_ground_filtering_out_dir + fn)
        print fn 



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

    for cell in list_all_cells:
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
    for cell in list_ref_rest:
        mm, nn= read_cellname(cell)
        fn = in_dir_ref + cell + '\\' + cell + '.ply'
        data = read_ply(fn, 7)
        write_points_double(data + [mm,nn,0], path_final + cell + '.ply')
    print 'rest ref saved'
