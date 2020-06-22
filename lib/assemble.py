import os
import numpy as np
from ply import write_points, write_points_double, read_bin, read_bin_double

from read import rasterize, read_ascii_xyz
from tool import mergecloud, read_ply

from cell2world import read_cellname, coord_fn_from_cell_index
#from checkRunids import check_and_create
from util import check_and_create

#from cell2world import read_fn, runid_size, max_runid_range
#from checkRunids import list_runid_read_both_scanner

import shutil
def copytoDst(in_path, fn, out_path):
    shutil.copy(in_path + '\\' + fn, out_path)


#####################################################################
# Split ref ply into cells
# Support multipy ply file and handle the borders automatically
#####################################################################


def split_ref_to_tiles(ref_path, ref_out_dir, r, x_offset, y_offset, res_ref):

    check_and_create(ref_out_dir)
    list_ref_ply = os.listdir(ref_path)
    print("Loading")
    num_point_per_kacheln = int(pow(r/res_ref, 2))
    imcomplete = []
    
    for fn_ref in list_ref_ply:
        data = read_bin_double(ref_path + fn_ref, 9)
        data = np.array(data) - [x_offset, y_offset, 0]
        d = rasterize(data, r, dim = 2)
        check_and_create(ref_out_dir + fn_ref)
        
        for cell_idx in d.keys():
            x,y = [int(idx) for idx in cell_idx.split('+')]
            ply_name, cell_name = coord_fn_from_cell_index(x,y,'')
            
            subdata = data[d[cell_idx]] - [x*r, y*r, 0]

            output_fn = ref_out_dir + fn_ref + '\\'+ cell_name + '.ply'
            write_points(subdata, output_fn)
            
            if len(d[cell_idx]) < num_point_per_kacheln:
                imcomplete.append(cell_name + '.ply')

            del subdata
        del data, d
    print("Load finished")
    list_ref_tiles = os.listdir(ref_out_dir)
    removed = []
    
    for i in xrange(len(list_ref_tiles)-1):

        list_left = ref_out_dir + list_ref_tiles[i]
        list_right = ref_out_dir + list_ref_tiles[i+1]
        
        left = os.listdir(list_left)
        right = os.listdir(list_right)

        intersect = set(left).intersection(right)
        for fn in intersect:
            data_left = read_bin(list_left + '\\' + fn, 7)
            data_right = read_bin(list_right + '\\' + fn, 7)
            data_new = mergecloud(data_left,data_right)

            if len(data_new) == num_point_per_kacheln:
                os.remove(list_left + '\\' + fn)
                os.remove(list_right + '\\' + fn)
                write_points(data_new, ref_out_dir + fn)
                removed.append(fn)

    intersection = list(set(imcomplete) - set(removed))
    check_and_create(ref_out_dir + 'reduced\\')
    
    for fn_list in list_ref_tiles:

        path = ref_out_dir + fn_list + '\\'
        list_path = os.listdir(path)

        for fn_file in list_path:
            if fn_file in intersection:
                shutil.move(path + fn_file, ref_out_dir + 'reduced\\' + fn_file)
            else:
                shutil.move(path + fn_file, ref_out_dir+fn_file)

    [os.rmdir(ref_out_dir + fn_list) for fn_list in list_ref_tiles]
    print 'Finish spliting'
    

#####################################################################
# Split runid into cells
#####################################################################

def split_runid(in_dir_update, out_dir_update, start_index):

##    # Kaffeezimmer IKG
##    x_offset = 548495
##    y_offset = 5804458
##    r = 25
    
    list_runids = os.listdir(in_dir_update)

    for fn_runid in list_runids[start_index:]:
        runid = fn_runid[:9]
        
        data = read_bin_double(in_dir_update + fn_runid, 7)
        data = np.array(data) - [x_offset, y_offset, 0]
        d = rasterize(data, r, dim = 2)

        for cell_idx in d.keys():
            x,y = [int(idx) for idx in cell_idx.split('+')]
            ply_name, cell_name = coord_fn_from_cell_index(x,y,runid)
            subdata = data[d[cell_idx]] - [x*r, y*r, 0]

            check_and_create(out_dir_update + cell_name)
            write_points(subdata, out_dir_update + cell_name + '\\'+ ply_name + '.ply')

            del subdata

        del data, d
        print fn_runid
        
#####################################################################
# Merge mms by mean
#####################################################################

#Final process for mms
def calc_mms(path_mms, out_dir_mms, method):

    list_cells = os.listdir(out_dir_mms)
    for fn_cell in list_cells:
        
        list_ply = os.listdir(out_dir_mms + fn_cell)

        print fn_cell, len(list_ply)

        if len(list_ply) == 1:
            fn_ply = list_ply[0]
            fn = out_dir_mms + fn_cell + '\\' + fn_ply
            data = read_bin(fn, 7)
            d = rasterize(data, 0.1, dim = 2)
            list_keys = d.keys()
            new_points = []
            data = np.array(data)
            for key in list_keys:
                list_xyz = data[d[key]]
                new_points.append(list(method(list_xyz, axis = 0)))

            write_points(new_points, path_mms + fn_cell +'.ply')
            del new_points
            
        else:
            list_ply = os.listdir(out_dir_mms + fn_cell)
            
            list_data = []
            list_d = []
            list_keys = []
            
            for fn_ply in list_ply:
                fn = out_dir_mms + fn_cell + '\\' + fn_ply
                
                data = read_bin(fn, 7)
                d = rasterize(data, 0.1, dim = 2)
                                
                list_data.append(np.array(data))
                list_d.append(d)
                list_keys.extend(d.keys())
                del data, d

            list_keys = list(set(list_keys))

            new_points = []
            for key in list_keys:
                list_xyz = np.array([0,0,0])
                for i in xrange(len(list_d)):
                    data = list_data[i]
                    d = list_d[i]
                    if d.has_key(key):
                        list_xyz = mergecloud(list_xyz, data[d[key]])

                list_xyz = list_xyz[1:]
                new_points.append(list(method(list_xyz, axis = 0)))

            write_points(new_points, path_mms + fn_cell +'.ply')
            del new_points


#####################################################################
# Merge update ref cells by median
#####################################################################

def cal_ref_duplicate(full_list, out_dir_update, path):

    list_cells = os.listdir(out_dir_update)
    for fn_cell in list_cells:
        list_ply = os.listdir(out_dir_update + fn_cell)
        if len(list_ply) == 1:
            copytoDst(out_dir_update + fn_cell, list_ply[0], path)
        else:
            list_count = []
            for fn_ply in list_ply:
                fn = out_dir_update + fn_cell + '\\' + fn_ply
                data = read_bin(fn, 7)
                list_count.append(len(data))

            if np.mean(list_count) == 2500:
                copytoDst(out_dir_update + fn_cell, list_ply[0], path)
            else:
                list_count = np.array(list_count)
                new_count = list_count[list_count!=2500]
                new_list = np.array(list_ply)[list_count!=2500]

                
                if len(new_count) == 1:
                    copytoDst(out_dir_update + fn_cell, new_list[0], path)
                else:

                    union = []
                    list_data = []
                    list_d = []
                    
                    for fn_ply in new_list:
                        fn = out_dir_update + fn_cell + '\\' + fn_ply
                        data = read_bin(fn, 7)
                        d = rasterize(data, 0.5, dim = 2)
                        
                        list_data.append(data)
                        list_d.append(d)

                        del_list = set(full_list) - set(d.keys())
                        union.extend(del_list)

                    res = list(set(full_list) - set(union))

                    new_points = []
                    for key in res:
                        list_z = []
                        x = 0
                        y = 0
                        for i in xrange(len(list_d)):
                            data = list_data[i]
                            d = list_d[i]
                            list_z.append(data[d[key][0]][2])
                            x,y = data[d[key][0]][0:2]

                        new_points.append([x,y,np.median(list_z)])


                    if len(res) > 0 :
                        write_points(new_points, path + fn_cell +'.ply')
                        
                    if np.std(list_z) > 0.5:
                        print fn_cell, len(res), np.mean(list_z), np.median(list_z), np.std(list_z)


#####################################################################
# Calculate the whole update for all the boundary areas
#####################################################################

def cal_ref_main(list_all_cells, path, out_fn):

    result_points = np.array([0,0,0])
    for cell in list_all_cells:
        x0,y0 = read_cellname(cell[:17])
        fn = path + cell
        data = read_bin(fn, 7)
        if len(data)>0:
            data = np.array(data) + [x0,y0,0]
            result_points = mergecloud(result_points, data)
            del data

    write_points_double(result_points[1:], out_fn)

    
#####################################################################
# Find the cells in reference data but not in the list of mms cells
#####################################################################

def cal_ref_rest(list_all_cells, list_ref, in_dir_ref, out_fn):

    list_all_cells = [cell[:17] for cell in list_all_cells]

    result_ref_points = np.array([0,0,0])
    for ref in list_ref:
        if ref not in list_all_cells:
            x0,y0 = read_cellname(ref)
            fn_ref = in_dir_ref + '%s\\%s.ply' % (ref,ref)
            data = read_ply(fn_ref, 7)
            data = np.array(data) + [x0,y0,0]
            result_ref_points = mergecloud(result_ref_points, data)
            del data

    write_points_double(result_ref_points[1:], out_fn) 
