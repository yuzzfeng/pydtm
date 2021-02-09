#########################################################################################
# Apply transform on points
# Author: Yu Feng
#########################################################################################
import numpy as np
from tqdm import tqdm

from lib.read import rasterize
from lib.cell2world import coord
from lib.util import global2local, local2global
from lib.ply import read_bin_xyzrid_scale, read_bin_double, read_bin_xyz_scale, write_points_double

# Issue due to points at borders - remove the cell from indexes
def remove_border_pts(d_mms, res_update, r):
    
    # Find cell and pts on boarder
    cell_at_border = [k for k in d_mms.keys() if str(int(r/res_update)) in k]

    if len(cell_at_border) == 0:
        return d_mms
    
    print(cell_at_border)
    
    # Remove cell index
    for k in cell_at_border:
        d_mms.pop(k)

    return d_mms

# Apply height updates on single mms file
def apply_height_updates(data_mms, data_updates, res_update, r, global_shift = 0):
    
    # Indexing
    d_update = rasterize(data_updates, res_update, dim = 2)
    d_mms = rasterize(data_mms, res_update, dim = 2)
    
    # Points at the borders leads bugs - thus remove the cell-id
    d_mms = remove_border_pts(d_mms, res_update, r)
    
    # Initialize container
    data_updated = []
    
    for key, value in d_mms.iteritems():
        sub_mms = data_mms[value]
        update_value = data_updates[d_update[key][0],2]
        sub_mms[:,:3] = sub_mms[:,:3] - [0,0,update_value] - [0,0,global_shift]# Must be minus here
        data_updated.extend(sub_mms) 
        
    return np.array(data_updated)  


def load_corr_mms(fn, mms_dir, args):
    
    r, x_offset, y_offset = args
    
    # Get origin from filename
    m,n = fn[:17].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
    # Load points
    try: # Some of are ddd, and some are ddfi
        data_mms = read_bin_double(mms_dir + fn, 7)
    except:
        data_mms = read_bin_xyz_scale(mms_dir + fn, 8)
        data_mms = data_mms[:,:3]
        print('Changed loaded...', fn)
    
    # Trans to local system
    data_mms = global2local(data_mms, mm, nn)
    return data_mms


# Apply height updates on mms files in a folder
def apply_height_updates_folder(list_pts, pts_dir, dict_update_pts, res_ref, res_update,  
                                out_dir, args, global_shift = 0, is_xyz = False):

    r, x_offset, y_offset = args
    
    for fn in tqdm(list_pts):
        
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
        # Apply updates
        data_updates = dict_update_pts[fn]

        # Load mms original points
        if is_xyz:
            data_mms = load_corr_mms(fn, pts_dir, args)
        
        else:
            data_mms = read_bin_xyzrid_scale(pts_dir + fn, 12)
        
            # Trans to local system
            data_mms = global2local(data_mms, mm, nn)
        
        data_updated = apply_height_updates(data_mms, data_updates, res_update, r, global_shift)
        
        # Write updated points
        if len(data_updated) > 0:
            write_points_double(local2global(data_updated[:,:3], mm, nn), 
                                out_dir + '//' + fn)