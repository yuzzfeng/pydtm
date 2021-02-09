import os
import numpy as np

from lib.ply import read_bin_xyz_scale, read_bin_double
from lib.cell2world import coord
from lib.util import global2local

def load_mms(fn, args):
    
    #mms_dir, ref_dir, res_ref, r, x_offset, y_offset, geoid, sigma_geoid  = args
    mms_dir, _, _, r, x_offset, y_offset = args
    
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
        
    data_mms = global2local(data_mms, mm, nn)
    return data_mms

























#from tool import mergecloud
#from read import read_bin_resmapling, rasterize, read_ascii_xyz
#
#from lib.ply import read_bin_xyz_scale, read_bin_xyzrid_scale, read_bin_xyzr_scale, read_bin_xyz_norm_scale, read_bin, read_bin_double

#geoid = 42.9664
#sigma_geoid = 0.4


#def load_aligned_data(fn, pointcloud_path, ref_out_dir, mm, nn, geoid, sigma_geoid):
#
#    data_ref = np.array(read_bin(ref_out_dir + fn, 7))
#
#    # Height range of the point cloud
#    height_max, height_min = max(data_ref[:,2]) + geoid + sigma_geoid, min(data_ref[:,2]) + geoid - sigma_geoid
#
#    del data_ref
#    
#    #data_mms = read_bin_xyz_scale(pointcloud_path + fn, 10)        # 40m Ricklingen
#    #data_mms = read_bin_xyz_norm_scale(pointcloud_path + fn, 13)   # Broken tiles
#    #data_mms = read_bin_xyzr_scale(pointcloud_path + fn, 11)        # 25m Hildesheim
#    data_mms = read_bin_xyzrid_scale(pointcloud_path + fn, 12)       # 25m Hildesheim
#
#    data_mms = np.array(data_mms)[:,0:3] - [mm,nn,0]
#
#    data_mms = data_mms[(data_mms[:,2] > height_min) * (data_mms[:,2]< height_max)]
#
#    return data_mms


#def load_data(fn_mms, r, in_dir_ref, res_ref):
#
#    cell_id = fn_mms.split('\\')[-1][:17]
#    fn_ref = in_dir_ref + '%s\\%s.ply'%(cell_id,cell_id)
#
#    data_ref,d_ref = read_ascii_xyz(fn_ref, delimiter=' ', skiprows=7, dtype=np.float, e=res_ref, dim=2)
#
#    # Height range of the point cloud
#    height_max, height_min = max(data_ref[:,2]) + geoid + sigma_geoid, min(data_ref[:,2]) + geoid - sigma_geoid
#
#    args = [height_min, height_max, 0.02]
#    
#    data_mms = read_bin_resmapling(fn_mms, 7, args)
#    return np.array(data_mms),data_ref,d_ref


#def load_data_both_scanners(fn_mms, r, in_dir_ref, res_ref):
#
#
#    cell_id = fn_mms.split('\\')[-1][:17]
#    fn_ref = in_dir_ref + '%s\\%s.ply'%(cell_id,cell_id)
#
#    data_ref,d_ref = read_ascii_xyz(fn_ref, delimiter=' ', skiprows=7, dtype=np.float, e=res_ref, dim=2)
#
#    # Height range of the point cloud
#    height_max, height_min = max(data_ref[:,2]) + geoid + sigma_geoid, min(data_ref[:,2]) + geoid - sigma_geoid
#
#    args = [height_min, height_max, 0.02]
#    
#    data_mms = read_bin_resmapling(fn_mms, 7, args)
#
#    runid = fn_mms.split('\\')[-1].split('.')[0].split('_')[-1]
#    runid_left = str(int(runid) - 1)
#    runid_right = str(int(runid) + 1)
#
#    fn_left = fn_mms.replace(runid, runid_left)
#    fn_right = fn_mms.replace(runid, runid_right)
#
#    fn_mms_scanner2 = None
#    if os.path.isfile(fn_left):
#        fn_mms_scanner2 = fn_left
#    if os.path.isfile(fn_right):
#        fn_mms_scanner2 = fn_right
#
#    if fn_mms_scanner2:
#        data_mms_scanner2 = read_bin_resmapling(fn_mms_scanner2, 7, args)
#
#        if len(data_mms)>0 and len(data_mms_scanner2)>0:
#            data_mms = mergecloud(np.array(data_mms), np.array(data_mms_scanner2))
#
#    return np.array(data_mms),data_ref,d_ref
