###############################################################################
# This step is only necessary, when one of the runid are changed and save at
# given folder 
#
# Yu Feng, 25.03.2020
###############################################################################

import os
import numpy as np

from environmentSetting import res_update, radius, res_list
from environmentSetting import pts_dir, ref_dir, tmp_dir
from environmentSetting import r, x_offset, y_offset, geoid, sigma_geoid

from lib.runid import create_fn_runid_indexes, calc_height_diff_grunid
from lib.runid import gen_dtm_runid, assign_unknown_cells
from lib.draw import plot_tiles_hdiff, plot_tiles
from lib.util import flatten
    
def remove_nones(d_shifts_runid):
    new_d_shifts_runid = dict()
    for k,v in d_shifts_runid.iteritems():
        if v!=None:
            new_d_shifts_runid[k]=v
    return new_d_shifts_runid


def get_shifts(grunid, runid2fn_index, args):
    
    tmp_dir = args[2]
    runidstr = str(grunid[0])
    
    fn_shifts_runid   = "shifts_runid_"   + runidstr + ".npy"
    fn_shiftimg_runid = "shiftimg_runid_" + runidstr + ".npy"
    
    queried_fns = np.unique(list(flatten([runid2fn_index[runid] for runid in grunid])))
    plot_tiles(queried_fns, args)

    # Calculate height difference
    if fn_shifts_runid not in os.listdir(tmp_dir):
        d_shifts_runid, d_shiftimg_runid = calc_height_diff_grunid(grunid, queried_fns, args,
                                                                   fn_shifts_runid, fn_shiftimg_runid)
    else:
        d_shifts_runid = np.load(tmp_dir + fn_shifts_runid, allow_pickle=True)[0]
        d_shiftimg_runid = np.load(tmp_dir + fn_shiftimg_runid, allow_pickle=True)[0]
    
    d_shifts_runid = remove_nones(d_shifts_runid)
    plot_tiles_hdiff(d_shifts_runid, args)
        
    # For the unknow nearby cell, calculate a height change base on the neigbours
    d_shifts_runid = assign_unknown_cells(queried_fns, d_shifts_runid, d_shiftimg_runid, 
                                          r, res_update, threshold_count = 140)
    
    plot_tiles_hdiff(d_shifts_runid, args)
    
    return queried_fns, d_shifts_runid

if 1:
    
    
    #res_update = 0.1
    #r = 25
    #radius = 3
    #res_list = [1.0, 0.5, 0.25, 0.1, 0.05]
    
    ## IKG-PC
    #pts_dir = 'X:\\Products\\20190919_hildesheim_adjustment\\adjusted_cloud_ply\\'
    #fns_mms = os.listdir(pointcloud_path)
    
    # Feng-PC
    #pts_dir = "D:/_data/_mms/Hildesheim/"
    #pts_dir = "D:/_data/_mms/adjusted_cloud_ply/"
    #ref_dir = "D:/_data/_dgm/reference/"
    #tmp_dir = "tmp/" + project_name + '/'
    
    out_dir = "D:/_data/_mms/Hildesheim_out/"
    
    args = [pts_dir, ref_dir, tmp_dir, out_dir, 
            r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list]

    fn2runid_index, runid2fn_index = create_fn_runid_indexes(pts_dir, tmp_dir)

    grunid = (1150893024, 1150893025)
    queried_fns, d_shifts_runid = get_shifts(grunid, runid2fn_index, args)
    # Create update values for select grunid
    gen_dtm_runid(grunid, queried_fns, d_shifts_runid, args, res_update)
    
    grunid = (1150892210, 1150892211)
    queried_fns, d_shifts_runid = get_shifts(grunid, runid2fn_index, args)
    # Create update values for select grunid
    gen_dtm_runid(grunid, queried_fns, d_shifts_runid, args, res_update)