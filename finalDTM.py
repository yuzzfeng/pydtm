import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from lib.info import *

from lib.ply import write_points, write_points_double, read_bin_double, read_bin
from lib.read import rasterize

##from lib.read import read_bin_resmapling, rasterize, read_ascii_xyz
##from lib.shift import read_from_json, reject_outliers, shiftvalue

from lib.cell2world import coord_fn_from_cell_index
from lib.checkRunids import check_and_create, list_runid_read_both_scanner

from lib.tool import mergecloud
from lib.assemble import calc_mms, split_runid,cal_ref_duplicate, cal_ref_main, cal_ref_rest, copytoDst
from lib.produceDTM import final_dtm

##import struct


# Kaffeezimmer IKG
x_offset = 548495
y_offset = 5804458
r = 25


if __name__ == '__main__':

    in_dir_mms = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_mms\\'
    in_dir_update = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_update\\'

    out_dir_mms = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_mms_in_kracheln\\'
    out_dir_update = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_update_in_kracheln\\'

    final_dir_mms = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_mms_final\\'
    final_dir_update = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_update_final\\'

    path_ref = 'C:\\temp\\test\\'
    path_mms = 'C:\\temp\\test_mms\\'

##    #Final process for ref
##    split_runid(in_dir_update, out_dir_update, 0)
##    fn_full_cell = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_update_in_kracheln\\00000000_ffffff2d\\00000000_ffffff2d_638147048.ply'
##    data = read_bin(fn_full_cell, 7)
##    d = rasterize(data, 0.5, dim = 2)
##    full_list = d.keys()
##    del data, d
##        
##    cal_ref_duplicate(full_list, out_dir_update, path_ref)

##    # Optional save the big mask
##    list_all_cells = os.listdir(path_ref)
##    cal_ref_main(list_all_cells, path_ref, 'C:\\temp\\result_ref_main.ply')
##    cal_ref_rest(list_all_cells, list_ref, in_dir_ref, 'C:\\temp\\result_ref_rest.ply')


##    #Final process for mms
##    split_runid(in_dir_mms, out_dir_mms, 0)
##    split_runid(in_dir_mms, out_dir_mms, 98)
##    split_runid(in_dir_mms, out_dir_mms, 105)
##    calc_mms(path_mms, out_dir_mms, np.mean)

##    # Assemble the whole region 
##    path_final = 'C:\\temp\\once\\'
##    final_dtm(path_final, path_mms, path_ref, in_dir_ref)

