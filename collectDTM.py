import os 
from environmentSetting import *

from lib.checkRunids import check_and_create
from lib.ply import write_points, write_points_double, read_bin, read_bin_double

out_dtm_dir = out_path + 'finalout\\'
check_and_create(out_dtm_dir)

pathA = geo_ground_filtering_out_dir
pathB = final_dir
pathC = rest_dir

listA = os.listdir(pathA)
listB = os.listdir(pathB)
listC = os.listdir(pathC)
# 
fns_dtm = ['ffffff9', 'ffffffa', 'ffffffb', 'ffffffc', 'ffffffd', 'ffffffe', 'fffffff', 
           '0000000', '0000001', '0000002', '0000003', '0000004', '0000005', '0000006']
#fns_dtm = ['0000002']
for fn_dtm in fns_dtm:
    data_dtm = []
    
    for flist, fpath in zip([listA, listB, listC], [pathA, pathB, pathC]):
        sublist = [fn for fn in flist if fn_dtm == fn[:7]]
        for fn in sublist:
            data_mms = read_bin_double(fpath + fn, 7)
            data_dtm.extend(data_mms)
    
    write_points_double(data_dtm, out_dtm_dir + fn_dtm + '.ply')