# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:51:42 2019
@author: wenjun / Yu Feng

CloudCompare Batch for calculate omnivarious, surfacevariance, curvature features
    
parameters
----------
CCPath : cloudcompare executive file location
file_path : point cloud file location
    
Returns
----------
ply file with features
 
"""
import os
from lib.util import check_and_create

# data path and out path
out_path = 'D:/_data/_mms/20200325Hildesheim/'
mms_dir = out_path + 'aligned_Corr/'
mms_local_dir = out_path + 'aligned_Corr_local/'
check_and_create(mms_local_dir)

sfeat_dir = out_path + 'surface_feat/'
check_and_create(sfeat_dir)
list_mms = os.listdir(mms_dir)

if len(os.listdir(mms_local_dir))==0:
    
    # Move to local coordinate system
    from tqdm import tqdm
    from environmentSetting import r, x_offset, y_offset
    from lib.cell2world import coord
    from lib.load import load_mms
    from lib.ply import write_points_double
    for fn in tqdm(list_mms):
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        args = mms_dir, 0, 0, r, x_offset, y_offset
        data_mms = load_mms(fn, args)
        write_points_double(data_mms[:,:3], mms_local_dir + '//' + fn)

else:
    
    #absolute path of executive software cloudcompare v2.11
    CCPath = r"C:/Program Files/CloudCompare/"
    os.chdir(CCPath)

    # Feature names
    features = ["FEATURE OMNIVARIANCE", "FEATURE SURFACE_VARIATION", "CURV NORMAL_CHANGE"]
    radius = 0.25
    features_cmd = " ".join(["-"+feat+" "+str(radius) for feat in features])
    
    # Calculate features
    for fn in tqdm(os.listdir(mms_local_dir)):
        fn = mms_local_dir + fn  
        print(fn)
        openFile=os.system("CloudCompare -SILENT -O " + fn + " " + features_cmd + \
                           " -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_BE -NO_TIMESTAMP -SAVE_CLOUDS")