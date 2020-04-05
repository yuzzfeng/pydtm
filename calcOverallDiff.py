import os
import numpy as np

from lib.diff import calc_diff_tqdm
from lib.report import generate_report

from environmentSetting import project_name, res_ref, res_update, nonvalue
from environmentSetting import gfilter_out_dir, correct_out_dir, out_path
from environmentSetting import gupdate_out_dir, ref_dir, tmp_dir
from environmentSetting import r, x_offset, y_offset, geoid, sigma_geoid
    
if 1:
    
    # Check if runid changed, replace the filtered by the updated ones
    if os.path.isdir(correct_out_dir):
        gfilter_dir = correct_out_dir
    else:
        gfilter_dir = gfilter_out_dir
    
    # Many of the gf files are processed
    if len(os.listdir(gupdate_out_dir)) > len(os.listdir(gfilter_dir))*2/3:
        list_pointcloud_filtered = os.listdir(gupdate_out_dir)
        mms_dir = gupdate_out_dir
    else:
        list_pointcloud_filtered = os.listdir(gfilter_dir)
        mms_dir = gfilter_dir
        
    args = mms_dir, ref_dir, res_ref, r, x_offset, y_offset, geoid, sigma_geoid

if 1:
    if project_name + "_shift.npy" not in os.listdir(tmp_dir):
        
        list_shift_value, list_shift_img, dict_shift_value = calc_diff_tqdm(list_pointcloud_filtered, args)
        print ('Shifts calculation finished.')
        
        # Save the dict for shift values
        np.save(tmp_dir + project_name + "_shift.npy",       [dict_shift_value, 0])   
        np.save(tmp_dir + project_name + "_shift_img.npy",   [list_shift_img, 0])    
        np.save(tmp_dir + project_name + "_shift_value.npy", [list_shift_value, 0])   
    
    else:
        
        dict_shift_value = np.load(tmp_dir + project_name + "_shift.npy", allow_pickle=True)[0]
        list_shift_img = np.load(tmp_dir + project_name + "_shift_img.npy", allow_pickle=True)[0]
        list_shift_value = np.load(tmp_dir + project_name + "_shift_value.npy", allow_pickle=True)[0]
    
    list_shift_value = np.array(list_shift_value)[np.array(list_shift_value)!=None]
    
    # Find the invalid tiles
    invalid = []
    for key,value in dict_shift_value.iteritems():
        if not value:
            invalid.append(key)
        else:
            if np.isnan(value):
                invalid.append(key)
    
    # Remove the invalid from dictionary
    for key in invalid:
        list_shift_img.pop(key)
        dict_shift_value.pop(key)

    shift = generate_report(list_shift_value, list_shift_img, dict_shift_value, 
                            out_path, r, x_offset,y_offset, res_ref)
    print 'report generated'

    
if 1:    
    raster_size = r/res_ref #30
    radius = 1
    
    from environmentSetting import merged_dir, final_dir, rest_dir
    from environmentSetting import ref_cut_dir, ref_update_dir
    from lib.util import check_and_create
    
    from lib.update import update_dtm
    from lib.produceDTM import local_to_UTM_tqdm, local_to_UTM_update_ref, local_to_UTM_rest_ref

    check_and_create(merged_dir)
    check_and_create(final_dir)
    check_and_create(rest_dir)
    
    list_pointcloud_ref = [fn for fn in os.listdir(ref_dir) if '.ply' in fn] 
   
    update_dtm(list_shift_img, raster_size, radius, ref_cut_dir, ref_update_dir,
               shift, res_ref, list_pointcloud_ref, ref_dir)
    print 'updates generated'

if 1:   
    #shift = 42.9317700613 # Hannover - 42.9316920581
    #shift = 43.5341477257 # Hildesheim - After reject outlier 
    #shift = 43.5346042674 # Hildesheim - Before reject outlier 

    # Process the mms and update of ref together 
    local_to_UTM_tqdm(mms_dir, dict_shift_value.keys(), merged_dir, 
                      ref_cut_dir, shift, r, x_offset, y_offset)

    # Process the update ref and combine the duplicated ones
    local_to_UTM_update_ref(ref_update_dir, final_dir, r, x_offset, y_offset)

    # Process the rest of tiles into UTM32 global coordinate system
    list_rest = list(set(list_pointcloud_ref) - set(os.listdir(final_dir)) - set(os.listdir(merged_dir)))
    local_to_UTM_rest_ref(list_rest, ref_dir, rest_dir, r, x_offset, y_offset)
        
    print 'Process finished'