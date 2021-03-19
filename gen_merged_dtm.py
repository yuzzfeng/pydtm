import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from environmentSetting import res_ref, r, x_offset, y_offset
from environmentSetting import tmp_dir, project_name, res_update
from lib.neighbour import search_neigh_tiles, get_distance_without_padding

from lib.draw import plot_tiles_hdiff

def column(matrix, i):
    return [row[i] for row in matrix]

def find_time_without_value(dict_shift_value, list_mms):
    height_needed = []
    for fn in list_mms:
        neighbours = search_neigh_tiles(fn)
        for nfn in neighbours:
            if nfn not in dict_shift_value or np.isnan(dict_shift_value[nfn]):
                height_needed.append(nfn)
    return list(set(height_needed))

def insert_values_based_on_neigbours(dict_shift_value, height_needed):    
    ndict = dict()
    for fn in height_needed:
        neighbours = search_neigh_tiles(fn, radius = 1)
        z = get_distance_without_padding(dict_shift_value, neighbours) 
        if len(z)>0:
            if not np.isnan(np.median(z)):
                ndict[fn] = np.median(z)  
        
    for key in ndict.keys():
        dict_shift_value[key] = ndict[key]
    return dict_shift_value
        
def reject_single_change(dict_shift_value, thre = 0.6):
    # Delete outliers which heigher than threshold
    for fn in dict_shift_value.keys():
        dz = dict_shift_value[fn] - shift
        if abs(dz) > thre:
            print(fn, dz)
            dict_shift_value.pop(fn)
    return dict_shift_value        

def reject_local_change(dict_shift_value, list_mms, thre = 0.5):
    # Reject local changes over threshold
    tobedeleted = []
    for fn in list_mms:
        neighbours = search_neigh_tiles(fn, radius = 2)
        data = [[nfn, dict_shift_value[nfn] - shift] for nfn in neighbours if nfn in dict_shift_value.keys()]
        
        fns = np.array(column(data, 0))
        z = np.array(column(data, 1))       
        idx = abs(z - np.median(z)) > thre
        
        if (True) in idx:
#            print(list(fns[idx]), list(z[idx]))
#            print(list(z))
            tobedeleted.extend(fns[idx])
    tobedeleted = list(set(tobedeleted))
    
    print(len(tobedeleted))
    
    for fn in tobedeleted:
        dict_shift_value.pop(fn)
    return dict_shift_value,  len(tobedeleted)


def shifts_cleaning(dict_shift_value, list_mms, args):
    # Correct height changes
    invalid = []
    for key,value in dict_shift_value.iteritems():
        if not value:
            invalid.append(key)
        else:
            if np.isnan(value):
                invalid.append(key)
            
    for key in invalid:
        dict_shift_value.pop(key)
        
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])
    
    # Delete outliers which heigher than 0.8m
    dict_shift_value = reject_single_change(dict_shift_value)

    plot_tiles_hdiff(dict_shift_value, args)

    height_needed = find_time_without_value(dict_shift_value, list_mms)                
    while len(height_needed) > 0:    
        dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
        height_needed = find_time_without_value(dict_shift_value, list_mms)
    
    print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])

    dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
    
    plot_tiles_hdiff(dict_shift_value, args)
    
    while num_change > 3:
        height_needed = find_time_without_value(dict_shift_value, list_mms)                
        while len(height_needed) > 0:    
            dict_shift_value = insert_values_based_on_neigbours(dict_shift_value, height_needed)
            height_needed = find_time_without_value(dict_shift_value, list_mms)
        dict_shift_value, num_change = reject_local_change(dict_shift_value, list_mms)
        print sum([np.isnan(dict_shift_value[key]) for key,value in dict_shift_value.iteritems()])
        
    plot_tiles_hdiff(dict_shift_value, args)
    
    return dict_shift_value

def find_nans_in_dict(dict_shift_value, list_shift_img):
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
        print('Removed - find_nans_in_dict', key)
        
    return dict_shift_value, list_shift_img

def calc_height_diff_entire(tmp_dir, project_name, mms_dir, ref_dir, is_after_dtm_alignment = False):
    
    if is_after_dtm_alignment:
        project_name = project_name + "_after_"
    else:
        project_name = project_name + "_before_"

    # Height difference 
    if project_name + "_shift.npy" not in os.listdir(tmp_dir): # calculate from data 
        
        from lib.diff import calc_diff_tqdm
        args = mms_dir, ref_dir, res_ref, r, x_offset, y_offset
        list_shift_value, list_shift_img, dict_shift_value = calc_diff_tqdm(list_mms, args)
        print ('Shifts calculation finished.')
        
        from lib.report import generate_report
        dict_shift_value, list_shift_img = find_nans_in_dict(dict_shift_value, list_shift_img)
        shift = generate_report(list_shift_value, list_shift_img, dict_shift_value, 
                                out_path + project_name + 'report\\', 
                                r, x_offset,y_offset, res_ref)
        print 'report generated'
    
        # Save the dict for shift values
        np.save(tmp_dir + project_name + "_shift.npy",       [dict_shift_value, 0])   
        np.save(tmp_dir + project_name + "_shift_img.npy",   [list_shift_img, 0])    
        np.save(tmp_dir + project_name + "_shift_value.npy", [list_shift_value, 0])   
        
    else: # Load existing height difference files
        
        dict_shift_value = np.load(tmp_dir + project_name + "_shift.npy", allow_pickle=True)[0]
        list_shift_img   = np.load(tmp_dir + project_name + "_shift_img.npy", allow_pickle=True)[0]
        list_shift_value = np.load(tmp_dir + project_name + "_shift_value.npy", allow_pickle=True)[0]
        
    return dict_shift_value, list_shift_img

if __name__ == "__main__":
    
    # Global shift due to geoid vs gps datum
    shift = 43.53326697865561 # Report0
    shift = 43.5341477257 # Hildesheim - After reject outlier 
    
    pts_dir = "D:/_data/_mms/adjusted_cloud_ply/"
    ref_dir = "D:/_data/_dgm/reference/"
    out_path = 'D:/_data/_mms/20200325Hildesheim/'
    gfilter_out_dir = out_path + 'aligned_GF/'
    correct_out_dir = out_path + 'aligned_Corr/'
    
    # Outputs
    merged_dir = out_path + 'aligned_a/'
    final_dir = out_path + 'aligned_b/'
    rest_dir = out_path + 'aligned_c/'
    gr_utm_dir = out_path + 'aligned_GR/'
    pts_utm_dir = out_path + 'aligned_MMS/'
    updates_dir = out_path + 'aligned_update_values/'
    ref_update_dir = out_path + 'aligned_ref_update/'
    ref_cut_dir = out_path + 'aligned_ref_update_cut/'

    from lib.util import check_and_create
    check_and_create(merged_dir)
    check_and_create(final_dir)
    check_and_create(rest_dir)
    check_and_create(gr_utm_dir)
    check_and_create(pts_utm_dir)
    check_and_create(updates_dir)
    check_and_create(ref_update_dir)
    check_and_create(ref_cut_dir)
    
    
    # If points corrected based on runid exist
    if os.path.isdir(correct_out_dir) and len(os.listdir(correct_out_dir)) >0 :
        mms_dir = correct_out_dir
    else:
        mms_dir = gfilter_out_dir
        
    list_mms = os.listdir(mms_dir)
    list_pts = os.listdir(pts_dir)
    
    # Calculate difference and report
    dict_shift_value, list_shift_img = calc_height_diff_entire(tmp_dir, project_name, mms_dir, ref_dir, 
                                                               is_after_dtm_alignment = False)
    
    ## Generate basic mask for data extent - terrain/mms
    from lib.visual_tools import gen_mask4mms
    gen_mask4mms("mask_gf.tiff", list_mms, (r, x_offset, y_offset))
    gen_mask4mms("mask_shifts.tiff", list_pts, (r, x_offset, y_offset), dict_shift_value)
    
    # Remove outliers - manualy selected by finding HL and LH - Project Hildesheim
    checked = ['00000014_fffffff6.ply', '0000000e_fffffff4.ply', "00000021_ffffffc0.ply", 
               "00000023_fffffff1.ply", '00000024_fffffff3.ply', '00000020_ffffffc0.ply',
               '0000001a_ffffffea.ply', '00000022_ffffffe7.ply', '0000001d_ffffffe5.ply', 
               "00000024_ffffffea.ply", '0000002f_ffffffe8.ply', '0000000c_fffffff2.ply',
               '00000032_ffffffe7.ply', '0000002e_ffffffe4.ply', '00000010_fffffff5.ply', 
               "00000029_ffffffe2.ply", "00000014_ffffffe2.ply", 'fffffff7_fffffff0.ply',
               'fffffffc_fffffff0.ply', "00000024_ffffffdc.ply", "00000018_ffffffdf.ply", 
               "0000002a_ffffffe9.ply", "0000002a_ffffffe7.ply"
               ]
    gen_mask4mms("mask.tiff", checked, (r, x_offset, y_offset)) # Draw a mask for the selected tiles
    
    for fn in checked:
        if fn in dict_shift_value:
            dict_shift_value.pop(fn)
            list_shift_img.pop(fn)
            print("Poped", fn)
    
    # Clean the noisy height updates
    dict_shift_value = shifts_cleaning(dict_shift_value, list_mms, (r, x_offset, y_offset))

    # Generate dense update values
    from lib.adapt_height import gen_updates4mms
    dict_update_img, dict_update_pts = gen_updates4mms(list_pts, dict_shift_value, r, res_update, shift)
    
    # Generate overview of the updates
    from lib.visual_tools import merge_updates4mms
    final_image = merge_updates4mms("test.tiff", dict_update_img, res_update, (r, x_offset, y_offset)) 
    
    """
    # Update MMS measurements
    """
    from lib.apply_transform import apply_height_updates_folder
    list_pts =  ["0000001e_ffffffee.ply", "0000001f_ffffffee.ply"] #os.listdir(pts_dir)
    apply_height_updates_folder(list_pts, pts_dir, dict_update_pts, res_ref, res_update, 
                                pts_utm_dir, (r, x_offset, y_offset), shift)
    
    
    """
    # Update MMS ground measurements
    """
    from lib.apply_transform import apply_height_updates_folder
    apply_height_updates_folder(list_mms, mms_dir, dict_update_pts, res_ref, res_update, 
                                gr_utm_dir, (r, x_offset, y_offset), shift, is_xyz = True)
    
    
    """    
    # Save updates values to tile ply
    """
    from lib.cell2world import coord
    from lib.ply import write_points_double
    from lib.util import global2local, local2global
    for fn, update_values in dict_update_pts.iteritems():
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        write_points_double(local2global(update_values[:,:3], mm, nn), 
                            updates_dir + '//' + fn)
    
    
    # Calculate difference and report
    dict_shift_value, list_shift_img = calc_height_diff_entire(tmp_dir, project_name, gr_utm_dir, ref_dir, 
                                                               is_after_dtm_alignment = True)
    
    """
    # Proc to final merged dtm
    """
    raster_size = r/res_ref #30
    radius = 1
    
    from lib.update import update_dtm
    from lib.produceDTM import local_to_UTM, local_to_UTM_update_ref, local_to_UTM_rest_ref
    
    shift = -0.00017232050548621046 # Hildesheim - Report-after
    
    list_pointcloud_ref = [fn for fn in os.listdir(ref_dir) if '.ply' in fn] 
    
    update_dtm(list_shift_img, raster_size, radius, ref_cut_dir, ref_update_dir,
               shift, res_ref, list_pointcloud_ref, ref_dir)
    print 'updates generated'

    # Process the mms and update of ref together 
    local_to_UTM(gr_utm_dir, dict_shift_value.keys(), merged_dir, 
                 ref_cut_dir, shift, r, x_offset, y_offset)

    # Process the update ref and combine the duplicated ones
    local_to_UTM_update_ref(ref_update_dir, final_dir, r, x_offset, y_offset)

    # Process the rest of tiles into UTM32 global coordinate system
    list_rest = list(set(list_pointcloud_ref) - set(os.listdir(final_dir)) - set(os.listdir(merged_dir)))
    local_to_UTM_rest_ref(list_rest, ref_dir, rest_dir, r, x_offset, y_offset)
        
    print 'Process finished'
    