import os
import numpy as np
from tqdm import tqdm

if 1:
    from lib.cell2world import coord, search_neigh_tiles
    from lib.ply import read_bin_xyz_scale, read_bin_double
    from lib.ply import read_bin_xyzrid_scale, write_points_double, write_points_ddfi
    from lib.ground_filtering import ground_filter
    from lib.util import global2local, local2global
    from lib.adapt_height import generate_updates_mms

def create_fn2runid_index(pointcloud_path):
    # Create filename to runid index
    runid_index = dict()
    for fn in tqdm(os.listdir(pointcloud_path)):
        data_mms = read_bin_xyzrid_scale(pointcloud_path + fn, 12)
        data_runids = data_mms[:,4]
        runid_index[fn] = list(np.int0(np.unique(data_runids)))
    return runid_index

def create_runid2fn_index(fn2runid_index):
    # Generate revese from fn2runid_index
    runid2fn_index = dict()
    for fn in fn2runid_index:
        for runid in fn2runid_index[fn]:
            if runid in runid2fn_index:
                runid2fn_index[runid].append(fn)
            else:
                runid2fn_index[runid] = [fn]
    return runid2fn_index

def create_fn_runid_indexes(pointcloud_path, tmp_path):
    
    if "fn2runid_index.npy" not in os.listdir(tmp_path):
        # Create filename to runid index - run once
        fn2runid_index = create_fn2runid_index(pointcloud_path)
        np.save(tmp_path + "fn2runid_index.npy", [fn2runid_index,0])
    else:
        # When exit, load from file
        fn2runid_index = np.load(tmp_path + "fn2runid_index.npy",
                                 allow_pickle=True)[0]

    runid2fn_index = create_runid2fn_index(fn2runid_index)
    return fn2runid_index, runid2fn_index

def generate_mask_grunid(data_runids, grunid):
    # Get points based on grunids
    overall_mask = np.zeros(len(data_runids))
    
    for runid in grunid:
        mask = data_runids == runid
        overall_mask = overall_mask + np.int0(mask)
    return np.bool8(overall_mask)

def assign_unknown_cells(queried_fns, d_shifts_runid, d_shiftimg_runid, 
                        r, res_update, threshold_count = 140):
    # For the unknow nearby cell, calculate a height change base on the neigbours
    def merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z
    
    for fn,value in d_shifts_runid.iteritems():
        if value==None:
            d_shifts_runid[fn] = 0
        else:
            img = d_shiftimg_runid[fn]
            count = np.sum(~np.isnan(img))
            if count < threshold_count:
                d_shifts_runid[fn] = 0
                
    dict_updates = dict()
    
    for fn in queried_fns:
        
        neigh_list = search_neigh_tiles(fn)
        data_update, z = generate_updates_mms(d_shifts_runid, neigh_list, r, res_update)
        
        if type(z) == type(None):
            continue
        
        for f, v in zip(neigh_list, z):
            if f in dict_updates:
                dict_updates[f].append(v)
            else:
                dict_updates[f] = [v]
    
    dict_unknown_updates = dict()
    for k,v in dict_updates.iteritems():
        if len(set(v))==1 and len(v)>1:
            pass
        else:
            dict_unknown_updates[k] = np.median(v)
            #print(k, np.mean(v), v)
    
    return merge_two_dicts(d_shifts_runid, dict_unknown_updates)


def calc_height_diff_grunid(grunid, queried_fns, args, fn_shifts_runid, fn_shiftimg_runid, 
                            is_save=True):
    # For a given runid group, calculate the difference to the other runids
    from lib.reference import filter_by_reference
    from lib.shift import calculate_tile_runid_shift
    
    pts_dir, ref_dir, tmp_dir, out_dir, r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list = args
    
    dict_shifts_runid = dict()
    dict_shiftimg_runid = dict()
    
    for fn in tqdm(queried_fns):
        
        # Get origin from filename
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
            
        if fn.split('.')[0] + '_changedx.ply' in os.listdir(out_dir):
            # Load points
            data_mms = read_bin_xyz_scale(out_dir + fn.split('.')[0] + '_changedx.ply', 8)
            print('read', fn.split('.')[0] + '_changedx.ply')
        else:
            # Load mms
            data_mms = read_bin_xyzrid_scale(pts_dir + fn, 12)
            # Rough filter based on given DTM
            data_mms = filter_by_reference(data_mms, ref_dir + fn, 
                                           geoid, sigma_geoid)
            data_mms = data_mms[:,[0,1,2,4]] # Remove reflectance
            
        if len(data_mms) < 200:
            continue
        
        # Get xyz, relectance and runids
        data_mms = global2local(data_mms, mm, nn)
        
        # Get mask for select grunid
        mask = generate_mask_grunid(data_mms[:,3], grunid)
        data_mms_grunid = data_mms[mask]
        data_mms_others = data_mms[~mask]
        
        if len(data_mms_grunid) < 200:
            continue

        # Ground filter seperately
        reduced_grunid, _  = ground_filter(data_mms_grunid, radius, res_list, r, cleaning = [])
        reduced_others, _  = ground_filter(data_mms_others, radius, res_list, r, cleaning = [])
        
        shift, shift_img = calculate_tile_runid_shift(reduced_grunid, reduced_others, 
                                                      res_ref=1.0, reject_threshold=1.0, r=r)
        
        # If the point are too less, ignore the points and save others
        if shift==None:
            save_pts = data_mms_others
        else:
            save_pts = data_mms
            
        if is_save:
            write_points_ddfi(local2global(save_pts, mm, nn), 
                              out_dir + fn.split('.')[0] + '_changed.ply')
        
        dict_shifts_runid[fn] = shift
        dict_shiftimg_runid[fn] = shift_img
    
    if is_save: 
        np.save(tmp_dir + fn_shifts_runid, [dict_shifts_runid, 0])
        np.save(tmp_dir + fn_shiftimg_runid, [dict_shiftimg_runid, 0]) 
    
    return dict_shifts_runid, dict_shiftimg_runid

def gen_dtm_runid(grunid, queried_fns, d_shifts_runid, args, res_update):
    # Create update values for select grunid
    
    from lib.adapt_height import check_update_necessity, update_pointcloud
    
    pts_dir, ref_dir, tmp_dir, out_dir, r, x_offset, y_offset, geoid, sigma_geoid, radius, res_list = args
    
    for fn in tqdm(queried_fns):    
        
        print(fn)
        
        # Get origin from filename
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
        if fn.split('.')[0] + '_changed.ply' not in os.listdir(out_dir):
            continue
        
        if str(grunid[0]) + "_updated__" + fn in os.listdir(out_dir):
            continue
        
        # Load points
        data_mms = read_bin_xyz_scale(out_dir + fn.split('.')[0] + '_changed.ply', 8)
        data_mms = global2local(data_mms, mm, nn)
        
        # Get mask for select grunid
        mask = generate_mask_grunid(data_mms[:,3], grunid)
        data_mms_grunid = data_mms[mask]
        data_mms_others = data_mms[~mask]
        
        if len(data_mms_grunid)<200:
            print(fn, 'ignored because too less mms points')
            continue
        
        #data_mms_grunid = read_bin_double(out_dir + fn.split('.')[0] + '_grunid.ply', 7)
        #data_mms_others = read_bin_double(out_dir + fn.split('.')[0] + '_others.ply', 7)
        #data_mms_grunid = global2local(data_mms_grunid, mm, nn)
        #data_mms_others = global2local(data_mms_others, mm, nn)
                
        for nfn in search_neigh_tiles(fn):
            if nfn in d_shifts_runid:
                print(nfn, d_shifts_runid[nfn])
            else:
                print(nfn, "not found 0")
        
        if check_update_necessity(search_neigh_tiles(fn), d_shifts_runid):
        
            neigh_list = search_neigh_tiles(fn)
            data_update, z = generate_updates_mms(d_shifts_runid, neigh_list, r, res_update)
            
            # Only change selected runid
            mms_updated = update_pointcloud(data_mms_grunid, data_update, res_update)
            updated_pointcloud = np.vstack([mms_updated, data_mms_others])
            
            # Update the point clouds
            write_points_ddfi(local2global(updated_pointcloud.copy(), mm, nn), 
                              out_dir + fn.split('.')[0] + '_changedx.ply')
            
            reduced_pointcloud, _  = ground_filter(updated_pointcloud, radius, res_list, r) 
            identifier = "_updated__"               
        else:
            data_mms = np.vstack([data_mms_grunid, data_mms_others])
            reduced_pointcloud, _  = ground_filter(data_mms, radius, res_list, r)
            identifier = "_original_" 
            
        if len(reduced_pointcloud)>200: 
            write_points_ddfi(local2global(reduced_pointcloud.copy(), mm, nn), 
                              out_dir + str(grunid[0]) + identifier + fn)