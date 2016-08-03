import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# Multiprocessing
from multiprocessing import Pool
from itertools import izip
from itertools import repeat

from lib.info import *
from lib.ply import write_points, write_points_double
from lib.read import read_bin_resmapling, rasterize, read_bin, read_ascii_xyz
from lib.shift import read_from_json, reject_outliers, shiftvalue
from lib.cell2world import read_fn, runid_size, max_runid_range
from lib.checkRunids import check_and_create, list_runid_read_both_scanner

from lib.ground_filtering import ground_filter
from lib.tool import mergecloud
from lib.diff import grid_difference, down_sampling, grid_difference_filter_based
from lib.boundaries import apply_gaussian, update_low_resolusion_DTM
from lib.load import load_data, load_data_both_scanners


is_both_scanner = True

threshold_points_number = 200


def save_in_UTM(out_dir, runid, shift_value):
    
    data_runid = []
    min_runid = []
    fn_list = os.listdir(out_dir)
    
    for fn_mms in fn_list:
        
        data_mms = np.array(read_bin(out_dir + fn_mms, 7))
        min_runid.append(min(data_mms[:,2]))
        data_runid.append(data_mms)

    if len(data_runid)>0:
        
        rej = reject_outliers(min_runid, 10)

        result = []
        i = 0
        for fn_mms in fn_list:
            mm,nn = read_fn(fn_mms.split('\\')[-1])
            if len(min_runid) == 1:
                result.extend(data_runid[i] + [mm,nn,-shift_value])
            else:
                if rej[i] == True:
                    result.extend(data_runid[i] + [mm,nn,-shift_value])
            i +=1

        
        write_points_double(np.array(result), runid + '.ply')

    else:
        print 'no data in this runid'

def plot_fig(x,y,z):
    
    plt.figure()
    plt.pcolor(x, y, z)
    plt.colorbar()

def plot_img(img):
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()


def save_img(fn, img):
    plt.figure()
    plt.imshow(img, vmin=-0.3, vmax=0.3)
    plt.colorbar()
    plt.savefig(fn)
##    plt.imsave(fn, img)
    
def save_in_UTM3(out_dir, runid, shift_value, threshold_height_diff):

    if 1:
        data_runid = []
        min_runid = []

        res_output = 0.05
        method = np.min
        gauss_sigma, reject_threshold = 5, 20.

        fn_list = os.listdir(out_dir)
        M, N, minM, minN = runid_size(fn_list)
        
        
        count = 0
        for fn_mms in fn_list:
            
            data_mms = np.array(read_bin(out_dir + fn_mms, 7))
            min_runid.append(min(data_mms[:,2]))
            count = count + len(data_mms)
            
##            data_runid.append(data_mms)                        # Command this line if memory error

            del data_mms

    
    if count > 0:
        
        rej = reject_outliers(min_runid, 10)

        result = []
        diff_img = np.zeros((M * scale, N * scale))

        for i in xrange(len(fn_list)):

            fn_mms = fn_list[i]
            
            data_mms = np.array(read_bin(out_dir + fn_mms, 7)) -  [0,0,shift_value] # Uncommand this line if memory error
##            data_mms = data_runid[i] -  [0,0,shift_value]       # Command this line if memory error
            
            cell_id = fn_mms[:17]
            mm,nn = read_fn(fn_mms.split('\\')[-1])
            if len(min_runid) == 1:
                
                # Reference pointcloud and calculate the difference
                fn_ref = in_dir_ref + '%s\\%s.ply'%(cell_id,cell_id)
                diff, reduce_list = grid_difference_filter_based(fn_ref, res_ref, data_mms, -999, raster_size, threshold_height_diff)

                reduced_indexes = list(set(range(len(data_mms))) - set(reduce_list))
                result.extend(data_mms[reduced_indexes] + [mm,nn,0])
                del data_mms

                mm,nn = mm-minM, nn-minN
                diff_img[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale] = diff
                del diff

            else:
                if rej[i] == True:
                    # Reference pointcloud and calculate the difference
                    fn_ref = in_dir_ref + '%s\\%s.ply'%(cell_id,cell_id)
                    diff, reduce_list = grid_difference_filter_based(fn_ref, res_ref, data_mms, -999, raster_size, threshold_height_diff)

                    reduced_indexes = list(set(range(len(data_mms))) - set(reduce_list))
                    result.extend(data_mms[reduced_indexes] + [mm,nn,0])
                    del data_mms

                    mm,nn = mm-minM, nn-minN
                    diff_img[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale] = diff
                    del diff


        del data_runid
        result = np.array(result)
        index = reject_outliers(result[:,2], 20.0)
        write_points_double(result[index], out_dir_mms + '%s_mms.ply' % runid)
        del result

        np.place(diff_img, diff_img == nonvalue, 0)
        filtered, update, mask = apply_gaussian(diff_img, gauss_sigma, reject_threshold, nonvalue, 'linear')

        new_fn = max_runid_range(M, N, minM, minN, runid)
        cloud_ref, cloud_ref_org = update_low_resolusion_DTM(new_fn, minM, minN, update, mask,
                                                      scale, in_dir_ref, res_ref, r, raster_size)

        
        write_points_double(cloud_ref[1:], out_dir_update + '%s_update.ply' % runid)
        write_points_double(cloud_ref_org[1:], out_dir_ref + '%s_ref.ply' % runid)
        del cloud_ref, cloud_ref_org

##        plot_img(diff_img)
##        plot_img(update)
##        plt.show() 

        save_img(out_dir_diff_img + runid + '_diff.png', diff_img)
        save_img(out_dir_diff_img + runid + '_update.png', update)
        plt.close("all")
        
    else:
        print 'no data in this runid'


# Eleminate the objects and buildings on the terrain surface
def eliminate_object_grid_based(fn_mms, args):

    [out_dir_runid, runid, is_both_scanner] = args

    try:
        if is_both_scanner:
            data_mms,data_ref,d_ref = load_data_both_scanners(fn_mms, r, in_dir_ref, res_ref)
        else:
            data_mms,data_ref,d_ref = load_data(fn_mms, r, in_dir_ref)
    except:
        data_mms = []
    
    if len(data_mms)>200:
        
        radius = 3
        res_list = [1.0, 0.5, 0.25, 0.1, 0.05]

        reduced  = ground_filter(data_mms, radius, res_list, r)
        del data_mms

        if len(reduced)>200:
            path  = out_dir_runid + '%s\\'% runid
            check_and_create(path)
            write_points(reduced,  path + fn_mms.split('\\')[-1])
            shift_value, shift_img = shiftvalue(reduced, res_ref, data_ref, d_ref, 1., r)

            del reduced
        
            if shift_img != None:
                L = shift_img.flatten()
                L = L[~np.isnan(L)]
                
##                plot_img(shift_img - shift_value)
##                plt.show()

                return L
            else:
                return nonvalue
        else:
            return nonvalue
    else:
        return nonvalue
        
def eliminate_object_grid_based_star(a_b):
    return eliminate_object_grid_based(*a_b)


if __name__ == '__main__':

    
    
    #################################################################################

    # Output direction
    out_dir_mms = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_mms\\'
    out_dir_ref = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_ref\\'
    out_dir_runid = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_runid\\'
    out_dir_update = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_update\\'
    out_dir_diff_img = 'C:\\_EVUS_DGM\\Output_DTM\\DTM_diff_img\\'
    

##    out_dir = 'C:\\SVN\\feng\\curbDetection\\test\\'

    
    # Exceptions list
    failed_case = []
    no_ref_case = []

    #################################################################################

    
##    list_address = [
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_4_Kacheln - ply\\00000027_ffffff45\\00000027_ffffff45_638154672.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_8_Kacheln - ply\\00000015_ffffff63\\00000015_ffffff63_638183092.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_8_Kacheln - ply\\00000018_ffffff33\\00000018_ffffff33_638184023.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_8_Kacheln - ply\\0000001b_ffffff62\\0000001b_ffffff62_638180266.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_4_Kacheln - ply\\00000026_ffffff43\\00000026_ffffff43_638154673.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_8_Kacheln - ply\\0000001b_ffffff62\\0000001b_ffffff62_638180266.ply",
##    "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_3_Kacheln - ply\\ffffffe4_ffffff2f\\ffffffe4_ffffff2f_638147048.ply"
##    ]


    if is_both_scanner:
        list_runids = list_runid_read_both_scanner(dRunids.keys())
    else:
        list_runids = dRunids.keys()


    blacklist = ['638183777', '638173194', '638173197', '638134631', '638144595', '638171030']
    whitelist = ['638190451', '638189026']

##    if 1:
##        runid =  '638190451' # '638154672' #'638172627' #'638147048' 
    
    for runid in whitelist:

        if runid in blacklist:
            break
        
        t = time.time()
        list_mms = dRunids[runid]
        print runid,np.where(np.array(dRunids.keys())==runid)[0][0], len(list_mms)
        args = [out_dir_runid, runid, is_both_scanner]

##        # Delete all the data in tmp directory
##        [os.remove(out_dir + fn) for fn in os.listdir(out_dir)]
##        list_processed = os.listdir(out_dir)
##        list_mms = [fn for fn in list_mms if fn.split('\\')[-1] not in list_processed]

##        eliminate_object_grid_based(list_address[0], args)
        
##        result = []
##        ## One Core
##        for fn_mms in list_mms:
##            print fn_mms
##            result.append(eliminate_object_grid_based(fn_mms, args))


        ## 6 Cores
        p = Pool(6)
        result = p.map(eliminate_object_grid_based_star, zip(list_mms, repeat(args)))
        p.close()
        p.join()

       
        if 1:
            delta = []
            [delta.extend(i) for i in result if type(i) != float]
            del result
            delta = np.array(delta)

            ind = reject_outliers(delta, 1)
            alldelta = np.copy(delta)
            np.place(delta, ind==False, None)
            print np.median(delta[ind]), np.mean(delta[ind])
            shift_value = np.median(delta[ind])

##            shift_value = 42.897632852277482 #42.861879624586841 #42.896694785670235

            out_path  = out_dir_runid + '%s\\'% runid
            save_in_UTM3(out_path, runid, shift_value, 1.0)
            print time.time()-t

        if 1:
            plt.figure()
            plt.plot(range(len(alldelta)),alldelta,'bo')
            plt.plot(range(len(delta)),delta,'r+')
            plt.savefig(out_dir_diff_img + runid + '_shifts.png')
            plt.figure()
            n, bins, patches = plt.hist(delta[ind], 200, normed=1, facecolor='green', alpha=0.75)
            plt.grid(True)
            plt.savefig(out_dir_diff_img + runid + '_hist.png')
            plt.close("all")

##        plt.show()
