if 1:
    import os 
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    from ply import write_points, write_points_double, read_bin, read_bin_double, read_bin_xyz_norm_scale
    from cell2world import Hexa2Decimal, int2hex, coord_fn_from_cell_index
    from checkRunids import check_and_create
    from boundaries import apply_gaussian
    from read import rasterize

    from report import plot_img


def update_dtm(list_shift_img, raster_size, radius, ref_cut_dir, ref_update_dir,
               shift, res_ref, list_pointcloud_ref, ref_out_dir):

    # nonvalue
    nonvalue = -999.0

    check_and_create(ref_cut_dir)
    check_and_create(ref_update_dir)
    
    for fn in list_shift_img.keys():

        img = list_shift_img[fn] - shift
        single_len = img.shape[0]
        new_size = (2*radius + 1) * img.shape[0]
        neighbour = np.zeros((new_size, new_size))
        
        m,n = fn[:17].split('_')

        int_m = Hexa2Decimal(m)
        int_n = Hexa2Decimal(n)
        combi = np.array(list((product(range(-radius,radius+1), range(-radius,radius+1)))))
        combi_global = combi + [int_m, int_n]

        neigh_list = [coord_fn_from_cell_index(m,n,'')[1]+'.ply' for m,n in combi_global]

        not_in_list = []
        for neigh, loc in zip(neigh_list,combi):
            if neigh in list_shift_img.keys():
                a,b = (loc + radius) * single_len
                neighbour[a:a+single_len, b:b+single_len] =  list_shift_img[neigh] - shift
            else:
                if neigh in list_pointcloud_ref:
                    a,b = (loc + radius) * single_len
                    not_in_list.append([neigh,(a,b)])

        print fn, not_in_list

    
        img = neighbour
        
        img = np.nan_to_num(img)
        filtered, boundbuffer, mask = apply_gaussian(img, 0, 0, nonvalue, 'linear')
        boundbuffer = np.nan_to_num(boundbuffer)

        a,b = (np.array([0,0]) + radius) * single_len
        update = boundbuffer[a:a+single_len, b:b+single_len]
        upmask = mask[a:a+single_len, b:b+single_len]
        data_ref = read_bin(ref_out_dir + fn, 7)
        d_ref = rasterize(data_ref, res_ref, dim=2)
        data_ref = np.array(data_ref)

        raster_size = single_len
        data_output = []
        for i in xrange(0,raster_size):
            for j in xrange(0,raster_size):
                string = str.join('+',[str(i), str(j)])
                index = d_ref[string]
                if upmask[i,j] == 0:
                    data_output.append(data_ref[index][0] + [0,0,update[i,j]])

        write_points(data_output, ref_cut_dir + fn)
        

        for fn_not, (a,b) in not_in_list:
            
            update = boundbuffer[a:a+single_len, b:b+single_len]
            print np.sum(update)
            if abs(np.sum(update)) > 0.01:
                data_ref = read_bin(ref_out_dir + fn_not, 7)
                d_ref = rasterize(data_ref, res_ref, dim=2)

                data_ref = np.array(data_ref)

                data_output = []
                for i in xrange(0,raster_size):
                    for j in xrange(0,raster_size):
                        string = str.join('+',[str(i), str(j)])
                        index = d_ref[string]
                        data_output.append(data_ref[index][0] + [0,0,update[i,j]])

                check_and_create(ref_update_dir + fn_not)
                write_points(data_output, ref_update_dir + fn_not +'//' +fn_not + '_from_' + fn)

##                plot_img(update)
##        plot_img(boundbuffer)
##        plot_img(filtered)
##        plot_img(img)
##        plt.show()
