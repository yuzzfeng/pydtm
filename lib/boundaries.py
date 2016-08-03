from scipy.ndimage import sobel, generic_gradient_magnitude, binary_opening, gaussian_gradient_magnitude, generate_binary_structure, binary_dilation
from scipy.ndimage import convolve, gaussian_filter
from scipy.interpolate import griddata
import numpy as np
import random

from tool import mergecloud, read_ascii_xyz 
from cell2world import read_fn


#########################################################################################
# Detect the boundaries using gradient and reduce the noise using opening
# Smooth the boundaries using gaussian kernal
# Author: Yu Feng
#########################################################################################

def reject_outliers(data, m):

    data = np.array(data).flatten()
    data = data[data!=0]
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    return data[s<m]

# return the boundaries cells
def detect_boundarymask(data, nonvalue):
    
    np.place(data, data == 0, nonvalue)
    gradient = generic_gradient_magnitude(data, sobel)
    gradient = np.where(gradient < 1, 0, 1)
    np.place(data, data == nonvalue, 0)

    return gradient

# Opening to reduce errors
def reduce_isolated_horizontal(data):
    
    mask_data = np.where(data != 0, 1, 0)
    mask_data = binary_opening(mask_data)
    
    return data * mask_data


# Apply gaussian kernal to smooth the hight jumps
def apply_gaussian(data, gauss_sigma, reject_threshold, nonvalue, inter_method):

##    np.place(data, data == nonvalue, 0)
    data = reduce_isolated_horizontal(data)  

##    inliers = reject_outliers(data, reject_threshold)
##    np.place(data, data > max(inliers), 0)
##    np.place(data, data < min(inliers), 0)

    gradient = detect_boundarymask(data, nonvalue)

    filtered = np.copy(data)
    mask = np.copy(data)
    np.place(mask, mask != 0, 1)

    np.place(data, gradient == 0, 0)
    
##    boundbuffer = gaussian_filter(data, gauss_sigma)
##    boundbuffer = boundbuffer * np.max(data) / np.max(boundbuffer) # Problem hier

##    boundbuffer = IDW(data, mask, nonvalue, 25)

    height, width = data.shape

    if height<1000 and width <1000:
        boundbuffer = filter_intepolation(data, mask, inter_method)
    else:
        boundbuffer = local_linear(data, mask, nonvalue, 200, inter_method)


    np.place(boundbuffer , mask == 1, 0)
    return filtered, boundbuffer, mask


def IDW(data, mask, nonvalue, d):

    struct1 = generate_binary_structure(2, 1)
    masknan = binary_dilation(mask, structure=struct1,iterations=10).astype(mask.dtype)
    
    img = data + nonvalue * (masknan-mask)
    np.place(img, img == nonvalue, None)

    m,n = img.shape
    img2 = np.copy(img)

    xx,yy = np.mgrid[0:2*d +1,0:2*d +1]
    dist= np.sqrt(pow(xx-d,2) + pow(yy-d,2))

    while 1:
        indexes = zip(*np.where(np.isnan(img2)))

        if len(indexes)==0:
            break

        for (i,j) in indexes:

            start_i = i-d if i-d>0 else 0
            start_j = j-d if j-d>0 else 0
            end_i = i+d+1 if i+d+1 <m else m
            end_j = j+d+1 if j+d+1 <n else n
            L = img2[start_i:end_i, start_j:end_j]

            if end_i -start_i==2*d +1 and end_j -start_j==2*d +1:
                local_dist = dist
            else:    
                list_i = np.array(range(0, 2*d+1))[(np.array(range(i-d, i+d+1))>=0) * (np.array(range(i-d, i+d+1))<m)]
                list_j = np.array(range(0, 2*d+1))[(np.array(range(j-d, j+d+1))>=0) * (np.array(range(j-d, j+d+1))<n)]
                local_dist = dist[min(list_i):max(list_i)+1, min(list_j):max(list_j)+1]

            idw_up = (L / local_dist).flatten()
            idw_down = (1 / local_dist).flatten()

            index_nan_inf = ~np.isnan(idw_up) * ~np.isinf(idw_down)

            if sum(index_nan_inf) > int(pow(2*d +1,2) * 0.3):
                img2[i,j] = sum(idw_up[index_nan_inf])/sum(idw_down[index_nan_inf])

    return img2



def local_linear(data, mask, nonvalue, d, inter_method):
    
    struct1 = generate_binary_structure(2, 1)
    masknan = binary_dilation(mask, structure=struct1,iterations=10).astype(mask.dtype)
    
    img = data + nonvalue * (masknan-mask)
    np.place(img, img == nonvalue, None)

    m,n = img.shape
    img2 = np.copy(img)

    count_indexes = 99999999999
    count_break = 0
    
    while 1:
        indexes = zip(*np.where(np.isnan(img2)))

        if len(indexes) == count_indexes:
            count_break += 1
        else:
            count_break = 0


        if len(indexes) < count_indexes:
            count_indexes = len(indexes)


        if count_break > 20:
            for (i,j) in indexes:
                img2[i,j] = 0

        if len(indexes)==0:
            break

        if len(indexes) > 0:
            (i,j) = indexes[random.randint(0, len(indexes)-1)]

            start_i, start_j, end_i, end_j = adapte_bound(i,j,m,n,d)
            L = img2[start_i:end_i, start_j:end_j]

            lm,ln = L.shape
            x, y = np.mgrid[0:lm, 0:ln]

            fx = x.flatten()
            fy = y.flatten()
            fz = L[fx,fy]

            mat = zip(fx,fy,fz)
            del fx,fy

            c = np.array(mat)[~np.isnan(fz)]
            grid_z0 = griddata(c[:,0:2], c[:,2], (x, y), method = inter_method)

            cstart_i, cstart_j, cend_i, cend_j = adapte_bound(i,j,m,n,d/2)

            image = np.zeros((m,n))
            image[start_i:end_i, start_j:end_j] = grid_z0
            grid = image[cstart_i:cend_i, cstart_j:cend_j]


            img2[cstart_i:cend_i, cstart_j:cend_j] = grid
            del image
            
    return img2     



def adapte_bound(i,j,m,n,d):

    start_i = i-d if i-d>0 else 0
    start_j = j-d if j-d>0 else 0
    end_i = i+d+1 if i+d+1 <m else m
    end_j = j+d+1 if j+d+1 <n else n

    return start_i, start_j, end_i, end_j


    

def filter_intepolation(data, mask, inter_method):

    m,n = data.shape
    x, y = np.mgrid[0:m, 0:n]

    fx = x.flatten()
    fy = y.flatten()
    fz = data[fx,fy]

    mat = zip(fx,fy,fz)
    del fx,fy

    struct1 = generate_binary_structure(2, 1)
    masknan = binary_dilation(mask, structure=struct1,iterations=10).astype(mask.dtype).flatten()

    c = np.array(mat)[(masknan==0) + (fz!=0)]
    
    grid_z0 = griddata(c[:,0:2], c[:,2], (x, y), method = inter_method)

    return grid_z0

def filter_intepolation_bigger_matrix(data, mask, inter_method):

    struct1 = generate_binary_structure(2, 1)
    masknan = binary_dilation(mask, structure=struct1,iterations=10).astype(mask.dtype)
    


def update_low_resolusion_DTM(fn_list, minM, minN, dnew, mask,
                              scale, in_dir_ref, res_ref, r, raster_size):

    cloud_ref = np.array([0,0,0])
    cloud_ref_org = np.array([0,0,0])
    
    for fn in fn_list:

        mm,nn = read_fn(fn)
        saved_mm = mm
        saved_nn = nn
        cell_id = fn[:17]

        # Reference pointcloud
        fn_ref = in_dir_ref + '%s\\%s.ply'%(cell_id,cell_id)
        data_ref,d_ref = read_ascii_xyz(fn_ref, delimiter=' ', skiprows=7, dtype=np.float, e=res_ref, dim=2)

        cloud_ref_org = mergecloud(cloud_ref_org, data_ref + [mm,nn,0])
        
        mm,nn = mm-minM, nn-minN
        upmask = mask[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale]
        update = dnew[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale]

        data_output = []
        for i in xrange(0,raster_size):
            for j in xrange(0,raster_size):
                string = str.join('+',[str(i), str(j)])
                index = d_ref[string]
                if upmask[i,j] == 0:
                    data_output.append(data_ref[index][0] + [0,0,update[i,j]])


        # Apply the global origin for UTM32
        if len(data_output)>0:
            data_output = np.array(data_output) + [saved_mm,saved_nn,0]
            cloud_ref = mergecloud(cloud_ref, data_output)

    return cloud_ref, cloud_ref_org
