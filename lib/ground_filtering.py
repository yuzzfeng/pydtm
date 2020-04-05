import numpy as np
from scipy import stats, ndimage

import matplotlib.pyplot as plt
from read import rasterize

########################################################################################
# Main loop
########################################################################################
def ground_filter(data_mms, radius, res_list, r, 
                  cleaning=[1.0, 0.5, 0.25, 0.1, 0.05], is_plot = False):

    save_img = None
    
    for res in res_list:
        
        reduced = []
        radius += 1

        d_mms1 = rasterize(data_mms, res, dim = 2)
        if len(data_mms) == 0:
            break

        threshold = 2 * res 
        img, d_mms1 = read_points_to_image(data_mms, d_mms1, res, r)
        
        if res == 0.1:
            save_img = img
        
        if res in cleaning: #[1.0, 0.5, 0.25, 0.1, 0.05]: #[0.25, 0.1]: 0.05 is necessary for reflection case
            d_mms1 = remove_isolated_seg(img, d_mms1, is_plot = is_plot)      
        
        indexs_res, new_data = update_points(data_mms, d_mms1, img, threshold, res, radius, r)

        reduced = data_mms[indexs_res] 
        data_mms = data_mms[new_data]

    return reduced, save_img


def check_sparsity(d_mms1):
    
    new_d_mms1 = dict()
    for key, value in d_mms1.iteritems():
        
        if len(value)>2:
            new_d_mms1[key] = value
    
    return new_d_mms1

def remove_isolated_seg(img, d_mms1, is_plot = False):
    """
    Find tiny segments and remove them    
    """
    from scipy import ndimage as ndi
    
    h,w = img.shape
    
    seg = np.float32(~np.isnan(img))
    seg = ndi.binary_fill_holes(seg)
    img_labels, n_labels = ndi.label(seg)
    
    if is_plot:
        plt.figure(); plt.imshow(img_labels)    
    
    collect_keys = []
    for i in np.unique(img_labels):
        #print(i, np.sum(img_labels==i))
        if np.sum(img_labels==i) < h*w*0.002: #25:
            xs, ys = np.where(img_labels==i)
            for i, j in zip(xs,ys):
                key = str(i) + '+' + str(j)
                collect_keys.append(key)
    
    for key in collect_keys:
        if key in d_mms1:
            d_mms1.pop(key)
        
    return d_mms1
        
        
########################################################################################
# Grid based method - hierarchical approach
########################################################################################

# read the rasterized points into images
def read_points_to_image(data_mms, d_mms, res, r):
    
    data_mms = data_mms[:,:3]
    
    img = np.zeros((int(r/res),int(r/res)))

    for key in d_mms.keys():
        x, y = [int(k) for k in key.split('+')]
        data = np.array(data_mms[d_mms[key],2])
        
        if x == int(r/res) or y == int(r/res): # Ignore the out of bound data
            print(x,y, int(r/res))
            continue

        if len(data) > 2:
            conf_int = stats.norm.interval(0.99, loc=np.median(data), scale=np.std(data))
            rest_ind = np.where((data >conf_int[0]) * (data <conf_int[1]))[0]

            if len(rest_ind) > 1:
                d_mms[key] = [d_mms[key][i] for i in rest_ind]
                index = d_mms[key][np.argmin(data[rest_ind])]

                img[x, y] = data_mms[index,2]
            else:
                img[x, y] = min(data)
        else:
            img[x, y] = min(data)
        
    np.place(img, img==0, None)
    return img, d_mms


# use the nerest n neighobours to detect anomoly
def neigh_filter(img, threshold_height_difference, radius, threshold_neighbour_count):
    
    size, size = img.shape
    mask = np.zeros((size,size))

    for i in xrange(size):
        for j in xrange(size):
            if ~np.isnan(img[i,j]):
                d = radius
                while 1:
                    start_i = i-d if i-d>0 else 0
                    start_j = j-d if j-d>0 else 0
                    end_i = i+d+1 if i+d+1 <size else size-1
                    end_j = j+d+1 if j+d+1 <size else size-1
                    L = img[start_i:end_i, start_j:end_j].flatten()
                    reduced_nan = L[~np.isnan(L)]
                    if len(reduced_nan) > threshold_neighbour_count or d > 50:
                        break
                    else:
                        d = d + 1
                if len(reduced_nan) > 0:
                    if abs(img[i,j] - np.median(reduced_nan)) > threshold_height_difference or len(reduced_nan) < threshold_neighbour_count: 
                        mask[i,j] = 1
    return mask


###################################################################################
## Use absolute height to filter the points

# Update the point cloud using threshold
def update_points(data_mms, d_mms1, img, threshold, res, radius, r):
    
    data_mms = data_mms[:,:3]
    
    mask =neigh_filter(img, 2*res/3, radius, 48)
##    mask =neigh_filter(img, 2*res/3, radius, 48)
##    mask =neigh_filter(img, res, radius, 48)
    
    index2d = np.array(np.where(mask == True)).T
    reduced = [str(x) + '+' + str(y) for x,y in index2d]
    keys = list(set(d_mms1.keys())- set(reduced))
    
    indexs_low_res = []
    indexs_new_data = []
    
    for key in keys:
        i, j = [int(k) for k in key.split('+')]
        
        if i == int(r/res) or j == int(r/res): # Ignore the out of bound data
            print(i, j, int(r/res))
            continue

        index_res = d_mms1[key][np.argmin(data_mms[d_mms1[key],2])]
        indexs_low_res.append(index_res)

        index = (data_mms[d_mms1[key],2] >= img[i,j]) * (data_mms[d_mms1[key],2] < img[i,j] + threshold)  
        indexs_new_data.extend(list(np.array(d_mms1[key])[index]))

##    plt.figure()
##    imgplot = plt.imshow(img)
##    plt.colorbar()
##    plt.figure()
##    imgplot = plt.imshow(mask)
##    plt.colorbar()
##    plt.show()

    return indexs_low_res, indexs_new_data


###################################################################################
## Use gradient to filter as parpendicular plane -- failed
###################################################################################
#
#def from_gradient_to_norm(img, res, size):
#
#    ## Cite: http://stackoverflow.com/questions/30993211/surface-normal-on-depth-image?lq=1
###    img = ndimage.grey_dilation(img, footprint=np.ones((5,5)))
#
#    gx = ndimage.sobel(img, 0, mode='nearest')/res
#    gy = ndimage.sobel(img, 1, mode='nearest')/res
#
#    ax = np.arctan(gx)
#    ay = np.arctan(gy)
#
#    dx = np.array([np.cos(ax), np.zeros((size,size)), - np.sin(ax)])
#    dy = np.array([np.zeros((size,size)), np.cos(ay), np.sin(ay)])
#
#    N = np.zeros((size,size,3))
#
#    for i in xrange(size):
#        for j in xrange(size):
#            N[i, j] = np.cross(dx[:,i,j],dy[:,i,j])
#    return N
#
#
#def update_by_gradient(data_mms, d_mms1, img, threshold, res, radius):
#
#    N =from_gradient_to_norm(img, res, len(img))
#
#    mask =neigh_filter(N[:,:,2], 0.4, radius, 8)
#
#    index2d = np.array(np.where(mask == True)).T
#    reduced = [str(x) + '+' + str(y) for x,y in index2d]
#    keys = list(set(d_mms1.keys())- set(reduced))
#
#    indexs_low_res = []
#    indexs_new_data = []
#    for key in keys:
#        i, j = [int(k) for k in key.split('+')]
#
#        points = data_mms[d_mms1[key]]
#        if ~np.isnan(N[i,j,2]):
#            n = N[i,j]
#            dis = sum(np.transpose(n*points))
#
#            index_res = d_mms1[key][np.argmin(dis)]
#            indexs_low_res.append(index_res)
#        
#            index = (dis < min(dis) + threshold)  
#            indexs_new_data.extend(list(np.array(d_mms1[key])[index]))
#            
###        else:
###            index_res = d_mms1[key][np.argmin(data_mms[d_mms1[key],2])]
###            indexs_low_res.append(index_res)
###            
###            index = (data_mms[d_mms1[key],2] >= img[i,j]) * (data_mms[d_mms1[key],2] < img[i,j] + threshold)
###            indexs_new_data.extend(list(np.array(d_mms1[key])[index]))
#
#
#    plt.figure()
#    imgplot = plt.imshow(img)
#    plt.colorbar()
#    plt.figure()
#    imgplot = plt.imshow(mask)
#    plt.colorbar()
#    plt.figure()
#    imgplot = plt.imshow(N[:,:,2])
#    plt.colorbar()
#    plt.show()
#
#    return indexs_low_res, indexs_new_data
#
#
#########################################################################################
## Slope based method -- failed
#########################################################################################
#
## Slope based method
#def slope_filter(data_mms, d_mms1, threshold_slope):
#
#    new_data = []
#    for key in d_mms1.keys():
#        X = np.array(data_mms[d_mms1[key]])
#        
#        for i in range(1, len(X) * 10):
#            i, j = np.random.randint(len(X), size=2)
#            dx,dy,dz = X[i]-X[j]
#            slope = np.sqrt(dz*dz / (dx*dx + dy*dy))
#            if slope > threshold_slope:
#                if dz > 0 :
#                    X = np.delete(X, (i), axis=0)
#                else:
#                    X = np.delete(X, (j), axis=0)
#        new_data.extend(X)
#    return np.array(new_data)
#
#def slope_filter2(data_mms, d_mms1, threshold_slope):
#
#    new_data = []
#    for key in d_mms1.keys():
#        X = np.array(data_mms[d_mms1[key]])
#
#        len_X = len(X)
#        for count in range(2):
#            i = np.random.randint(len(X), size=1) #np.argmax(X[:,2])
#            delta = X - X[i]
#            slope = np.array([np.sqrt( c*c /(a*a + b*b)) for a,b,c in delta])
#            index = np.where( (slope < 1) * ( delta[:,2] < 0 ) )[0]
#
#            X = X[index,:]
#
###            if len(X) < len_X:
###                len_X = len(X)
###            else:
###                break
#            
#        new_data.extend(X)
#    return np.array(new_data)