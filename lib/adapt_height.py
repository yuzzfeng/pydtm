import numpy as np
from tqdm import tqdm
from scipy import interpolate
from itertools import product
import matplotlib.pyplot as plt

def check_update_necessity(neigh_list, dict_shifts):
    # Check if update is necessary
    count = 0
    for nfn in neigh_list:
        if nfn in dict_shifts:
            if dict_shifts[nfn]!=0:
               count += 1 
    return count>0
    
def interpolationNeigbours(z, r, res_update):
    
    # Generate update grid in 0.1m
    xnew = np.arange(res_update/2, r, res_update)
    ynew = np.arange(res_update/2, r, res_update)    
    xx, yy = np.meshgrid(xnew, ynew)

    # Center of the neibouring 9 tiles
    c9n = [-r/2.0, r/2.0, 3*r/2.0]
    coords9n = list(product(c9n, c9n))
    coords = np.array(coords9n)
    xn = coords[:,0]
    yn = coords[:,1]
    
    width = int(r/res_update)
    
    lowleft = [0,1,3,4]
    lowright = [3,4,6,7]
    upperleft = [1,2,4,5]
    upperright = [4,5,7,8]

    img_lowleft = [0, width/2, 0, width/2]
    img_lowright = [0, width/2, width/2, width]
    img_upperleft = [width/2, width, 0, width/2]
    img_upperright = [width/2, width, width/2, width]
    
    zfinal = np.zeros((width,width))
    for pos, img in zip([lowleft, lowright, upperleft, upperright], 
                        [img_lowleft, img_lowright, img_upperleft, img_upperright]):

        f = interpolate.interp2d(xn[pos], yn[pos], z[pos], kind='linear')
        znew = f(xnew, ynew)
        
        sa, sb, sc, sd = img
        zfinal[sa:sb, sc:sd] = znew[sa:sb, sc:sd]
    
    return xx, yy, zfinal

def generate_updates_mms(dict_shifts, neigh_list, r, res_update, 
                         shift=0, is_plot = False):

    z = np.array([dict_shifts[nfn] - shift if nfn in dict_shifts.keys() else 0 for nfn in neigh_list])
    #print(z)    
    
    max_change = 0.8
    
    zz= filter(lambda a: a != 0 and abs(a) < max_change, z)    
    
    if len(zz) == 0 :
        return None, None
    
    
    if len(zz) < len(z):
        mz = np.median(zz)
        
        for idx in np.where(np.array(z) == 0)[0]:
            z[idx] = mz
        
        for idx in np.where(np.abs(z) > max_change)[0]:
            z[idx] = mz            
            
        #print('changed', z)
    
    xx, yy, znew = interpolationNeigbours(z, r, res_update)    
    
    if is_plot:
        plt.figure()    
        plt.imshow(znew)
        plt.colorbar()
        plt.show()
        
    data_update = np.array(zip(xx.flatten(), yy.flatten(), znew.flatten()))
    return data_update, z


def gen_updates4mms(list_fns, dict_shift_value, r, res_update, shift):
    
    from lib.neighbour import search_neigh_tiles
    
    # Tile Size in meter
    size = int(r/res_update)
    
    # Container of upadte images
    dict_update_img = dict()
    
    # Container of upadte points
    dict_update_pts = dict()
    
    # Collect updates by bilinear intepolation
    for fn in tqdm(list_fns):
        
        # 9 Neighbour tiles
        neigh_list = search_neigh_tiles(fn)
        data_update, z = generate_updates_mms(dict_shift_value, neigh_list, 
                                              r, res_update, shift)
        # Save image
        if type(data_update)!=type(None):
            update_img = data_update[:,2].reshape(size,size) 
            dict_update_img[fn] = update_img
            dict_update_pts[fn] = data_update

    return dict_update_img, dict_update_pts


# Remove because for runids
#def update_pointcloud(data_mms, data_update, res_update):
#    
#    # Create index according to updates resolution
#    d_update = rasterize(data_update, res_update, dim = 2)        
#    d_mms_runid = rasterize(data_mms, res_update, dim = 2)
#    
#    # Apply change for each cell regarding updates resolution
#    data_updated = []
#    for key, value in d_mms_runid.iteritems():
#        sub_mms = data_mms[value]
#        if key in d_update:
#            update_value = data_update[d_update[key][0],2]
#            sub_mms[:,:3] = sub_mms[:,:3] - [0,0,update_value]
#            data_updated.extend(sub_mms) # Must be minus here
#    return np.array(data_updated) 