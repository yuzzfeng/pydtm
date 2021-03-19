# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:13:40 2019
@author: wenjun
test on master thesis data
"""
import numpy as np
import os
import struct
from plyfile import PlyData
import math
import pandas as pd
import copy
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations_with_replacement
###############################################################################
##Read file and shift to gloabl coordinate system
###############################################################################

def read_all_filename(Upath):
    return os.listdir(Upath)
    
    
def load_plydata(address):
    #plydata = PlyData.read(address)
    _data = PlyData.read(address).elements[0].data
    if len(_data[1])==3:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z']                    
                                ),axis=1)


    if len(_data[1])==9:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'],
                                _data['nx'],_data['ny'],_data['nz'],
                                _data['scalar_Omnivariance_(0.25)'],
                                _data['scalar_Surface_variation_(0.25)'],
                                _data['scalar_Normal_change_rate_(0.25)'],                               
                                ),axis=1)
    if len(_data[1])==11:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'],
                                _data['nx'],_data['ny'],_data['nz'],
                                _data['scalar_Omnivariance_(0.25)'],
                                _data['scalar_Surface_variation_(0.25)'],
                                _data['scalar_Normal_change_rate_(0.25)'],                               
                                _data['scalar_std_ominivarious'],
                                _data['scalar_std_nz']
                                ),axis=1)

    if len(_data[1])==29:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'],
                                _data['nx'],_data['ny'],_data['nz'],
                                _data['scalar_sigma'],
                                _data['scalar_sigma_plane'],
                                _data['scalar_Omnivariance_(0.25)'],
                                _data['scalar_Surface_variation_(0.25)'],
                                _data['scalar_Normal_change_rate_(0.25)'],                               
                                _data['scalar_std_ominivarious'],
                                _data['scalar_std_nz'],
                                _data['scalar_dist_histogram_bin1'],
                                _data['scalar_dist_histogram_bin2'],
                                _data['scalar_dist_histogram_bin3'],
                                _data['scalar_dist_histogram_bin4'],
                                _data['scalar_dist_histogram_bin5'],
                                _data['scalar_dist_histogram_bin6'],
                                _data['scalar_dist_histogram_bin7'],
                                _data['scalar_dist_histogram_bin8'],
                                _data['scalar_dist_histogram_bin9'],
                                _data['scalar_dist_histogram_bin10'],
                                _data['scalar_dist_histogram_bin11'],
                                _data['scalar_dist_histogram_bin12'],
                                _data['scalar_dist_histogram_bin13'],
                                _data['scalar_dist_histogram_bin14'],
                                _data['scalar_dist_histogram_bin15'],
                                _data['scalar_dist_histogram_bin16'],
                                ),axis=1)



    if len(_data[1])==5:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'],_data['reflectance'],_data['run-id']), axis=1)
    return plane_tango


# transfer GF to global coordinate system
def GF_to_global(name_of_ply,origin,cellsize):
    transfer_coords =  name_of_ply.split('.') [0]
    # get index
    x = transfer_coords.split('_')[0]
    y = transfer_coords.split('_')[1]
    
    #xs = int(x,16)
    '''
    #inverse 16
    y = y[::-1]
    #去16进制化
    neghex_pack = binascii.unhexlify(y)
    #解析
    ys = struct.unpack('i',neghex_pack)[0]
    '''
    # works for python2
    #ys = struct.unpack('>i',y.encode('hex'))[0]
    xs = struct.unpack('>i',bytes.fromhex(x))[0]
    ys = struct.unpack('>i',bytes.fromhex(y))[0]
    
    coordinate = [origin[0]+cellsize*xs,origin[1]+cellsize*ys,0]
    return np.array(coordinate)

# get the surrounding 8 tiles
def Neighbor_tiles(name_of_ply):
    transfer_coords =  name_of_ply.split('.') [0]
    # get index
    x = transfer_coords.split('_')[0]
    y = transfer_coords.split('_')[1]
    
    #xs = int(x,16)
    '''
    #inverse 16
    y = y[::-1]
    #去16进制化
    neghex_pack = binascii.unhexlify(y)
    #解析
    ys = struct.unpack('i',neghex_pack)[0]
    '''
    # works for python2
    #ys = struct.unpack('>i',y.encode('hex'))[0]
    xs = struct.unpack('>i',bytes.fromhex(x))[0]
    ys = struct.unpack('>i',bytes.fromhex(y))[0]
    
    neighbors =[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    s =[]
    for i in neighbors:
        a = [xs+i[0],ys+i[1]]
        s.append( "%08x_%08x.ply" % (int(a[0]) & 0xffffffff, int(a[1]) & 0xffffffff))

    return list(s)

###############################################################################
    ##create voxel of data
###############################################################################
floor= math.floor
join = str.join
# insert and update keys and values
def gen_str(coords, i):

    num = [str(int(floor(float(coord)/eps))) for coord in coords] #generate key
    string = join('+',num)#voxel key
    
   # if d.has_key(string) == True:# python3 delete dict.has_key() use in intead
   # if key exist, append point id, if key not exist create key and add point id 
    if string in d:
        app = d[string].append # This is a method
        app(i)
    else:    
        d[string] = [i]


# iterate each line in txt or ply data
# return point id of each voxel, e is the voxel size
def rasterize(data, e, dim = 2):

    global eps, d
    eps = e
    d = dict()
    i = 0

    if dim == 2:
        for point in data:
            gen_str(point[0:2], i)
            i = i + 1

    if dim == 3:
        for point in data:
            gen_str(point, i)
            i = i + 1   
    return d


#read label image
def read_raster_to_label_image(data_mms, d_mms, res, r):
    #image 
    img = np.zeros((int(r/res),int(r/res),1))
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            key = str(x)+'+'+str(y)
            if key in d_mms.keys():
                img[x,y,:] = np.array(data_mms[d_mms[key],-1])
            else:
                continue
    return img

#####################################################################
# Write binary pointcloud
#####################################################################
def write_origin_points(points, path):
    write_origin_points_header(len(points), path)
    expand_origin_points(points, path)

def write_origin_points_header(count, path):
    with open(path, "w+") as file_:
        file_.write("""ply
format binary_little_endian 1.0
element vertex {0}
property double x
property double y
property double z
property float nx
property float ny
property float nz
property float scalar_sigma
property float scalar_sigma_plane
property float scalar_Omnivariance_(0.25)
property float scalar_Surface_variation_(0.25)
property float scalar_Normal_change_rate_(0.25)
property float scalar_std_ominivarious
property float scalar_std_nz
property float scalar_dist_histogram_bin1
property float scalar_dist_histogram_bin2
property float scalar_dist_histogram_bin3
property float scalar_dist_histogram_bin4
property float scalar_dist_histogram_bin5
property float scalar_dist_histogram_bin6
property float scalar_dist_histogram_bin7
property float scalar_dist_histogram_bin8
property float scalar_dist_histogram_bin9
property float scalar_dist_histogram_bin10
property float scalar_dist_histogram_bin11
property float scalar_dist_histogram_bin12
property float scalar_dist_histogram_bin13
property float scalar_dist_histogram_bin14
property float scalar_dist_histogram_bin15
property float scalar_dist_histogram_bin16
property float label
end_header\n""".format(count))

def expand_origin_points(points, path):

    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("dddfffffffffffffffffffffffffff", *p)
            file_.write(txt)

#####################################################################
# Write binary pointcloud with feature
#####################################################################

def write_feature_points(points, path):
    write_feature_points_header(len(points), path)
    expand_feature_points(points, path)

def write_feature_points_header(count, path):
    
    with open(path, "w+") as file_:
        file_.write("""ply
format binary_little_endian 1.0
element vertex {0}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float scalar_sigma
property float scalar_sigma_plane
property float scalar_Omnivariance_(0.25)
property float scalar_Surface_variation_(0.25)
property float scalar_Normal_change_rate_(0.25)
property float scalar_std_ominivarious
property float scalar_std_nz
property float scalar_dist_histogram_bin1
property float scalar_dist_histogram_bin2
property float scalar_dist_histogram_bin3
property float scalar_dist_histogram_bin4
property float scalar_dist_histogram_bin5
property float scalar_dist_histogram_bin6
property float scalar_dist_histogram_bin7
property float scalar_dist_histogram_bin8
property float scalar_dist_histogram_bin9
property float scalar_dist_histogram_bin10
property float scalar_dist_histogram_bin11
property float scalar_dist_histogram_bin12
property float scalar_dist_histogram_bin13
property float scalar_dist_histogram_bin14
property float scalar_dist_histogram_bin15
property float scalar_dist_histogram_bin16
end_header\n""".format(count))
def expand_feature_points(points, path):

    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("fffffffffffffffffffffffffffff", *p)#*p=[p[0],p[1],p[2]] pack将点转换成float型字节流,float是对local坐标合理，如果是global下x,y应该是double型
            file_.write(txt)

#####################################################################
# Write binary pointcloud with label
#####################################################################

def write_label_points(points, path):
    write_label_points_header(len(points), path)
    expand_label_points(points, path)

def write_label_points_header(count, path):
    with open(path, "w+") as file_:
        file_.write("""ply
format binary_little_endian 1.0
element vertex {0}
property double x
property double y
property double z
property float nx
property float ny
property float nz
property float scalar_sigma
property float scalar_sigma_plane
property float scalar_Omnivariance_(0.25)
property float scalar_Surface_variation_(0.25)
property float scalar_Normal_change_rate_(0.25)
property float scalar_std_ominivarious
property float scalar_std_nz
property float scalar_dist_histogram_bin1
property float scalar_dist_histogram_bin2
property float scalar_dist_histogram_bin3
property float scalar_dist_histogram_bin4
property float scalar_dist_histogram_bin5
property float scalar_dist_histogram_bin6
property float scalar_dist_histogram_bin7
property float scalar_dist_histogram_bin8
property float scalar_dist_histogram_bin9
property float scalar_dist_histogram_bin10
property float scalar_dist_histogram_bin11
property float scalar_dist_histogram_bin12
property float scalar_dist_histogram_bin13
property float scalar_dist_histogram_bin14
property float scalar_dist_histogram_bin15
property float scalar_dist_histogram_bin16
property float scalar_label
end_header\n""".format(count))
def expand_label_points(points, path):

    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack("dddfffffffffffffffffffffffffff", *p)#*p=[p[0],p[1],p[2]] pack将点转换成float型字节流,float是对local坐标合理，如果是global下x,y应该是double型
            file_.write(txt)


# Read original binary pointcloud
def read_origin(path, skiprows):

    xyz = []
    n = (skiprows-4)*4
    with open(path, "rb") as file_:
        skip = 0
        for _ in range(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(n)
            point = struct.unpack("ffffffff", binary)
            xyz.append(point)

    return np.array(xyz)

# Read all_feature binary pointcloud
def read_bin_features(path, skiprows):

    xyz = []
    n = 29*4
    with open(path, "rb") as file_:
        skip = 0
        for _ in range(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(n)
            point = struct.unpack("fffffffffffffffffffffffffffff", binary)
            xyz.append(point)

    return np.array(xyz)

# Read labelled pointcloud
def read_label_bin(path, skiprows):

    xyz = []

    with open(path, "rb") as file_:
        skip = 0
        for _ in range(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(3*8+27*4)
            point = struct.unpack("dddfffffffffffffffffffffffffff", binary)
            xyz.append(point)

    return np.array(xyz)

############################################################################## 
##read raster with probabilites into image
##############################################################################        

#read label image
def read_raster_points_to_label_image(data_mms, d_mms, res, r):
    #image 
    img = np.zeros((int(r/res),int(r/res),5))
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            key = str(x)+'+'+str(y)
            if key in d_mms.keys():
                label = np.array(data_mms[d_mms[key],29:])
                ser = pd.DataFrame(label)
                if len(label)!=0 :
                    labels = ser.mode().values
                    img[x,y,0:4]=labels
            else:
                img[x,y,4] = 1 #add additional one dimention to deal with empty pixel
  
    return img


#neighbors_raster use the mode label as raster label if multiple value use the maximal as mode
def raster_label(data):
    raster = rasterize(data[:,0:3],0.25,dim=2)        
    for key in raster.keys():
        labels = data[raster[key],-1]
        df = pd.DataFrame(labels)
        label =df.mode().values

    if len(label)==1:
        data[raster[key],-1] = label
    else:
        data[raster[key],-1] = max(label)
    
    return data




############################################################################## 
## get the best feature combination for classification
##############################################################################  

def select_combine(X_train,y_train,features_name,features_imp,select_num,rf):
    oob_result = []
    fea_result = []
    features_imp = list(features_imp)
    iter_count = X_train.shape[1] - select_num  #iteration number
    if iter_count < 0:
        print("select_nume must less or equal X_train columns")
    else:
        features_test  = copy.deepcopy(features_imp)   #generate a order list of features
        features_test.sort()
        features_test.reverse() 
        
        while iter_count >= 0:
            iter_count -= 1
            train_index = [features_imp.index(j) for j in features_test[:select_num]]
            #train_feature_name = [features_name[k] for k in train_index][0]
            train_data = X_train[:,train_index]
            rf.fit(train_data,y_train)
            acc = rf.oob_score_
            print(acc)
            oob_result.append(acc)
            fea_result.append(train_index)
            if select_num < X_train.shape[1]:
                select_num += 1
            else:
                break
    return max(oob_result),oob_result,fea_result[oob_result.index(max(oob_result))]


 
#####################################
#                                   #
#    Random Forest Parameter test   #
#                                   #
#####################################
def rf_parameters(x_train,y_train,rf):
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 800, num = 7)]

    # number of features at every split
  
    max_features = [8,10,12,16]
    
    # max depth
    max_depth = [int(x) for x in np.linspace(4, 8, num = 2)]

    #min_sample_split 
    min_sample_split= [1,3,10]
    
    min_sample_leaf =[1,3,10]
    # create random grid
    random_grid = {
     'n_estimators': n_estimators,
     'max_features': max_features,
     'max_depth': max_depth,
     'min_samples_split':min_sample_split,
     'min_samples_leaf': min_sample_leaf
     }
    
    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                    n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the model
    rfc_random.fit(x_train, y_train)
    # print results
    print(rfc_random.best_params_)
    
    report = pd.DataFrame(rfc_random.cv_results_)
    
    
    return rfc_random, rfc_random.best_params_, rfc_random.best_score_,report