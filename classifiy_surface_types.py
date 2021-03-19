# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:43:36 2020
@author: wenjun / feng
Classification using Random Forest

1. Use label data do training and testing
2. use unlabel data do validation
"""


import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from surface.roughness import read_all_filename, read_bin_features, read_label_bin


#from lib.all_function import *
#import pickle
#import joblib
#from sklearn.preprocessing import label_binarize
#from sklearn.neighbors import KDTree
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc,roc_auc_score


#################Lable data####################################################
path_roughness = "D:/_data/_mms/20200618_Roughness/"
TRAIN_LOC = path_roughness + 'train/'
TEST_LOC = path_roughness + 'test/'

names = ['asphalt','stone','grass','road boundary']
label = [1,2,3,4]

# Feature names
columnindex1 = ['x','y','z','nx','ny','nz',
               'scalar_sigma','scalar_sigma_plane',               
               'scalar_Omnivariance_(0.25)',
               'scalar_Surface_variation_(0.25)',
               'scalar_Normal_change_rate_(0.25)',
               'std_omnivarious','std_nz',
               'scalar_dist_histogram_bin1',
               'scalar_dist_histogram_bin2',
               'scalar_dist_histogram_bin3',
               'scalar_dist_histogram_bin4',
               'scalar_dist_histogram_bin5',
               'scalar_dist_histogram_bin6',
               'scalar_dist_histogram_bin7',
               'scalar_dist_histogram_bin8',
               'scalar_dist_histogram_bin9',
               'scalar_dist_histogram_bin10',
               'scalar_dist_histogram_bin11',
               'scalar_dist_histogram_bin12',
               'scalar_dist_histogram_bin13',
               'scalar_dist_histogram_bin14',
               'scalar_dist_histogram_bin15',          
               'scalar_dist_histogram_bin16']
columnindex2 = columnindex1 + ['label']


#get train data
train=pd.DataFrame(columns=columnindex1)
for i in range(len(names)):
    loc = os.path.join(TRAIN_LOC,names[i])
    data_name = read_all_filename(loc)  
    for m in range(0,len(data_name)):
        name = data_name[m]
        load_data = read_bin_features(os.path.join(loc,name),36)           
        data = pd.DataFrame(load_data,columns=columnindex1)
        data['label']=label[i]
        train=train.append(data,ignore_index=True,sort=False)
train = train.dropna(axis=0)

#get test data
test_name= read_all_filename(TEST_LOC)
test=pd.DataFrame(columns=columnindex2)
for j in range(0,len(test_name)):
    jn = test_name[j]
    load_data = read_label_bin(os.path.join(TEST_LOC,jn),34)
    data = pd.DataFrame(load_data,columns=columnindex2)
    test=test.append(data,ignore_index=True,sort=False)
test = test.dropna(axis=0)

print("Data loaded")

num_s = len(test.loc[test['label']==2])/3
num_rb = num_s/len(test.loc[test['label']==4])-1
copy_rb = test.loc[test['label']==4].sample(frac=num_rb,replace=True,random_state=1)
test = test.append(copy_rb,ignore_index=True,sort=False)

#data normalization use the same standard scalar get the mean and standard variance of each feature columns

X=train.values
np.random.shuffle(X)

train_prep=X[:,7:29]
#train_prep=X[:,7:14]

sc=StandardScaler().fit(train_prep)
x_train = sc.transform(train_prep)
y_train = X[:,-1]
y_train = np.int0(y_train) - 1

# =============================================================================
#joblib.dump(sc, 'Scalarstandard_22')
# =============================================================================

x_t = test.values
np.random.shuffle(x_t)
xt=x_t[:,7:29]
#xt=x_t[:,7:14]

x_test = sc.transform(xt)
y_test = x_t[:,-1]
y_test = np.int0(y_test) - 1



# train classifier
class_weight={1:1,2:1,3:1,4:3}

if 1: #Yu: run xgboost
    
    from sklearn.utils import class_weight
    import xgboost as xgb
    

    class_weights = list(class_weight.compute_class_weight('balanced',
                                                           np.unique(y_train),
                                                           y_train))
    class_weights = np.array(class_weights) / sum(class_weights)
    
    dtrain = xgb.DMatrix(x_train, y_train)#, weight = class_weights)
    dvalid = xgb.DMatrix(x_test, y_test)
    
    param = {'max_depth':6, 'eta':0.3, 'num_class':4,
             'objective':'multi:softprob', 'silent':1,
             'eval_metric': ['mlogloss']}
            
    num_round = 300
            
    evallist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    
    predict_train = np.argmax(bst.predict(dtrain), axis=1)
    predict_test = np.argmax(bst.predict(dvalid), axis=1)
    
else: #Wenjun: run rf
    
    # =============================================================================
    # with open("rf_random.pkl", "rb") as f:
    #      rf1 = pickle.load(f)
    # =============================================================================
    
    rf = RandomForestClassifier(n_estimators=350, criterion='entropy', 
                                min_samples_leaf=20, max_depth=10,
                                max_features=10,min_samples_split=50,
                                oob_score=True,random_state=0, 
                                class_weight=class_weight)
    rf.fit(x_train,y_train)
    
    print("Training finished")
    
    # output feature importance
    features_imp = rf.feature_importances_

    #save model

    #select features
    #oob_score, oob_list,features = select_combine(x_train,y_train,columnindex1,features_imp,18,rf)
    
    # get parameters
    #rf_random,parameters,score,report = rf_parameters(x_train,y_train,rf)
    
    # use classifer to predict data
    predict_train=rf.predict(x_train)
    
    #predict probability
    #predict_prob = rf.predict_proba(x_train)
    
    #predict test data
    predict_test=rf.predict(x_test) 


# classification report
train_report=classification_report(y_train,predict_train,digits=4)
print(train_report)
test_report=classification_report(y_test,predict_test,digits=4)
print(test_report)

# generate confusion matrix and plot
conf_train = confusion_matrix(y_train,predict_train,normalize='true')
pd_train = pd.DataFrame(conf_train)
fig,ax = plt.subplots(figsize=(6,5))
ax=sn.heatmap(conf_train,cmap="Blues",annot=True,xticklabels=names,yticklabels=names)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Normlaized confusion matrix of train')

conf_test = confusion_matrix(y_test,predict_test,normalize='true')
pd_test = pd.DataFrame(conf_test)
fig,ax = plt.subplots(figsize=(6,5))
ax=sn.heatmap(conf_test,cmap="Blues",annot=True,xticklabels=names,yticklabels=names)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Normlaized confusion matrix of test')



######################################Validation###############################
from surface.roughness import GF_to_global, write_label_points
from lib.tool import write_xyz, write_ply

## Write PLY file
def write_ply_no_header(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        np.savetxt(f, verts, '%f %f %f %d %d %d')
        
test_loc = path_roughness + 'test/'
path_out = path_roughness + 'testpred/'
path_xyz = path_roughness + 'testtmp/'
origin = [564546,5778458]
cellsize = 25
list_mms =os.listdir(test_loc)

"""
for fn in list_mms:# range(3,200):
    
    load_data = read_label_bin(os.path.join(test_loc,fn),34)
    #load_data = read_bin_features(os.path.join(test_loc,fn),33)
    test_data = np.delete(load_data,np.where(np.isnan(load_data))[0],axis=0)

    test_data1 = test_data[:,7:29]
    
    test_standard = sc.transform(test_data1)
    
    dtest = xgb.DMatrix(test_standard)
    predict_validate = np.argmax(bst.predict(dtest), axis=1) 

    result2 = pd.DataFrame(test_data,columns = columnindex2)
    result2['label']=predict_validate
    
    data = result2.iloc[:,0:3].values
    
    def normalize(x):
        
        min_val = np.percentile(x, 10, axis=0)
        max_val = np.percentile(x, 90, axis=0)
        
        norm = (x-min_val)/(max_val-min_val)
        
        norm[norm>1] = 1
        norm[norm<0] = 0
        
        return 255.0*norm

    color = normalize(result2.iloc[:,7:10].values) #np.zeros(data.shape)
    write_ply_no_header(path_xyz + fn[:17] + '.xyz', data, color)
    
    result2.iloc[:,0:3]=result2.iloc[:,0:3] + GF_to_global(fn,origin,cellsize)
    write_label_points(result2.values, os.path.join(path_out,fn))
"""

path_sv = path_roughness + 'testsupervoxel/'

## Read PLY file (Quicker than np.genfromtxt)
def iter_loadtxt(filename, delimiter, skiprows, dtype):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def read_ply(filename, sk):
    data = iter_loadtxt(filename, delimiter=' ', skiprows=sk, dtype=np.float)
    return data


for fn in os.listdir(path_sv)[:2]:
    
    data = read_ply(path_sv+fn,0)
    
    su_dict = dict()
    for i, l in enumerate(data):
        key = ' '.join([str(int(elem)) for elem in l[3:]])
        if key in su_dict:
            su_dict[key] = su_dict[key] + [i]
        else:
            su_dict[key] = [i]
    
    
    load_data = read_label_bin(os.path.join(test_loc,fn[:17] + '.ply'),34)
    test_data = np.delete(load_data,np.where(np.isnan(load_data))[0],axis=0)

    test_data1 = test_data[:,7:29]
    test_standard = sc.transform(test_data1)
    
    dtest = xgb.DMatrix(test_standard)
    predict_validate = np.argmax(bst.predict(dtest), axis=1) 

    
    for key in su_dict.keys():
        ind = su_dict[key]
        su_preds = predict_validate[ind]
        counts = np.bincount(su_preds)
        su_pred = np.argmax(counts)
        predict_validate[ind] = su_pred
    
    result2 = pd.DataFrame(test_data,columns = columnindex2)
    result2['label']=predict_validate
    
    result2.iloc[:,0:3]=result2.iloc[:,0:3] + GF_to_global(fn,origin,cellsize)
    write_label_points(result2.values, os.path.join(path_out,'x'+fn[:17] + '.ply'))