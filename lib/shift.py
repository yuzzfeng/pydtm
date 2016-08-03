import numpy as np
import json
import os.path
import tool

from read import rasterize

# return the index for true or false
def reject_outliers(data, m):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s<m


# read shift json file
def read_from_json(jsonfn):
    if os.path.exists(jsonfn):
        with open(jsonfn, 'r') as fp:
            dShift = json.load(fp)
    else:
        dShift = dict()
        print 'No shift data'

    return dShift


# select the best from the shifts
def select_from_shifts(dShift, reject_threshold):
    
    if len(dShift) >1:
        data = np.array(dShift)
        index = reject_outliers(data, m = reject_threshold)
        shifts = data[index]
    else:
        shifts = dShift

    shift_value = np.mean(shifts)
    print shift_value, np.std(shifts), len(shifts), len(dShift)

    return shift_value


def shiftvalue(data_mms, res_ref, data_ref, d_ref, reject_threshold, r):

    d_mms = rasterize(data_mms, res_ref, dim = 2)
    img = np.zeros((int(r/res_ref),int(r/res_ref)))

    delta = []
    for key, value in d_mms.iteritems():
        x, y = [int(k) for k in key.split('+')]
        
        z_mms = data_mms[value,2]
        mean_mms = np.mean(z_mms)
        l = len(z_mms)

        if l >5:
            mean_ref = data_ref[d_ref[key],2]
            delta.append(mean_mms - mean_ref)
            img[x, y] = mean_mms - mean_ref

    if len(delta)>1:
        np.place(img, img==0, None)
        delta = np.array(delta)
        alldelta = np.copy(delta)

        ind = reject_outliers(delta, reject_threshold)
        
##        np.place(delta, ind==False, None)
##        import matplotlib.pyplot as plt
##        plt.figure()
##        plt.plot(range(len(alldelta)),alldelta,'bo')
##        plt.plot(range(len(delta)),delta,'r+')
##        plt.show()

        return np.mean(delta[ind]), img
    else:
        return None,None
    
def calcMeanVoxelHeight(data_mms, d_mms):

    deltadict = dict()
    for key, value in d_mms.iteritems():

        mean_mms = np.mean(data_mms[value,2])
        deltadict[key] = np.array([mean_mms])

    delta = np.array(deltadict.values())
    return delta, deltadict
