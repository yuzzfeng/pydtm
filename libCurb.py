import math
import numpy as np

from scipy.optimize import fmin
from scipy.optimize import curve_fit
##from matplotlib import pyplot as plt

from multiprocessing import Pool, freeze_support
from itertools import izip
from itertools import repeat

cos = math.cos
sin = math.sin
pi = math.pi

mean = np.mean
exp = np.exp
dot =np.dot

join = str.join

def sigmoid(x, x0, y0, k, z):
    y = z*(1 / (1 + exp(-k*(x-x0))))+y0
    return y

def sigmoidworld(x, meanX, maxY, minY, x0, y0, k, z):
    y = (z*(1 / (1 + exp(-k*(x-meanX-x0))))+y0)*(maxY-minY)+minY
    return y

def rotZ(kappa):
    # Reference Lasescanning Lecture Chapter 3
    Rz = np.array([[cos(kappa),sin(kappa),0],
                   [-sin(kappa),cos(kappa),0],
                   [0,0,1]])
    return Rz

def curv(X,Y):
    try:
        popt, pcov = curve_fit(sigmoid, X, Y)
        y = sigmoid(X, *popt) - Y
        return sum(abs(y))
    except:
        return 100000


def rotPoints(heading,delta,pos):
    
    heading = heading*pi/180
    new_list= dot(rotZ(heading),delta.transpose())
    new_p=pos+new_list.transpose()

    newX = new_p[:,0]
    newY = new_p[:,2]

    X = newX - mean(newX)
    minY = min(newY)
    maxY = max(newY)
    Y = (newY-minY)/(maxY-minY)
    
    delta = curv(X,Y)
    
    return delta



def sigmoidFitting(list_z):

    pos = mean(list_z,axis=0)
    delta= list_z-pos


    i = fmin(rotPoints, 0, args = (delta,pos), maxiter =50 , disp = False)[0]

    if i!=0:

        heading = i*pi/180
        new_list= dot(rotZ(heading),delta.transpose())
        new_p=pos+new_list.transpose()

        newX = new_p[:,0]
        newY = new_p[:,2]

        meanX = mean(newX)
        minY = min(newY)
        maxY = max(newY)
        
        X = newX-meanX 
        Y = (newY-minY)/(maxY-minY)

        popt, pcov = curve_fit(sigmoid, X, Y)

        [x0, y0, k, z] = popt

        newx = np.arange(min(newX),max(newX),0.01)
        newy = sigmoidworld(newx, meanX, maxY, minY, *popt)

        if z >= 0:
            feature = np.array([[2/k + meanX +x0,pos[1],max(newy)],[-2/k + meanX +x0,pos[1],min(newy)]])
        else:
            feature = np.array([[-2/k + meanX +x0,pos[1],max(newy)],[2/k + meanX +x0,pos[1],min(newy)]])
        features = feature - pos
        featureR= dot(rotZ(-heading),features.transpose()).transpose()+ pos


##        plt.plot(newx,newy, label='fit')
##        plt.plot(newX,newY, '.')
##        plt.plot(feature[:,0],feature[:,2], 'ro')
##        plt.show()
        return featureR, z, i
    else:
        return 0,0,0


# search for neighbours
def curbDetectiono(key,value,m,d,data,threshold):

    min_z, max_z, min_v, max_v, min_delta, max_delta = threshold

    intensity = []
    list_points = []
    feature = []

    coord = [int(k) for k in key.split('+')]
    list_neighnour = []

    
    for win in m:
        num = [str(coord[0] + win[0]),str(coord[1] + win[1]),str(coord[2] + win[2])]
        st = join('+',num)

        if d.has_key(st) == True:
            list_neighnour += d[st]


    list_z = data[list_neighnour,2]
    v = np.var(list_z)

    if v >min_v and v <max_v:
        delta = max(list_z)-min(list_z)
        if delta>min_delta and delta<max_delta:
            value_len = len(value)
        
            list_points.extend(value)
            feat,z,i = sigmoidFitting(data[list_neighnour,:])
            z = abs(z)

            if z > min_z and z < max_z:
                feature.extend(list(feat))
                intensity.extend([i,i])


    return (list_points,feature,intensity)


def curbDetection_star(a_b):
    return curbDetectiono(*a_b)


def curb(d,data,m,threshold,multi):

    dk = d.keys()
    dv = [np.array(x) for x in d.values()]
    p = Pool(multi)
    result = p.map(curbDetection_star, izip(dk, dv, repeat(m), repeat(d), repeat(data), repeat(threshold)))
    p.close()
    p.join()

    return result
