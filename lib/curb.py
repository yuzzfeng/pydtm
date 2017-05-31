if 1:
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    from scipy.optimize import fmin, fmin_cg
    from scipy.optimize import curve_fit
    
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
    floor = math.floor

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

def rotZDirection(heading, delta, pos):
    
    heading = heading*pi/180
    new_list= dot(rotZ(heading),delta.transpose())
    new_p=pos+new_list.transpose()
    
    newX = new_p[:,0]
    newY = new_p[:,2]
    
    X = newX - mean(newX)
    minY = min(newY)
    maxY = max(newY)
    Y = (newY-minY)/(maxY-minY)
    
    return X,Y
    
def calcProfile(heading, delta, pos):
    
    eps = 0.005;
    X, Y = rotZDirection(heading, delta, pos)
    
    hash_list = []
    for x,y in zip(X,Y):
        num = [str(int(floor(x/eps))),
               str(int(floor(y/eps)))]
        string = join('+',num)
        hash_list.append(string)

    return len(set(hash_list))

    
def searchProfile(delta, pos):
           
    occupied = [calcProfile(heading, delta, pos) for heading in np.arange(-180,180,1)]      
    print np.argmin(occupied)
    return np.arange(-180,180,1)[np.argmin(occupied)]
   
   
def curv(X,Y):
    try:
        popt, pcov = curve_fit(sigmoid, X, Y)
        y = sigmoid(X, *popt) - Y
        return np.sqrt(sum(pow(y,2))) #sum(abs(y))
    except:
        return 10000000


def rotPoints(heading,delta,pos):
    
    X, Y = rotZDirection(heading, delta, pos)
    delta = curv(X,Y)
    return delta



def sigmoidFitting(list_z):

    pos = mean(list_z,axis=0)
    delta= list_z-pos
    
    i = searchProfile(delta, pos)
    i = fmin_cg(rotPoints, i, args = (delta,pos), maxiter = 100, disp = False)[0]
    
    try:

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

        maxPoint = np.max(list_z, axis = 0)
        minPoint = np.min(list_z, axis = 0)
        
        if all(featureR[0]>minPoint) and all(featureR[0]<maxPoint) and all(featureR[1]>minPoint) and all(featureR[1]<maxPoint):
        
            if 1:
                print len(newX)           
                plt.plot(newx,newy, label='fit')
                plt.plot(newX,newY, '.')
                plt.plot(feature[:,0],feature[:,2], 'ro')
                plt.show()
        
            return featureR, z, i
        else:
            return [],0,0
    except:
        return [],0,0


# search for neighbourhoods
def curbDetectiono(key, value, m, d, data, threshold):

    min_z, max_z, min_v, max_v, min_delta, max_delta, hori_dist = threshold

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

    if v >min_v and v <max_v and len(list_neighnour)>150:
        delta = max(list_z)-min(list_z)
        if delta>min_delta and delta<max_delta:
            
#            list_points.extend(list_neighnour)
            
            feat,z,i = sigmoidFitting(data[list_neighnour,:])
            z = abs(z)
            
            if len(feat)>0:
                featarray = np.array(feat)
                dist_2d = np.linalg.norm((featarray[0,:] - featarray[1,:])[:2])
                height = abs((featarray[0,:] - featarray[1,:])[2])
            
                if height > min_delta and height < max_delta: # and dist_2d < hori_dist:
#                    print list(feat)
                    feature.extend([list(feat)[0]])
                    intensity.extend([i])
                
#            if z > min_z and z < max_z and dist_2d < hori_dist:
#                
#                feature.extend([list(feat)[0]]) # 1 Points
#                intensity.extend([i])         
#                feature.extend(list(feat)) # 2 Points
#                intensity.extend([i,i])

    return (list_points,feature,intensity)


# Functions for single or multiple processing #################################
def curbDetection_star(a_b):
    return curbDetectiono(*a_b)

def curb_single_core(d,data,m,threshold):
    
    results = []
    for dk, dv in zip(d.keys(), [np.array(x) for x in d.values()]):
        res = curbDetectiono(dk, dv, m, d, data, threshold)
        results.append(res)
        
    return results

def curb(d,data,m,threshold,multi):

    dk = d.keys()
    dv = [np.array(x) for x in d.values()]
    p = Pool(multi)
    result = p.map(curbDetection_star, izip(dk, dv, repeat(m), repeat(d), repeat(data), repeat(threshold)))
    p.close()
    p.join()

    return result
###############################################################################