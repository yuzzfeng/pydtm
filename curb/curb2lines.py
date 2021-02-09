import os.path
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN

# iterate each line in txt or ply data
def read_ascii_xyz(filename, delimiter, skiprows, dtype):

    def iter_func():
        with open(filename, 'r') as infile:
            for _ in xrange(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                v = line[0:3]
                
                for item in v:
                    yield dtype(item)
    
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, 3))
    return data

cur_dir = "C:\\SVN\\datasets\\lidar\\ricklingen_adjusted_curb_ply\\"
directory = [ x for x in os.listdir(cur_dir ) if x[-4:] == ".ply" ]

featurePoint = np.array([0,0,0])

for fn in directory:
    points = read_ascii_xyz(cur_dir + fn, ' ', 8, dtype= np.double)
    featurePoint = np.vstack([featurePoint, points])


X = featurePoint[:,:2]

db = DBSCAN(eps=0.20, min_samples=5).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

lines = []
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
angles = X[:,1] / X[:,0]

for k, col in zip(unique_labels, colors):
    if k != -1:
        class_member_mask = (labels == k)
        plt.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=3)
                 
        angles[class_member_mask]
                 
        line = LineString(X[class_member_mask][np.argsort(angles[class_member_mask])])
        tolerance = 0.025
        simplified_line = line.simplify(tolerance, preserve_topology=False)
        lines.append(simplified_line)

g = gp.GeoSeries(lines)
g.to_file("lines.shp")

