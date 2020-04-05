# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Get the max extent of the MMS data

import os 
import numpy as np

from environmentSetting import *

from lib.cell2world import read_fn

list_pointcloud = os.listdir(pointcloud_path)

origins = []

for fn in list_pointcloud:
    mm, nn = read_fn(fn,r,x_offset,y_offset)
    origins.append([mm, nn])
    
print "MMS Rnage", np.min(origins, axis=0), np.max(origins, axis=0) + np.array([r,r])
print "Task Range", [564438.24, 5777247.39, 565788.29, 5778450.66]

min_vals = np.vstack([np.min(origins, axis=0), [564438.24, 5777247.39]])
max_vals = np.vstack([np.max(origins, axis=0) + np.array([r,r]), [565788.29, 5778450.66]])

print "Result Range", np.min(min_vals, axis=0), np.max(max_vals, axis=0)
print "Middle Point", np.mean(np.vstack([np.min(min_vals, axis=0), np.max(max_vals, axis=0)]), axis=0)