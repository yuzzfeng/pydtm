#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


'''
Environment Parameters Setting - pydtm

Define the path of input lidar data and input reference data

With an given refence data such as a 2D Grid/Image, tranform it firstly to 3D point cloud, e.g. using ArcGIS.
If it is a large area, please partition to small. Then save under reference path and name them such as:

[xxx_1.ply, xxx_2.ply, xxx_3.ply] 

Input of the data should be partitioned into grids based on a global origin and grid size. The id of each
grid in x and y direction are then calculated in hex number in 8 digit, e.g. ffffffff_00000001. This step make
the file name contains coordinates in global coordinate system.

'''

##project_name = 'Hannover'
##project_name = '20190924_Hildesheim'
##project_name = '20200616_DTM_SEHi'
project_name = '20201022_Hildesheim'

# Temp file for saving height changes
tmp_dir = "tmp/" + project_name + '/'

# Output Path
#out_path = 'C:\\temp\\DTM_04042018\\'
#out_path = 'C:\\temp\\DTM_20180416\\'
#out_path = 'C:\\temp\\DTM_20190711\\'
#out_path = 'C:\\temp\\DTM_Hildesheim_20190923\\'
#out_path = 'D:\\_data\\_mms\\20200325Hildesheim\\'
#out_path = 'D:\\temp\\20200616_DTM_SEHi\\'
out_path = 'D:\\temp\\20201022_DTM_Hildesheim\\'


### Lidar Points Path
##pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'
##pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map_bauarbeit\\'
##pointcloud_path = 'X:\\Products\\ricklingen_adjusted_cloud_40m_ply\\'
pts_dir = 'X:\\Products\\20190919_hildesheim_adjustment\\adjusted_cloud_ply\\'
##pts_dir = "D:/_data/_mms/adjusted_cloud_ply/" # Feng-PC
##pts_dir = "X:/Products/20200117_hildesheim3/adjusted_cloud_15_ply/"

## Reference Points path
##ref_path = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_ply\\1in4\\'
ref_path = 'D:\\_data\\_airborne_laser_scanning\\20190924_DTM_Hildesheim\\ply\\'
##ref_path = 'D:\\_data\\_airborne_laser_scanning\\20200616_DTM_SEHi\\ply\\'
# Path for intermediat results
ref_dir = out_path + 'ref\\'

## Feng-PC
##ref_dir = "D:/_data/_dgm/reference/" # Feng-PC


# tmp foldersfor intermediat result
gfilter_out_dir = out_path + 'aligned_GF\\'
correct_out_dir = out_path + 'aligned_Corr\\'
gupdate_out_dir = out_path + 'aligned_GR\\'

ref_update_dir = out_path + 'aligned_ref_update\\'
ref_cut_dir = out_path + 'aligned_ref_update_cut\\'

merged_dir = out_path + 'aligned_a\\'
geo_ground_filtering_out_dir = out_path + 'aligned_a\\'
final_dir = out_path + 'aligned_b\\'
rest_dir = out_path + 'aligned_c\\'

# Glabal original in the meter coordinate system, e.g. UTM 32 Zone, as well as grid size

## Hannover Ricklingen
#x_offset = 548495 + 5
#y_offset = 5804458 + 42
#r = 15

## Hildesheim
x_offset = 564546
y_offset = 5778458
r = 25

# Resolusion of the reference DTM grids
res_ref = 0.5

tolerance_up = 3
tolerance_down = 0.5

# General geoid height in that area as a prior, 
# Reference: http://geographiclib.sourceforge.net/cgi-bin/GeoidEval

## Hannover 52.348816 9.725389
#geoid = 42.9664
#sigma_geoid = 0.4

## Hildesheim  52.144409 9.95422
geoid = 43.7786
sigma_geoid = 0.4

# Hildesheim SEHi 52.179003, 9.926203
#geoid = 43.6318
#sigma_geoid = 0.4

# Ground filtering parameters
radius = 3
res_list = [1.0, 0.5, 0.25, 0.1, 0.05] 

# Update parameter
#shift = 42.9317864988 # Hanvnoer
#shift = 43.5346042674 # hildesheim

res_update = 0.1

nonvalue = -999.0