#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


'''
Environment Parameters Setting - pydtm

Generally you need to define the input lidar data and input reference data

With an given refence data such as a 2D Grid/Image, tranform it firstly to 3D point cloud, e.g. using ArcGIS.
If it is a large area, please partition to small. Then save under reference path and name them such as:

[xxx_1.ply, xxx_2.ply, xxx_3.ply] 



'''

# Lidar Points Path
pointcloud_path = 'X:\\Proc\\ricklingen_yu\\map\\'

# Reference Points path
ref_path = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_ply\\1in4\\'

# Output Path
out_path = 'C:\\temp\\DTM_29092016\\'

# Path for intemediat results
ref_out_dir = out_path + 'ref\\'
ground_filtering_out_dir = out_path + 'aligned_GF\\'

ref_update_dir = out_path + 'aligned_ref_update\\'
ref_cut_dir = out_path + 'aligned_ref_update_cut\\'

geo_ground_filtering_out_dir = out_path + 'aligned_a\\'
final_dir = out_path + 'aligned_b\\'
rest_dir = out_path + 'aligned_c\\'

# Glabal original in the meter coordinate system, e.g. UTM 32 Zone, as well as grid size
x_offset = 548495 + 5
y_offset = 5804458 + 42
r = 15

# Resolusion of the reference DTM grids
res_ref = 0.5

# General geoid height in that area as a prior, http://geographiclib.sourceforge.net/cgi-bin/GeoidEval
geoid = 42.9664
sigma_geoid = 0.4
