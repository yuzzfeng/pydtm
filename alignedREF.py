# Copyright (C) 2016 Yu Feng <yuzz.feng@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

###################################################
## Split the given ply files in tiles
###################################################

from environmentSetting import * 
from lib.assemble import split_ref_to_tiles

split_ref_to_tiles(ref_path, ref_dir, r, x_offset, y_offset, res_ref)

