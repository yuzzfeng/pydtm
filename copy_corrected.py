###############################################################################
# This step is only necessary, when one of the runid are changed and save at
# Folder 
#
# Yu Feng, 25.03.2020
###############################################################################

import os
from shutil import copyfile

from environmentSetting import correct_out_dir, gfilter_out_dir
from lib.util import check_and_create

# Target
check_and_create(correct_out_dir)

# Source
correct_runid_out_dir = "D:/_data/_mms/Hildesheim_updated/"

"""Caution!!! Copy operation here!"""
if 0: 
    for fn in os.listdir(gfilter_out_dir):
        if fn not in os.listdir(correct_out_dir):
            copyfile(gfilter_out_dir + fn, correct_out_dir + fn)
            
    # Copy to new folder and rename
    for runid in ["1150893024", "1150892210"]:
        for fn in os.listdir(correct_runid_out_dir):
            fn_runid = fn[:10]
            if fn_runid == runid:
                base = fn[-21:]
                copyfile(correct_runid_out_dir + fn, correct_out_dir + base)
                print(fn, base)
                
