import os.path
import numpy as np
import matplotlib.pyplot as plt

from lib.info import *
from lib.cell2world import coord

global almost_black
almost_black = "#262626"


x_offset = 548495 + 5
y_offset = 5804458 + 42
r = 15

def rect(list_kr):    

    for line in list_kr:
        m,n = line[:17].split("_")
        [m,n] = coord(m,n,r,x_offset,y_offset)
        rectXY = plt.Rectangle((m, n), r, r, facecolor="w", edgecolor='b', alpha=1, zorder=1)
        plt.gca().add_patch(rectXY)
    plt.axis('scaled')
    plt.show()

def draw_kacheln_nets():
    r = np.linspace(0,r,2*r)
    for m in r:
        for n in r:
            rectXY = plt.Rectangle((m, n), 0.5, 0.5, facecolor="w", edgecolor='b', alpha=1, zorder=1)
            plt.gca().add_patch(rectXY)

    plt.axis('scaled')
    plt.show()


##list_mms = []
##
##for mms_dir in in_dir_mms:
##    list_mms += os.listdir(mms_dir )
##
##list_mms = list(set(list_mms))

ground_filtering_out_dir = 'C:\\temp\\aligned_GF_04082016\\'
list_pointcloud_filtered = os.listdir(ground_filtering_out_dir)
rect(list_pointcloud_filtered)

##draw_kacheln_nets()


