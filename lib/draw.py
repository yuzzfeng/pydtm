import math
import numpy as np
import matplotlib.pyplot as plt

from lib.cell2world import coord

def plot_tiles(fns, args):
    
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    
    _, _, _, _, r, x_offset, y_offset, _, _, _, _ = args
    
    # Get origin from initial filename
    m,n = fns[0][:17].split('_')
    [mm0,nn0] = coord(m, n, r, x_offset, y_offset)
    
    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    patches = []
    
    for fn in fns:
            
        # Get origin from filename
        m,n = fn[:17].split('_')
        [mm,nn] = coord(m, n, r, x_offset, y_offset)
        
        mm = float(mm) - mm0
        nn = float(nn) - nn0
        
        # Create a Rectangle patch
        rect = Rectangle((mm, nn), r, r, linewidth=1, 
                         edgecolor='r',facecolor='r')
        patches.append(rect)
    
    collection = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.8)
    
    #collection.set_clim(-0.5, 0.5)
    ax.add_collection(collection)
    
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_tiles_hdiff(dict_values, args):
    
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    
    fns = dict_values.keys()
    _, _, _, _, r, x_offset, y_offset, _, _, _, _ = args
    
    # Get origin from initial filename
    m,n = fns[0][:17].split('_')
    [mm0,nn0] = coord(m, n, r, x_offset, y_offset)
    
    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    patches = []
    values = []
    
    for fn in fns:
        
        if fn in dict_values:
            
            if dict_values[fn]!=None and not math.isnan(dict_values[fn]):

                values.append(dict_values[fn])
            
                # Get origin from filename
                m,n = fn[:17].split('_')
                [mm,nn] = coord(m, n, r, x_offset, y_offset)
                
                mm = float(mm) - mm0
                nn = float(nn) - nn0
                
                # Create a Rectangle patch
                rect = Rectangle((mm, nn), r, r, linewidth=1, 
                                 edgecolor='r',facecolor='r')
                patches.append(rect)
    
    collection = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.8)
    collection.set_array(np.abs(values))
    
    #collection.set_clim(-0.5, 0.5)
    ax.add_collection(collection)
    fig.colorbar(collection)
    
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()