import numpy as np
from itertools import product

from lib.cell2world import Hexa2Decimal, coord_fn_from_cell_index

def get_neighbour_matrix(num_elem, r):
    # Get the relative coordinates of neightbours
    if num_elem==9:
        # Center of the neibouring 9 tiles
        # c9n = [-7.5, 7.5, 22.5] # only for the case r=15
        c9n = [-r/2, r/2, 3*r/2]
        coords9n = list(product(c9n, c9n))
        return coords9n
    
    if num_elem==25:
        ## Center of the neibouring 25 tiles
        #c25n = [-22.5, -7.5, 7.5, 22.5, 37.5] # only for the case r=15
        c25n = [-3*r/2, -r/2, r/2, 3*r/2, 5*r/2]
        coords25n = list(product(c25n, c25n))
        return coords25n

    if num_elem==49:
        ## Center of the neibouring 49 tiles
        #c49n = [-37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5] # only for the case r=15
        c49n = [-5*r/2, -3*r/2, -r/2, r/2, 3*r/2, 5*r/2, 7*r/2]
        coords49n = list(product(c49n, c49n))
        return coords49n

def search_neigh_tiles(fn, radius = 1):
    # Search the nearest tiles based on grid and radius
    m,n = fn[:17].split('_')
    int_m = Hexa2Decimal(m)
    int_n = Hexa2Decimal(n)
    combi = np.array(list((product(range(-radius,radius+1), range(-radius,radius+1)))))
    combi_global = combi + [int_m, int_n]
    neigh_list = [coord_fn_from_cell_index(mx,nx,'')[1]+'.ply' for mx,nx in combi_global]
    return neigh_list

def get_distance_with_padding(dict_shift_value, neighbours, shift = 0):
    return np.array([dict_shift_value[nfn] - shift if nfn in dict_shift_value.keys() else 0 for nfn in neighbours])

def get_distance_without_padding(dict_shift_value, neighbours):
    return np.array([dict_shift_value[nfn] for nfn in neighbours if nfn in dict_shift_value.keys()])
