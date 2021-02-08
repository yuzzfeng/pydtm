import struct
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def plot_img(img):  
    plt.figure()
    plt.imshow(img)
    plt.colorbar()

def Hexa2Decimal(x):
    tile = struct.unpack('>i', x.decode('hex'))
    return tile[0]

def coord(m,n,r,x_offset,y_offset):
    return r*Hexa2Decimal(m)+x_offset, r*Hexa2Decimal(n)+y_offset

# Read file name
def read_fn(fn,r,x_offset,y_offset):
    m,n = fn.split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

# Calc extent
def get_range_from_fn_list(fn_list, r,x_offset,y_offset):

    mn = [read_fn(fn[:17],r,x_offset,y_offset) for fn in fn_list]
    minM, minN = np.min(mn, axis=0)
    maxM, maxN = np.max(mn, axis=0)

    M = maxM - minM + r
    N = maxN - minN + r

    return minM, minN, M/r, N/r

# Save array as image
def array_to_raster(dst_filename, img, x_min, y_max, PIXEL_SIZE):
    
    # Save array to tiff file
    from osgeo import gdal
    
    # Initial meta data
    geotransform = (x_min, PIXEL_SIZE, 0, y_max, 0, -PIXEL_SIZE)
    y_pixels, x_pixels = img.shape
    
    # Create file
    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform(geotransform)  

    #dataset.SetProjection(image.GetProjection())
    
    # Write data
    dataset.GetRasterBand(1).WriteArray(img)
    # Write to disk.
    dataset.FlushCache()  
    
    return dataset, dataset.GetRasterBand(1)

# Create binary or height mask map with single tile as pixel
def gen_mask4mms(dst_filename, list_fns, args, dict_fns = None, nonvalue = -999.0, is_plot = False):
    
    # Basic Info of origin and tile size in meter
    r, x_offset, y_offset = args
    
    # Collect tiles names for tile list
    minM, minN, lenM, lenN = get_range_from_fn_list(list_fns, r, x_offset, y_offset)
    
    # Empty big image
    img = np.nan * np.ones((lenN, lenM))
    
    # Insert tile into big image
    for fn in tqdm(list_fns):
        
        # Parse filename
        x, y = read_fn(fn[:17], r, x_offset, y_offset)
        
        # Get start ids
        x_id = (y-minN)/r
        y_id = (x-minM)/r
        #print(x_id, y_id)
        
        if type(dict_fns)==type(None):
            # Insert tile to big image
            img[x_id:(x_id+1), y_id:(y_id+1)] = 1
        else:
            if fn in dict_fns:
                img[x_id:(x_id+1), y_id:(y_id+1)] = dict_fns[fn]
            else:
                img[x_id:(x_id+1), y_id:(y_id+1)] = 1

    # Plot with matplotlib, need flip to visualize
    if is_plot:
        plot_img(np.flip(img, axis=0))
    
    # Save as tiff, get original, notice x_min and y_max
    x_min = minM
    y_max = int(img.shape[0]*r + minN)
    
    # Save as tiff, save need flip
    array_to_raster(dst_filename, np.flip(img, axis=0), x_min, y_max, r)

# Create height update map with intepolated raster
def merge_updates4mms(dst_filename, dict_shift_img, res, args, nonvalue = -999.0, is_plot = False):
    
    # Basic Info of origin and tile size in meter
    r, x_offset, y_offset = args
    
    # Tile Size in meter
    size = int(r/res)
    
    # Collect tiles names for tile list
    minM, minN, lenM, lenN = get_range_from_fn_list(dict_shift_img.keys(), r, x_offset, y_offset)
    
    # Empty big image
    img = np.nan * np.ones((lenN*size, lenM*size))
    
    # Insert tile into big image
    for fn in tqdm(dict_shift_img.keys()):
        
        # Parse filename
        x, y = read_fn(fn[:17], r, x_offset, y_offset)
        
        # Get start ids
        x_id = (y-minN)/r
        y_id = (x-minM)/r
        #print(x_id, y_id)
        
        # Insert tile to big image
        img[x_id*size:(x_id+1)*size, y_id*size:(y_id+1)*size] = dict_shift_img[fn]

    # Plot with matplotlib, need flip to visualize
    if is_plot:
        plot_img(np.flip(img, axis=0))
    
    # Save as tiff, get original, notice x_min and y_max
    x_min = minM
    y_max = int(img.shape[0]*res + minN)
    
    # Save as tiff, save need flip
    array_to_raster(dst_filename, np.flip(img, axis=0), x_min, y_max, res)
    
    return img

