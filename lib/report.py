if 1:
    import os 
    import numpy as np
    import matplotlib.pyplot as plt
    
    from lib.cell2world import coord
    from lib.shift import  shiftvalue, reject_outliers
    from lib.util import check_and_create
    from lib.asc import write_asc

def plot_img(img):  
    plt.figure()
    plt.imshow(img)
    plt.colorbar()

def read_fn(fn,r,x_offset,y_offset):
    m,n = fn.split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def get_range_from_fn_list(fn_list, r,x_offset,y_offset):

    mn = [read_fn(fn[:17],r,x_offset,y_offset) for fn in fn_list]
    minM, minN = np.min(mn, axis=0)
    maxM, maxN = np.max(mn, axis=0)

    M = maxM - minM + r
    N = maxN - minN + r

    return minM, minN, M/r, N/r


def generate_diff_image(list_shift_img, M, N, minM, minN,
                        scale, nonvalue, report_path,
                        r, x_offset,y_offset, res_ref):

    diff_img = nonvalue * np.ones((M * r / res_ref, N * r / res_ref))

    for fn, data in list_shift_img.iteritems():
        mm,nn = read_fn(fn[:17],r,x_offset,y_offset)
        mm,nn = mm-minM, nn-minN

        diff_img[mm*scale:mm*scale+r*scale,nn*scale:nn*scale+r*scale] = data

    write_asc('diff', report_path, minM, minN, np.flipud(diff_img.T), res_ref)



def generate_report(list_shift_value, list_shift_img, dict_shift_value, out_path, r, x_offset,y_offset, res_ref):

    report_path = out_path + 'report\\'
    check_and_create(report_path)

    ind = reject_outliers(list_shift_value, 20)
    list_shift_value = np.array(list_shift_value)
    values = np.array(list_shift_value)[ind]
    max_value = max(values)
    min_value = min(values)
    
    dict_shift_value_cp = dict_shift_value.copy()
    
    tobedeleted = []
    for key, value in dict_shift_value_cp.iteritems():       
        if value < min_value or value > max_value:
            tobedeleted.append(key)
            
            
    [dict_shift_value_cp.pop(key, None) for key in tobedeleted]
    

    plt.figure()
    hist, bins = np.histogram(list_shift_value, bins=1000)
    plt.bar(bins[:-1], hist, 0.001)
    plt.savefig(report_path + 'without_rejection.png')

    plt.figure()
    hist, bins = np.histogram(list_shift_value[ind], bins=100)
    plt.bar(bins[:-1], hist, 0.002)
    plt.savefig(report_path + 'after_rejection.png')

    shift = np.median(list_shift_value[ind])
    minM, minN, lenM, lenN = get_range_from_fn_list(list_shift_img.keys(), r,x_offset,y_offset)
    nonvalue = -999.0
    
#    generate_diff_image(list_shift_img, lenM, lenN, minM, minN,
#                        1/res_ref, nonvalue, report_path,
#                        r, x_offset,y_offset, res_ref)

    img = np.zeros((lenN, lenM))
    for fn in list_shift_img.keys():
        x,y = read_fn(fn[:17], r, x_offset, y_offset)
        img[lenN -1  - (y-minN)/r, (x-minM)/r] = dict_shift_value[fn] - shift

    plot_img(img)
    plt.savefig(report_path + 'without_rejection_distribution.png')

    img = np.zeros((lenN, lenM))

    for fn in list_shift_img.keys():
        if fn in dict_shift_value_cp:
            x,y = read_fn(fn[:17], r, x_offset, y_offset)
            img[lenN -1  - (y-minN)/r, (x-minM)/r] = dict_shift_value_cp[fn] - shift


    plot_img(img)
    plt.savefig(report_path + 'after_rejection_distribution.png')

    with open(report_path + "Output.txt", "w") as text_file:
        text_file.write("shift: %s \n" % shift)
        text_file.write("std: %s \n" % np.std(list_shift_value[ind] - shift))
        text_file.write("max: %s \n" % np.max(list_shift_value[ind] - shift))
        text_file.write("min: %s \n" % np.min(list_shift_value[ind] - shift))

    plt.show()
    print (shift, np.std(list_shift_value[ind] - shift))
    print (np.max(list_shift_value[ind] - shift), np.min(list_shift_value[ind] - shift))
    
    return shift
