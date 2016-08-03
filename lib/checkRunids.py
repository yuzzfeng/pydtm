import os.path

#######################################################################
# Find the cells ids for each runid
# Author: Yu Feng
#######################################################################

def countCellsRunids(in_dir_mms):

    list_all_cells = []
    list_all_runids = []

    for dir_mms in in_dir_mms:

        list_cells = os.listdir(dir_mms)
        list_all_cells.extend(list_cells)
        [list_all_runids.extend([ runid[18:27] for runid in os.listdir(dir_mms + cells)]) for cells in list_cells]
        
    list_all_cells = list(set(list_all_cells))
    list_all_runids = list(set(list_all_runids))
    return list_all_cells, list_all_runids


def dictRunids(in_dir_mms):

    list_all_cells, list_all_runids = countCellsRunids(in_dir_mms)

    d = dict()
    for runid in list_all_runids:
        d[runid] = []

    for dir_mms in in_dir_mms:

        list_mms = os.listdir(dir_mms)

        for cell_id in list_mms:
            path_mms = dir_mms + '%s\\'%cell_id
            [ d[runid[18:27]].append(os.path.join(dir_mms, cell_id) + '\\' + runid)
              for runid in os.listdir(path_mms) if runid[-4:] == ".ply" ]

    return d


def full2cell(full):
    return full.split('\\')[-2]

def full2cellrunid(full):
    return full.split('\\')[-1]

# Check path exist otherwise creat new
def check_and_create(out_dir):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)

def list_runid_read_both_scanner(l):

    for runid in l:
	runid_left = str(int(runid) - 1)
	runid_right = str(int(runid) + 1)
	if runid_left in l:
		l.remove(runid_left)
	if runid_right in l:
		l.remove(runid_right)
    return l
