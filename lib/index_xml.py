import xml.etree.ElementTree as ET


#return a tuple (fields, geometry, cells)
#fields is a list of tuples (fieldname, offset, codec), e.g. ("x", 0, F32)
#geometry is a tuple (cell-size, origin-x, origin-y)
#cells is a list of tuples (index-x, index-y, record_count)
def read_index_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    #read fields
    field_node = root.find("fields")
    fields = []
    possible_fields = ["x", "y", "z", "time", "reflectance", "run-id"]

    for p in possible_fields:
        p_node = field_node.find(p)
        if p_node is None:
            continue

        offset = int(p_node.find("offset").text)
        codec = p_node.find("codec").text
        fields.append((p, offset, codec))

    #read geometry
    geometry_node = root.find("geometry")
    cell_size = int(geometry_node.find("cell-size").text)
    origin_x = int(geometry_node.find("origin-x").text)
    origin_y = int(geometry_node.find("origin-y").text)
    geometry = (cell_size, origin_x, origin_y)

    #read cells
    cells_node = root.find("cells")
    cell_nodes = cells_node.findall("cell")
    cells = []
    for c in cell_nodes:
        index_x = int(c.find("index-x").text)
        index_y = int(c.find("index-y").text)
        record_count = int(c.find("records").text)
        cells.append((index_x, index_y, record_count))
    
    return fields, geometry, cells


def get_cell_id(index_x, index_y):
    if index_x >= 0:
        x = hex(index_x)[2:]
        x = x.rjust(8, '0')
    else:
        x = hex(index_x + 16**8)[2:-1]

    if index_y >= 0:
        y = hex(index_y)[2:]
        y = y.rjust(8, '0')
    else:
        y = hex(index_y + 16**8)[2:-1]

    return x + "_" + y


def get_xyz_runid_fields_indices(fields):
    #indices of certain fields in the point-tuples read from the dat-files
    fields.sort(key = lambda field: field[1])
    ix = -1
    iy = -1
    iz = -1
    irunid = -1

    for i in xrange(len(fields)):
        if fields[i][0] == "x":
            ix = i
        elif fields[i][0] == "y":
            iy = i
        elif fields[i][0] == "z":
            iz = i
        elif fields[i][0] == "run-id":
            irunid = i

    return (ix, iy, iz, irunid)
