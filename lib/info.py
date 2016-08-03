from checkRunids import dictRunids
import os.path

# Kaffeezimmer IKG
x_offset = 548495
y_offset = 5804458
r = 25

# Input args
in_dir_mms = ["C:\\_EVUS_DGM\\Output\\20151202_ricklingen_3_Kacheln - ply\\",
              "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_4_Kacheln - ply\\",
              "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_6_Kacheln - ply\\",
              "C:\\_EVUS_DGM\\Output\\20151202_ricklingen_8_Kacheln - ply\\"]

# Calculate the index for each rinid
dRunids = dictRunids(in_dir_mms)


# Reference Data Path
in_dir_ref = 'C:\\_EVUS_DGM\\DEM_2009_UTM_Zone_32_Ricklingen\\DEM_kacheln\\'

# Reference DTM DTM_2009 0.5m Resolusion
list_ref = os.listdir(in_dir_ref)

# Reference resolusion
res_ref = 0.5
scale = 1/res_ref # 1m / 0.5m cell

# Raster size of one cell in pixel unit
raster_size = int(r/res_ref)

# nonvalue
nonvalue = -999.0

# cell_radius
c = [1,1,1]

# Guess of geoid from EGM2008
geoid = 42.9664
