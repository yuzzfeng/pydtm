# pydtm

A library which aims to convert the mobile mapping point cloud to Digital Terrain Model(DTM)

The whole process can be understood as ground filtering on Mobile Mapping Data (LiDAR). The library offers also the possibility to use an given DTM with low resolusion as a prior to generate a mix of low and high resolusion DTM.

The processing support distributed computing using python multiprocessing units.

Input of the data should be partitioned into grids based on a global origin and grid size. The number of each grid in x and y direction are then calculated in hex number in 8 digit, e.g. ffffffff_00000001. This step make the file name contions global coordinates.



