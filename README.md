# pydtm

A library which aims to convert the mobile mapping point cloud to Digital Terrain Model(DTM)

The whole process can be understood as ground filtering on Mobile Mapping Data (LiDAR). The library offers also the possibility to use an given DTM with low resolusion as a prior to generate a mix of low and high resolusion DTM.

The processing support distributed computing using python multiprocessing units.
