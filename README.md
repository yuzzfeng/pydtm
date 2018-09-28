# pydtm

A library which aims to convert the Mobile Mapping LiDAR Point Cloud to Digital Terrain Model(DTM)

![](https://github.com/fy19891009/pydtm/blob/master/img/DTM.PNG)

The whole process can be understood as ground filtering on mobile LiDAR data. The library offers also the possibility to use an given DTM (e.g. ALS) with low resolusion as a prior to generate a mix of low and high resolusion DTM.

The processing support distributed computing using python multiprocessing units.

Input of the data should be partitioned into grids based on a global origin and grid size. The number of each grid in x and y direction are then calculated in hex number in 8 digit, e.g. ffffffff_00000001. This step make the file name contains coordinates in global coordinate system.

```console
python rejectOutliers.py
python alignedDTM.py
python collectDTM.py
```


```console
@Article{isprs-annals-IV-4-W6-11-2018,
AUTHOR = {Feng, Y. and Brenner, C. and Sester, M.},
TITLE = {ENHANCING THE RESOLUTION OF URBAN DIGITAL TERRAIN MODELS USING MOBILE MAPPING SYSTEMS},
JOURNAL = {ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {IV-4/W6},
YEAR = {2018},
PAGES = {11--18},
URL = {https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-4-W6/11/2018/},
DOI = {10.5194/isprs-annals-IV-4-W6-11-2018}
}
```
