# mgps-cpp
micro-gps project c++ implementation

What do we need?
- SIFT feature extraction (we will use siftgpu by wu changchang)
- kNN search (FLANN)
- random sampling
- SIFT feature matching
- RANSAC alignment
- PCA dimension reduction

Offline:
gather descriptors
compute global poses
construct grids (fixed size)

Online:
match descriptors
vote image locations


----------------
2017-04-17 v2

Goals:
- better gui.cpp. the exe should support:
  - GUI with controls
  - GUI with parameters passed by command line
  - pure command line

- automatically use pre-extracted sift feature