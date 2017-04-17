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
