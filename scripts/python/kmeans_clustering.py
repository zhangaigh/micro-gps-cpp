from __future__ import print_function
import sklearn.cluster
import numpy as np
import myutil


import precomputed_sift

sift_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed/fc_hallway_long_packed/precomputed_features/database.sift'


data_params = dict()
data_params['sift_root'] = sift_root
sift_loader = precomputed_sift.PrecomputedSift(data_params)


all_desc = sift_loader.get_all_descriptors()

myutil.save_obj(all_desc, 'all_desc.pkl')

all_desc = myutil.load_obj('all_desc.pkl')

kmeans = sklearn.cluster.KMeans(n_clusters=6553, verbose=1, n_jobs=-1).fit(all_desc[0:10000, :])

kmeans.labels