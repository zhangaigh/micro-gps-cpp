import struct
import numpy as np
import os
import os.path
import math
# from os import listdir
# from os.path import isfile, join

class PrecomputedSift(object):
    def __init__(self, data_params):
        self.sift_root = data_params['sift_root']

        self.file_list = []
        for f in os.listdir(self.sift_root):
            f = os.path.join(self.sift_root, f)
            if os.path.isfile(f):
                self.file_list.append(f)


    def read_sift_bin(self, sift_bin_path):
        with open(sift_bin_path, "rb") as f:
            d = struct.unpack('i'*2, f.read(4*2))
            print('{}: sift dimension = {} x {}'.format(sift_bin_path, d[0], d[1]))
            dim = d[1]
            num = d[0]

            d = struct.unpack('f' * (dim + 4) * num, f.read(num * (dim + 4) * 4))

            d = np.array(d)

            d = d.reshape((num, dim + 4))

            loc = d[:, 0:4]
            des = d[:, 4:]

        return des, loc

    def read_sift_bin_size(self, sift_bin_path):
        with open(sift_bin_path, "rb") as f:
            d = struct.unpack('i'*2, f.read(4*2))

        return d[0], d[1]

    def get_all_descriptors(self):
        num = 0
        dim = 0
        for f in self.file_list:
            n, d = self.read_sift_bin_size(f)
            num = num + n
            dim = max(d, dim)

        all_desc = np.zeros((num, dim))

        # print(all_desc.shape)
        s = 0
        for f in self.file_list:
            des, loc = self.read_sift_bin(f)
            all_desc[s:s+des.shape[0], :] = des
            s = s + des.shape[0]

        return all_desc




