import struct
import numpy as np


def read_precomputed_sift(sift_file):
    fid = open(sift_file, mode='rb')
    num_features = struct.unpack('i', fid.read(4))[0]
    num_dimensions = struct.unpack('i', fid.read(4))[0]
    data_chunk = np.array(struct.unpack('f' * (num_dimensions + 4) * num_features, fid.read()))
    fid.close()

    data_chunk = data_chunk.reshape(-1, (num_dimensions + 4))
    print('{}: {}x{}'.format(sift_file, num_features, num_dimensions))
    loc = data_chunk[:, 0:4]
    des = data_chunk[:, 4:]

    return loc, des


# loc, des = read_precomputed_sift('/Users/lgzhang/Documents/DATA/micro_gps_packed/equad_unloading_long_packed/precomputed_features/database.sift/frame000100.bin')
loc, des = read_precomputed_sift('/Users/lgzhang/Documents/DATA/micro_gps_packed/fields_wood_recapture_long_packed/precomputed_features/database.quad/frame000023.bin')

# np.set_printoptions(threshold=np.inf)
print(loc[:, 2])

print(loc[:, 2].mean())
print(loc[:, 2].max())
print(np.median(loc[:, 2]))