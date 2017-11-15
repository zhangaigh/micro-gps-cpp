import struct
import numpy as np


def read_precomputed_sift(sift_file):
    fid = open(sift_file, mode='rb')
    num_features = struct.unpack('i', fid.read(4))[0]
    num_dimensions = struct.unpack('i', fid.read(4))[0]
    data_chunk = np.array(struct.unpack('f' * (num_dimensions + 5) * num_features, fid.read()))
    fid.close()

    data_chunk = data_chunk.reshape(-1, (num_dimensions + 5))
    print('{}: {}x{}'.format(sift_file, num_features, num_dimensions))
    loc = data_chunk[:, 0:5]
    des = data_chunk[:, 5:]

    return loc, des


# loc, des = read_precomputed_sift('/Users/lgzhang/Documents/DATA/micro_gps_packed/equad_unloading_long_packed/precomputed_features/database.quadsift/frame000100.bin')
# loc, des = read_precomputed_sift('/data/linguangzhang/micro_gps_packed/cs4_hallway_long_packed/precomputed_features/sequence161208_normal.test.key_sift0.5_ori_sift0.5_scale_deep0.5_desc_deep0.5/frame000100.bin')

# loc, des = read_precomputed_sift('/data/linguangzhang/micro_gps_packed/cs4_hallway_long_packed/precomputed_features/database.key_sift0.5_ori_sift0.5_scale_deep0.5_desc_deep0.5/frame000200.bin')

loc, des = read_precomputed_sift('/data/linguangzhang/micro_gps_packed/fields_wood_recapture_long_packed/precomputed_features/database.key_sift1.0_ori_sift1.0_scale_const6.0_desc_sift1.0/frame000017.bin')

print(loc)
print(loc.shape)
# np.set_printoptions(threshold=10)
# print(loc[:, 3])

# print(loc[:, 3].mean())
# print(loc[:, 3].max())
# print(np.median(loc[:, 3]))