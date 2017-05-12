function num_samples = compute_num_samples2 (num_images, num_dimensions, micro_gps_num_samples, num_visual_vocabs)

% 4 * 4 bytes for location + scale
micro_gps_storage = num_images * micro_gps_num_samples * (num_dimensions + 4) * 4;

fprintf('%f mb\n', micro_gps_storage / (1024 * 1024));

visual_vocab_storage = num_visual_vocabs * 128 * 4;
% visual_vocab_storage = 0;

% 3 * 4 bytes for location, 2 bytes for vw idx
size_image_retrieval_feat = 3 * 4 + 2;

visual_vocab_storage = 0;

num_samples = ceil((micro_gps_storage - visual_vocab_storage) / size_image_retrieval_feat / num_images);

