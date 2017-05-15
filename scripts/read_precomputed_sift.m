function sift_features = read_precomputed_sift (precomputed_sift_path)

file_list = get_file_list(precomputed_sift_path, '*.bin', 0);

for i = 1 : length(file_list)
  fid = fopen(file_list{i}, 'r');
  num_features = fread(fid, 1, 'int');
  feature_dim = fread(fid, 1, 'int');
  fprintf('%s: %d x %d\n', file_list{i}, num_features, feature_dim);

  sift_features(i).loc = zeros(4, num_features);
  sift_features(i).des = zeros(feature_dim, num_features);

  data = fread(fid, (4+128)*num_features, 'float');
  data = reshape(data, 4+128, []);

  sift_features(i).loc = data(1:4, :); % x, y, scale, orientation
  sift_features(i).des = data(5:end, :);

  fclose(fid);  
end



end