function read_sift_extract_surf(dataset_path, sequence_name)

if strcmp(sequence_name, 'database.txt')
  sift_output_prefix = 'database';
  has_pose = 1;
else
  sift_output_prefix = sequence_name;
  has_pose = 0;
end

[image_list, ~] = read_image_list(dataset_path, sequence_name, has_pose);

sift_features = read_precomputed_sift (fullfile(dataset_path, 'precomputed_features', [sift_output_prefix, '.sift2']));

surf_output_path = fullfile(dataset_path, 'precomputed_features', [sift_output_prefix, '.surf']);
if ~exist(surf_output_path, 'file')
  mkdir(surf_output_path);
end

for im_idx = 1 : length(sift_features)
  im = imread(image_list{im_idx});
  if (size(im, 3) == 3) 
    im = rgb2gray(im);
  end
  im = imresize(im, 0.5);

  sift_feat = sift_features(im_idx);

  % [~, sel_idx, ~] = unique(sift_feat.loc(1:2, :)', 'rows');
  % sift_feat = sift_feat[:, sel_idx];

  surf_keypoints = SURFPoints(sift_feat.loc(1:2, :)' * 0.5);
  surf_keypoints.Scale = sift_feat.loc(3, :)' * 0.5;
  surf_keypoints.Orientation = sift_feat.loc(4, :)';

  start_time = tic;
  [features, valid_points] = extractFeatures(im, surf_keypoints ,'Method', 'SURF', 'SURFSize', 128);
  fprintf('%d / %d - surf extraction: %f ms\n', im_idx, length(sift_features), toc(start_time)*1000);

  % [~, invalid_idx] = setdiff(sift_feat.loc(1:2, :)'*0.5, valid_points.Location, 'rows');
  % sift_feat[:, invalid_idx] = [];

  datachunk = [valid_points.Location' * 2.0; valid_points.Scale' * 2.0; -valid_points.Orientation'; features'];

  % TODO: save datachunk
  fid = fopen(fullfile(surf_output_path, sprintf('frame%06d.bin', im_idx-1)),  'w');
  fwrite(fid, size(datachunk, 2), 'int');
  fwrite(fid, size(datachunk, 1) - 4, 'int');
  fwrite(fid, datachunk, 'float');
  fclose(fid);
end
