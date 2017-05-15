function read_sift_generate_others(dataset_name, image_sequence_name, mode)

if ismac
  dataset_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed';
  precomputed_features_root = '../bin_mac/features';
else
  dataset_root = '/data/linguang/micro_gps_packed';
  precomputed_features_root = '/data/linguang/features';
end

% dataset_name = 'equad_unloading_long_packed';
% dataset_name = 'cs4_hallway_long_packed';

%image_sequence_name = 'test00.test';
% image_sequence_name = 'database.txt';

% modes
% 1: do surf
% 2: output packets for deep-desc
% 3. read deep-desc output
% 4. output sift
% mode = 1;


precomputed_sift_name = sprintf('%s-%s-sift', dataset_name, image_sequence_name)

% output_feature_name = strrep(precomputed_sift_name, '-sift', '-surf');



switch mode
  case 1
    output_feature_name = strrep(precomputed_sift_name, '-sift', '-surf');
  case {2 3}
    output_feature_name = strrep(precomputed_sift_name, '-sift', '-deep');
  case 4
    output_feature_name = strrep(precomputed_sift_name, '-sift', '-sift2');
end

mkdir(fullfile(precomputed_features_root, output_feature_name));

dataset_path = fullfile(dataset_root, dataset_name);
if strcmp(image_sequence_name, 'database.txt')
  has_pose = 1;
else
  has_pose = 0;
end
[image_list, ~] = read_image_list(dataset_path, image_sequence_name, has_pose);
sift_features = read_precomputed_sift(fullfile(precomputed_features_root, precomputed_sift_name));

if mode == 2
  for im_idx = 1 : length(sift_features)
    im = imread(image_list{im_idx});
    if (size(im, 3) == 3) 
      im = rgb2gray(im);
    end
    im = double(im);
    im = imresize(im, 0.5);
    fprintf('extracting patches from %d-th frame\n', im_idx-1);
    sift_features(im_idx).loc([1 2 3], :) = sift_features(im_idx).loc([1 2 3], :) * 0.5;
    [patches, sift_loc_valid] = crop_patches(im, sift_features(im_idx).loc, 64);
    fprintf('extracted %d patches\n', size(patches, 2));
    patches = reshape(patches, 64, 64, 1, []);
    patches_output = fullfile(precomputed_features_root, output_feature_name, sprintf('patches%06d.mat', im_idx-1));
    save(patches_output, 'patches', 'sift_loc_valid');
  end
end

if mode == 3
  for im_idx = 1 : length(sift_features)
    fprintf('reading deep descriptors from %d-th frame\n', im_idx-1);
    patches_output = fullfile(precomputed_features_root, output_feature_name, sprintf('patches%06d.mat', im_idx-1));
    desc_output = fullfile(precomputed_features_root, output_feature_name, sprintf('output_desc%06d.mat', im_idx-1));
    load(desc_output);
    load(patches_output);
    datachunk = [sift_loc_valid; x];

    fid = fopen(fullfile(precomputed_features_root, output_feature_name, sprintf('frame%06d.bin', im_idx-1)),  'w');
    fwrite(fid, size(datachunk, 2), 'int');
    fwrite(fid, size(datachunk, 1) - 4, 'int');
    fwrite(fid, datachunk, 'float');
    fclose(fid);
  end
end

if mode == 1
  for im_idx = 1 : length(sift_features)

    im = imread(image_list{im_idx});
    if (size(im, 3) == 3) 
      im = rgb2gray(im);
    end
    im = imresize(im, 0.5);

    surf_keypoints = SURFPoints(sift_features(im_idx).loc(1:2, :)' * 0.5);
    surf_keypoints.Scale = sift_features(im_idx).loc(3, :)' * 0.5;
    surf_keypoints.Orientation = sift_features(im_idx).loc(4, :)';

    start_time = tic;
    [features, valid_points] = extractFeatures(im, surf_keypoints ,'Method', 'SURF', 'SURFSize', 128);
    fprintf('%d / %d - surf extraction: %f ms\n', im_idx, length(sift_features), toc(start_time));
    datachunk = [valid_points.Location'; valid_points.Scale'; valid_points.Orientation'; features'];

    % TODO: save datachunk
    fid = fopen(fullfile(precomputed_features_root, output_feature_name, sprintf('frame%06d.bin', im_idx-1)),  'w');
    fwrite(fid, size(datachunk, 2), 'int');
    fwrite(fid, size(datachunk, 1) - 4, 'int');
    fwrite(fid, datachunk, 'float');
    fclose(fid);
  end
end

if mode == 4
  for im_idx = 1 : length(sift_features)
    sift_features(im_idx).loc([1 2 3], :) = sift_features(im_idx).loc([1 2 3], :) * 0.5;
    datachunk = [sift_features(im_idx).loc; sift_features(im_idx).des];
    fid = fopen(fullfile(precomputed_features_root, output_feature_name, sprintf('frame%06d.bin', im_idx-1)),  'w');
    fwrite(fid, size(datachunk, 2), 'int');
    fwrite(fid, size(datachunk, 1) - 4, 'int');
    fwrite(fid, datachunk, 'float');
    fclose(fid);
  end
end


if mode == 1 || mode == 3 || mode == 4
  fid = fopen(fullfile(precomputed_features_root, [output_feature_name '.txt']),  'w');
  for im_idx = 1 : length(sift_features)
    fprintf(fid, 'features/%s/frame%06d.bin\n', output_feature_name, im_idx-1);
  end
  fclose(fid);
end
