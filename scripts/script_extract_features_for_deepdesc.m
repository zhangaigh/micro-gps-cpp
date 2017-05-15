clear

if ismac
  micro_gps_data_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed';
else
  micro_gps_data_root = '/data/linguang/micro_gps_packed';
end



i = 1;
% dataset_info(i).dataset = 'acee_asphalt_long_packed';
% dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal.test', ...
%                                'sequence_after_thxgiving_normal2.test'};
% i = i + 1;

dataset_info(i).dataset = 'equad_unloading_long_packed';
% dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal.test', ...
%                                   'sequence_after_thxgiving_normal2.test'};
dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal2.test'};
i = i + 1;

% dataset_info(i).dataset = 'acee_entrance_packed';
% dataset_info(i).test_sequences = {'test00.test', ...
%                                 'test01.test', ...
%                                 'test02.test'};
% i = i + 1;

dataset_info(i).dataset = 'cs4_hallway_long_packed';
% dataset_info(i).test_sequences = {'sequence161208_normal.test', ...
%                                   'sequence161208_normal2.test'};
dataset_info(i).test_sequences = {'sequence161208_normal.test'};
i = i + 1;

% dataset_info(i).dataset = 'equad_unloading_controlled_light_long_packed';
% dataset_info(i).test_sequences = {'sequence_controlled_light_normal.test', ...
%                                'sequence_controlled_light_normal2.test', ...
%                                'sequence_controlled_light_normal3.test'}; 
% i = i + 1;

% dataset_info(i).dataset = 'acee_unloading_long_packed';
% dataset_info(i).test_sequences = {'sequence161208_normal.test', ...
%                                'sequence161208_normal2.test'};
% i = i + 1;

% dataset_info(i).dataset = 'fc_hallway_long_packed';
% dataset_info(i).test_sequences = {'test00.test', ...
%                                'test01.test', ...
%                                'test02.test'};
% i = i + 1;

% dataset_info(i).dataset = 'fields_wood_recapture_long_packed';
% dataset_info(i).test_sequences = {'sequence0000.test'};

% i = i + 1;



file_content = [];
file_content = [file_content, '#!/bin/bash\n'];


for ds_idx = 1 : length(dataset_info)
  dataset_name = dataset_info(ds_idx).dataset;

  precomputed_features_dir = fullfile(micro_gps_data_root, dataset_name, 'precomputed_features');
  if ~exist(precomputed_features_dir, 'file')
    file_content = [file_content, sprintf('mkdir %s\n', precomputed_features_dir)];
  end

  sift_output_folder_path = fullfile(precomputed_features_dir, 'database.sift2');
  if ~exist(sift_output_folder_path, 'file')
    cmd = sprintf('./batch_extract_sift --dataset %s --output %s --rand_samples 50 --margin_thresh 91\n', dataset_name, sift_output_folder_path);
    file_content = [file_content, cmd, '\n'];
  end
  % crop patches
  patch_output_folder_path = [sift_output_folder_path, '.patches'];
  if ~exist(patch_output_folder_path, 'file')
    cmd = sprintf('./batch_crop_sift_patches --dataset %s --output %s --feat_suffix sift2 --margin_thresh 91\n', dataset_name, patch_output_folder_path);    
    file_content = [file_content, cmd, '\n'];
  end


  for ts_idx = 1 : length(dataset_info(ds_idx).test_sequences)
    test_sequence_name = dataset_info(ds_idx).test_sequences{ts_idx};
    sift_output_folder_path = fullfile(precomputed_features_dir, [test_sequence_name, '.sift2']);
    if ~exist(sift_output_folder_path, 'file')
      cmd = sprintf('./batch_extract_sift --dataset %s --testset %s --output %s --margin_thresh 91\n', dataset_name, test_sequence_name, sift_output_folder_path);
      file_content = [file_content, cmd, '\n'];
    end
    patch_output_folder_path = [sift_output_folder_path, '.patches'];
    % crop patches
    if ~exist(patch_output_folder_path, 'file')
      cmd = sprintf('./batch_crop_sift_patches --dataset %s --testset %s --output %s --feat_suffix sift2 --margin_thresh 91\n', dataset_name, test_sequence_name, patch_output_folder_path);    
      file_content = [file_content, cmd, '\n'];
    end
  end
end


if 1
  if ismac
    fid = fopen('../bin_mac/script_extract_feature_deepdesc.sh', 'w');
  else
    fid = fopen('../bin/script_extract_feature_deepdesc.sh', 'w');
  end

  fprintf(fid, file_content);

  fclose(fid);
end


