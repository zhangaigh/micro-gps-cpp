clear

if ismac
  micro_gps_data_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed';
else
  micro_gps_data_root = '/data/linguang/micro_gps_packed';
end



i = 1;
dataset_info(i).dataset = 'equad_unloading_long_packed';
dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal2.test'};
i = i + 1;

dataset_info(i).dataset = 'cs4_hallway_long_packed';
dataset_info(i).test_sequences = {'sequence161208_normal.test'};
i = i + 1;


% file_content = [];
% file_content = [file_content, '#!/bin/bash\n'];


for ds_idx = 1 : length(dataset_info)
  dataset_name = dataset_info(ds_idx).dataset;
  dataset_path = fullfile(micro_gps_data_root, dataset_name);
  read_sift_extract_surf(dataset_path, 'database.txt')



  for ts_idx = 1 : length(dataset_info(ds_idx).test_sequences)
    test_sequence_name = dataset_info(ds_idx).test_sequences{ts_idx};
    read_sift_extract_surf(dataset_path, test_sequence_name)

  end
end
