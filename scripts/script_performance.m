clear

if ismac
  micro_gps_data_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed';
else
  micro_gps_data_root = '/data/linguang/micro_gps_packed';
end

i = 1;
dataset_info(i).dataset = 'acee_asphalt_long_packed';
dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal.test', ...
                                  'sequence_after_thxgiving_normal2.test'};
i = i + 1;

dataset_info(i).dataset = 'equad_unloading_long_packed';
dataset_info(i).test_sequences = {'sequence_after_thxgiving_normal.test', ...
                                  'sequence_after_thxgiving_normal2.test'};
i = i + 1;

dataset_info(i).dataset = 'acee_entrance_packed';
dataset_info(i).test_sequences = {'test00.test', ...
                                  'test01.test', ...
                                  'test02.test'};
i = i + 1;

dataset_info(i).dataset = 'cs4_hallway_long_packed';
dataset_info(i).test_sequences = {'sequence161208_normal.test', ...
                                  'sequence161208_normal2.test'};
i = i + 1;

dataset_info(i).dataset = 'equad_unloading_controlled_light_long_packed';
dataset_info(i).test_sequences = {'sequence_controlled_light_normal.test', ...
                                  'sequence_controlled_light_normal2.test', ...
                                  'sequence_controlled_light_normal3.test'}; 
i = i + 1;

dataset_info(i).dataset = 'acee_unloading_long_packed';
dataset_info(i).test_sequences = {'sequence161208_normal.test', ...
                                  'sequence161208_normal2.test'};
i = i + 1;

dataset_info(i).dataset = 'fc_hallway_long_packed';
dataset_info(i).test_sequences = {'test00.test', ...
                                  'test01.test', ...
                                  'test02.test'};
i = i + 1;

dataset_info(i).dataset = 'fields_wood_recapture_long_packed';
dataset_info(i).test_sequences = {'sequence0000.test'};
dataset_info(i).db_sample_size = 100;
i = i + 1;

dataset_info(i).dataset = 'fields_wood_recapture_long_packed';
dataset_info(i).test_sequences = {'sequence0000.test'};
dataset_info(i).db_sample_size = 50;
i = i + 1;

if ismac
  fid = fopen('../bin_mac/script_performance.sh', 'w');
else
  fid = fopen('../bin/script_performance.sh', 'w');
end
fprintf(fid, '#!/bin/bash\n');


dimensionality_array = [8 16];


for dimensionality_idx = 1 : length(dimensionality_array)
  dimensionality = dimensionality_array(dimensionality_idx);

  for ds_idx = 1 : length(dataset_info)
    curr_dataset_info = dataset_info(ds_idx);
    dataset_name = curr_dataset_info.dataset;

    if ~isfield(curr_dataset_info, 'db_sample_size') || isempty(curr_dataset_info.db_sample_size)
      db_sample_size = 50;
    else
      db_sample_size = curr_dataset_info.db_sample_size;
    end

    feature_database_name = sprintf('%s-siftgpu.bin', dataset_name);
    pca_basis_name = sprintf('pca_%s-siftgpu.bin', dataset_name);

    for ts_idx = 1 : length(curr_dataset_info.test_sequences)
      testset_name = curr_dataset_info.test_sequences{ts_idx};
      output_folder = sprintf('output-performance-%s-%s-db-%d-dim%d', dataset_name, testset_name, db_sample_size, dimensionality);

% DEFINE_string (dataset_root,      "/Users/lgzhang/Documents/DATA/micro_gps_packed",   "dataset_root");
% DEFINE_string (dataset,           "fc_hallway_long_packed",                           "dataset to use");
% DEFINE_string (testset,           "test00.test",                                      "test sequence");
% DEFINE_string (output,            "tests",                                            "output");
% DEFINE_string (feature_db,        "tiles50-siftgpu.bin",                              "database features");
% DEFINE_string (pca_basis,         "pca_tiles50-siftgpu.bin",                          "pca basis to use");
% DEFINE_string (vw,                "",                                                 "visual words to use");

% DEFINE_double (cell_size,         50.0f,                                              "size of the voting cell");
% DEFINE_int32  (num_scale_groups,  10,                                                 "number of search indexes");
% DEFINE_int32  (feat_dim,          8,                                                  "dimensionality after PCA reduction");
% DEFINE_int32  (best_knn,          9999,                                               "use the best k nearest neighbors for voting");
% DEFINE_double (sift_ext_scale,    0.5,                                                "extract sift at this scale");
% DEFINE_int32  (frames_to_test,    9999999,                                            "max number of frames to test");
% // offline
% DEFINE_int32  (db_sample_size,    50,                                                 "number of features sampled from each database image");
% DEFINE_string (feat_suffix,       "sift",                                             "default suffix for precomputed feature");
      
      cmd = './micro_gps_exec ';
      cmd = [cmd, ' ', sprintf('--dataset %s',          dataset_name)];
      cmd = [cmd, ' ', sprintf('--testset %s',          testset_name)];
      cmd = [cmd, ' ', sprintf('--output %s',           output_folder)];
      cmd = [cmd, ' ', sprintf('--feature_db %s',       feature_database_name)];
      cmd = [cmd, ' ', sprintf('--pca_basis %s',        pca_basis_name)];
      % cmd = [cmd, ' ', sprintf('--vw %s',               x)];

      cmd = [cmd, ' ', sprintf('--cell_size %.2f',      50.0)];
      cmd = [cmd, ' ', sprintf('--num_scale_groups %d', 10)];
      cmd = [cmd, ' ', sprintf('--feat_dim %d',         dimensionality)];
      cmd = [cmd, ' ', sprintf('--best_knn %d',         9999)];
      cmd = [cmd, ' ', sprintf('--sift_ext_scale %.2f', 0.5)];
      cmd = [cmd, ' ', sprintf('--frames_to_test %d',   9999999)];

      cmd = [cmd, ' ', sprintf('--db_sample_size %d',   db_sample_size)];
      cmd = [cmd, ' ', sprintf('--feat_suffix %s',      'sift')];
      

      fprintf(fid, sprintf('%s\n', cmd));

    end

  end

end


fclose(fid);


