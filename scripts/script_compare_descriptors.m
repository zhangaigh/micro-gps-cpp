clear
addpath(genpath('/Users/lgzhang/Dropbox/Research/third_party_libs/export_fig'))

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

dimensionality_array = [2 4 8 16 32 64 128];
feature_type_array = {'sift2', 'deep', 'surf'};
% feature_type_array = {'surf'};
feature_type_official_name_array = {'SIFT', 'DeepDesc', 'SURF'};

x_thresh = 30;
y_thresh = 30;
angle_thresh = 1.5;

success_rate_matrix = zeros(length(feature_type_array), length(dimensionality_array), length(dataset_info));

for dimensionality_idx = 1 : length(dimensionality_array)
  dimensionality = dimensionality_array(dimensionality_idx);

  fprintf('dimensionality = %d\n', dimensionality);

  for ftype_idx = 1 : length(feature_type_array)
    feature_suffix = feature_type_array{ftype_idx};


    for ds_idx = 1 : length(dataset_info)
      curr_dataset_info = dataset_info(ds_idx);
      dataset_name = curr_dataset_info.dataset;
      fprintf('%s\n', dataset_name);

      success_flag_cat = [];

      if ~isfield(curr_dataset_info, 'db_sample_size') || isempty(curr_dataset_info.db_sample_size)
        db_sample_size = 50;
      else
        db_sample_size = curr_dataset_info.db_sample_size;
      end

      feature_database_name = sprintf('%s-db%d-%s.bin', dataset_name, db_sample_size, feature_suffix);
      pca_basis_name = sprintf('pca_%s', feature_database_name);

      for ts_idx = 1 : length(curr_dataset_info.test_sequences)
        testset_name = curr_dataset_info.test_sequences{ts_idx};
        output_folder = sprintf('output-compare_desc-%s-%s-%s-db%d-dim%d', feature_suffix, dataset_name, testset_name, db_sample_size, dimensionality);

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
        cmd = [cmd, ' ', sprintf('--feat_suffix %s',      feature_suffix)];
        
        file_content = [file_content, cmd, '\n'];

        output_folder_path = fullfile('../bin/test_results', output_folder);
        if exist(output_folder_path, 'file')
          file_list = get_file_list(output_folder_path, 'frame*.txt', 0);

          testing_time =    zeros(1, length(file_list));
          success_flag =    zeros(1, length(file_list));
          sift_time =       zeros(1, length(file_list), 1);
          nn_search_time =  zeros(1, length(file_list));
          x_y_angle_error = zeros(3, length(file_list));
          verified_poses =  zeros(3, 3, length(file_list));

          for i = 1 : length(file_list)
            [timing, result, debug_info] = parse_test_report(file_list{i});
            testing_time(i) = timing.total;
            sift_time(i) = timing.feature_extraction;
            nn_search_time(i) = timing.nn_search;
            success_flag(i) = result.success;
            x_y_angle_error(:, i) = result.x_y_angle_error;
            verified_poses(:, :, i) = result.verified_pose;
          end

          x_y_angle_error = abs(x_y_angle_error);
          success_flag = x_y_angle_error(1, :) < x_thresh & x_y_angle_error(2, :) < y_thresh & x_y_angle_error(3, :) < angle_thresh & success_flag > 0;
          % fprintf('success rate = %d / %d = %f%% \n', sum(success_flag), length(success_flag), sum(success_flag) / length(success_flag) * 100);
          success_flag_cat = [success_flag_cat success_flag];

        end
      end
      
      success_rate = sum(success_flag_cat) / length(success_flag_cat) * 100;
      success_rate_matrix(ftype_idx, dimensionality_idx, ds_idx) = success_rate;
      fprintf('final success rate = %d / %d = %f%% \n\n', sum(success_flag_cat), length(success_flag_cat), success_rate);
  
    end
  end
end


if 0
  if ismac
    fid = fopen('../bin_mac/script_compare_descriptors.sh', 'w');
  else
    fid = fopen('../bin/script_compare_descriptors.sh', 'w');
  end

  fprintf(fid, file_content);

  fclose(fid);
end


for ds_idx = 1 : length(dataset_info)
  legend_strs = {};
  figure
  hold on
  for i = 1 : size(success_rate_matrix, 1)
    plot(success_rate_matrix(i, :, ds_idx), '--s','LineWidth',2,...
                              'MarkerSize',10,...
                              'MarkerEdgeColor','b');

    str = feature_type_official_name_array{i};
    legend_strs = [legend_strs str];
  end
  grid on
  box on
  h_legend = legend(legend_strs);
  set(h_legend, 'Position', [0.64 0.13 0.25 0.15]);
  set(gca, 'FontSize', 20);
  ax = gca;
  ax.XTick = 1 : length(dimensionality_array);
  ax.XTickLabel = dimensionality_array;
  xlabel('descriptor dimensionality');
  ylabel('success rate (%)');

  savefig(['compare_descriptors-' dataset_info(ds_idx).dataset '.fig'])
  export_fig(['compare_descriptors-' dataset_info(ds_idx).dataset '.pdf'], '-transparent');
end
  