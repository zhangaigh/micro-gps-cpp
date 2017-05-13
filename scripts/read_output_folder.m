clear

output_folder_path = '/Users/lgzhang/Dropbox/Research/micro_gps/code/mgps-cpp-v2/bin_mac/test_results/output-vw-equad_unloading_long_packed-fine_asphalt_vw.bin-sequence_after_thxgiving_normal2.test-db50-dim128';

% x_thresh = 30;
% y_thresh = 30;
% angle_thresh = 1.5;

x_thresh = 999;
y_thresh = 999;
angle_thresh = 999;

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
  fprintf('success rate = %d / %d = %f%% \n', sum(success_flag), length(success_flag), sum(success_flag) / length(success_flag) * 100);

end