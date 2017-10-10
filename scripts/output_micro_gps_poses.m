clear

% output_folder_path = '../bin_mac/test_results/test_sequence170920_1';
output_folder_path = '../bin_mac/test_results/test_sequence170925_6';

% x_thresh = 30;
% y_thresh = 30;
% angle_thresh = 1.5;

% x_thresh = 999;
% y_thresh = 999;
% angle_thresh = 999;
 
file_list = get_file_list(output_folder_path, 'frame*.txt', 0);

% result_array = cell(length(file_list));
% debug_info_array = cell(length(file_list));
fp = fopen([output_folder_path '_poses.txt'], 'w');

traj = zeros(length(file_list), 2);

for i = 1 : length(file_list)
  [timing, result, debug_info] = parse_test_report(file_list{i});
  
  pose = result.estimated_pose';
  
  [~, test_fname, ~] = fileparts(debug_info.test_image_path);
  v = sscanf(test_fname, 'frame%d-%ld');
  ts = v(2);
   
  fprintf(fp, '%014d ', ts);
  fprintf(fp, '%d ', result.success);
  fprintf(fp, repmat('%.04f ', 1, 9), pose(1), pose(2), pose(3), ... 
                                      pose(4), pose(5), pose(6), ...
                                      pose(7), pose(8), pose(9));
  fprintf(fp, '\n');
  
  traj(i, :) = pose(3, 1:2);
end
fclose(fp);


traj((traj(:, 1) == 0 & traj(:, 2) == 0), :) = [];
plot(traj(:, 1), traj(:, 2), 'x')
axis equal
% h = gca;  % Handle to currently active axes
% set(h, 'YDir', 'reverse');



