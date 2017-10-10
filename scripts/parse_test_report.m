% ----- Timing info -----
% Total: 127.569000 ms
% Feature Extraction: 65.960000 ms
% NN Search: 48.587000 ms
% Compute Candidate Poses: 0.228000 ms
% Voting: 1.907000 ms
% RANSAC: 5.658000 ms
% ----- Result -----
% Estimated Pose: 0.780943 -0.624602 -6431.669434 0.624602 0.780943 35474.726562 0.000000 0.000000 1.000000
% Success: 1
% Top cells: 10 8 7 3 2 2 2 2 2 2 
% SIFTMatch Estimated Pose: 0.780240 -0.625480 -6430.858398 0.625480 0.780240 35474.847656 0.000000 0.000000 1.000000
% x_error = -0.811035, y_error = -0.121094, angle_error = 0.064415
% ----- Debug info -----
% Test image path: /Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed/sequence161208_normal/frame00005160-1481226409333457.pgm
% Closest database image path: /Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed/database/frame001259.pgm
% Grid step: 50.000000
% Peak top-left x: -6429.804688
% Peak top-left y: 35435.167969



% clear
% file_path = '../bin_mac/output-acee_asphalt_long_packed-acee_asphalt_siftgpu.bin-test00.test-dim2/frame000000.txt';
function [timing, result, debug_info] = parse_test_report(file_path)

timing = struct;
result = struct;
debug_info = struct;

fid = fopen(file_path);

line_idx = 0;
while ~feof(fid)
  l = fgetl(fid);
  switch line_idx
    case 1
      val = sscanf(l, 'Total: %f ms\n');
      timing.total = val; 
    case 2
      val = sscanf(l, 'Feature Extraction: %f ms\n');
      timing.feature_extraction = val;
    case 3
      val = sscanf(l, 'NN Search: %f ms\n');
      timing.nn_search = val;
    case 4
      val = sscanf(l, 'Compute Candidate Poses: %f ms\n');
      timing.compute_candidate_poses = val;
    case 5
      val = sscanf(l, 'Voting: %f ms\n');
      timing.voting = val;
    case 6
      val = sscanf(l, 'RANSAC: %f ms\n');
      timing.ransac = val;
    case 8
      val = sscanf(l, ['Estimated Pose:' repmat(' %f', 1, 9) '\n']);
      val = reshape(val, 3, 3)';
      result.estimated_pose = val;
    case 9
      val = sscanf(l, 'Success: %d\n');
      result.success = val;
    case 10
      val = sscanf(l, ['Top cells:' repmat(' %d', 1, 10) '\n']);
      result.top_cells = val;      
    case 11
      val = sscanf(l, ['SIFTMatch Estimated Pose:' repmat(' %f', 1, 9) '\n']);
      val = reshape(val, 3, 3)';
      result.verified_pose = val;
    case 12
      val = sscanf(l, 'x_error = %f, y_error = %f, angle_error = %f');
      result.x_y_angle_error = val;
    case 14
      val = sscanf(l, 'Test image path: %s');
      debug_info.test_image_path = val;
    case 15
      val = sscanf(l, 'Closest database image path :%s');
      debug_info.closest_database_image_path = val;
  end

  line_idx = line_idx + 1;
end

fclose(fid);