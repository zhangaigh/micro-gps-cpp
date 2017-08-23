clear
sequence_idx = 0 : 10;
speed_array =  20: 10: 120;
actual_speed_multiplier = 4.75;

sequences_root = '/Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed/';


idx = 1;


for idx = 1 : length(sequence_idx)
  image_list = get_file_list(fullfile(sequences_root, sprintf('sequence%04d_speed%d_20', sequence_idx(idx), speed_array(idx))), '*.pgm', 0);
  image_sample_idx = 40 : 10 : 100;


  w = 320;
  h = 240;
  spacing_width = 4;
  concat_img = [];
  n_rows = 3;
  for row_idx = 1 : n_rows
    im = imread(image_list{image_sample_idx(row_idx)});
    x_rng = size(im, 2) / 2 - w / 2 + 1 : size(im, 2) / 2 + w / 2;
    y_rng = size(im, 1) / 2 - h / 2 + 1 : size(im, 1) / 2 + h / 2;

    im = im(y_rng, x_rng);

    concat_img = cat(1, concat_img, im);

    if row_idx ~= n_rows
      concat_img = cat(1, concat_img, uint8(ones(spacing_width, w) * 255));
    end
  end

  out_filename = sprintf('motion_blur_speed_%d.png', round(4.75 * speed_array(idx)));

  imwrite(concat_img, out_filename);
end

% close all
% imshow(concat_img);