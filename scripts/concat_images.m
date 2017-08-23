close all
clear

color_table = [255 255 255; ...
               255  0   0; ...
               0   255  0; ...
               0    0  255];


% image_list = get_file_list('/Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed/sequence0010_speed120_20', '*.pgm', 0);
% cat_image.out_name = 'carpet_speed120_1x3concat.png';
% cat_image.crop_rect = [1 1 320 240];
% cat_image.image_list = image_list(40:5:100);
% cat_image.rows = 1;
% cat_image.cols = 3;
% cat_image.border_thickness = 1;
% cat_image.border_colors = [1 1 1 1];
% cat_image.im_w = 640;
% cat_image.im_h = 480;


image_list = get_file_list('/Users/lgzhang/Documents/DATA/micro_gps_packed/acee_asphalt_long_packed/sequence_after_rain', '*.pgm', 0);
cat_image.out_name = 'acee_asphalt_wet_1x3concat.png';
cat_image.crop_rect = [];
cat_image.image_list = image_list(40:5:100);
cat_image.rows = 1;
cat_image.cols = 3;
cat_image.border_thickness = 1;
cat_image.border_colors = [1 1 1 1];
cat_image.im_w = 640;
cat_image.im_h = 480;


for im_idx = 1 : cat_image.rows * cat_image.cols
  im = imread(cat_image.image_list{im_idx});
  % w = size(im, 2);
  % h = size(im, 1);
  if size(im, 3) == 1
    im = repmat(im, [1 1 3]);
  end
  if ~isempty(cat_image.crop_rect)
    im = im(cat_image.crop_rect(2):cat_image.crop_rect(4), cat_image.crop_rect(1):cat_image.crop_rect(3), :);
  end
  im = imresize(im, [cat_image.im_h, cat_image.im_w]);
  bc = color_table(cat_image.border_colors(im_idx), :);
  im_withborder = ones(size(im, 1) + cat_image.border_thickness*2, size(im, 2) + cat_image.border_thickness*2, 3);
  im_withborder(:, :, 1) = bc(1);
  im_withborder(:, :, 2) = bc(2);
  im_withborder(:, :, 3) = bc(3);

  im_withborder(cat_image.border_thickness+1 : end - cat_image.border_thickness, cat_image.border_thickness+1 : end - cat_image.border_thickness, :) = im;

  cat_image.image{im_idx} = uint8(im_withborder);
end


result = [];
idx = 1;
for i = 1 : cat_image.rows
    row_concat = [];
    for j = 1 : cat_image.cols
        row_concat = cat(2, row_concat, cat_image.image{idx});
        idx = idx + 1;
    end
    result = cat(1, result, row_concat);
end

% imshow(result);
imwrite(result, cat_image.out_name);
