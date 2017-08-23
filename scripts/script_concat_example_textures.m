clear
example_texture_path = { ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/fc_tiles/sample_frame.jpg', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/cs4_carpet/sample_frame.png', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/fields_wood/sample_frame.png', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/acee_tiles/sample_frame.jpg', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/equad_asphalt/sample_frame.png', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/equad_asphalt_controlled_light/sample_frame.png' ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/acee_asphalt/sample_frame.png', ...
                        '/Users/lgzhang/Google Drive/siggraph2017_microgps/acee_concrete/sample_frame.png', ...
                        };



cropped_w = 320;
cropped_h = 240;
spacing_width = 4.0;

concat_img = [];

for idx = 1 : length(example_texture_path)
  im_path = example_texture_path{idx};
  im = imread(im_path);
  if size(im, 1) > size(im, 2) 
    im = imrotate(im, 90);
  end

  if size(im, 3) ~= 3
    im = repmat(im, [1, 1, 3]);
  end

  x_rng = size(im, 2) / 2 - cropped_w / 2 + 1 : size(im, 2) / 2 + cropped_w / 2;
  y_rng = size(im, 1) / 2 - cropped_h / 2 + 1 : size(im, 1) / 2 + cropped_h / 2;

  im = im(y_rng, x_rng, :);

  concat_img = cat(2, concat_img, im);
  
  if idx ~= length(example_texture_path)
    concat_img = cat(2, concat_img, uint8(ones(cropped_h, spacing_width, 3) * 255));
  end
end

close all
imshow(concat_img);


imwrite(concat_img, 'concat_sample_textures.png');