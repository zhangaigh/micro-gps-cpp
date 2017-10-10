clear
close all

% map_path = '/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/bin_mac/maps/fc_hallway_pg_10per.png';
map_path = '/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/bin_mac/maps/asphalt2_map_10per.png';

im = imread(map_path);

angle = 13.5;
im_rotated = imrotate(im, angle);
imshow(im_rotated);