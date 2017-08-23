clear
% im_path = '/Users/lgzhang/Dropbox/Research/micro_gps/micro-gps-siggraph-asia/images/footprint/footprints_cropped.png';
% im_path = '/Users/lgzhang/Dropbox/Research/micro_gps/micro-gps-siggraph-asia/images/footprint/footprints.png';
im_path = '/Users/lgzhang/Dropbox/Research/micro_gps/micro-gps-siggraph-asia/images/footprint/warped_image_sample.png';


im = imread(im_path);
imwrite(im, im_path);