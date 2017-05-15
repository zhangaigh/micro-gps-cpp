clear
close all

I = imread('cameraman.tif');
corners = detectHarrisFeatures(I);
corners = corners.Location;

surf_keypoints = SURFPoints(corners);
surf_keypoints.Scale = ones(1, length(surf_keypoints))' * 1.6;

[features, valid_points] = extractFeatures(I, surf_keypoints ,'Method', 'SURF', 'SURFSize', 128);

figure; imshow(I); hold on;
valid_points(15).Orientation / pi * 180
plot(valid_points(15),'showOrientation',true);

% plot(valid_points.selectStrongest(50),'showOrientation',true);
% [features, valid_corners] = extractFeatures(I, corners);