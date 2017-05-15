function [image_list, image_poses] = read_image_list(dataset_path, image_list_txt, has_pose)

image_poses = [];

fid = fopen(fullfile(dataset_path, image_list_txt), 'r');

i = 1;
while ~feof(fid)
  l = fgetl(fid);
  im_path = sscanf(l, '%s\n');
  image_list{i} = fullfile(dataset_path, im_path);
  if has_pose
    l = fgetl(fid);
    val = sscanf(l, [repmat('%f ', 1, 9) '\n']);
    image_poses = cat(3, image_poses, reshape(val, 3, 3)');
  end
  i = i+1;
end

fclose(fid);
