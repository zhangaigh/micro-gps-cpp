function file_list = get_file_list(dir_path, pattern, filename_only)

file_list = dir(fullfile(dir_path, pattern));
file_list = {file_list.name}';

if ~filename_only
    file_list = fullfile(repmat({dir_path}, [length(file_list) 1]), file_list);
end

end