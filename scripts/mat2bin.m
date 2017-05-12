function mat2bin(m, f)

fid = fopen(f, 'w');

fwrite(fid, size(m, 1), 'uint64');
fwrite(fid, size(m, 2), 'uint64');

fwrite(fid, m, 'single');

fclose(fid);

end