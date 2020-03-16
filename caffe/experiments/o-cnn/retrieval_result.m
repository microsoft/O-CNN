% filename
filename_list = ''; % the data list file
filename_prob = ''; % the probability of each object
topK = 1000;

% get prob
fid = fopen(filename_prob, 'r');
N = fread(fid, 1, 'int32');
C = fread(fid, 1, 'int32');
prob = fread(fid, [C, N], 'float32');
fclose(fid);

% calculate the label
[~, label] = max(prob, [], 1);

% read data list
fid = fopen(filename_list, 'r');
tline = fgetl(fid);
label_gt = zeros(1, N);
i = 1;
items = cell(1, N);
while ischar(tline)
    p = strfind(tline, ' ');
    items{i} = tline(7:12);
    label_gt(i) = str2double(tline(p+1:end));
    i = i + 1;
    tline = fgetl(fid);
end
fclose(fid);


%% same class
for i = 1 : N
    idx = find(label == label(i));    
    dis = sum(bsxfun(@minus, prob(:, idx), prob(:, i)).^2);    
    [dis, di] = sort(dis);
    k = min(topK, length(idx));
    
    fid = fopen(['Result/S55/WANG_OCNN/test_normal/' items{i}], 'w');
    for j = 1 : k
        fprintf(fid, '%s\n', items{idx(di(j))});
    end
    fclose(fid);
end

