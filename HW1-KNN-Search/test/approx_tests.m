clc, clearvars, close all;

C = rand(10000000, 2);
Q = rand(1, 2);
K = int32(3);

% Create directory if it doesn't exist
if ~exist('approx_tests', 'dir')
    mkdir('approx_tests');
end

[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('approx_tests/test01.mat', 'C', 'Q', 'K', 'D', 'IDX');

disp('Approx test files generated successfully');
