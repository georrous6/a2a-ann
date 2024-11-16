%% Create test for 2D points

clc, clearvars, close all;

% Create directory if it doesn't exist
if ~exist('approx_tests', 'dir')
    mkdir('approx_tests');
end

C = rand(10000000, 2);
Q = rand(3, 2);
K = int32(10);

tic;
[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
ellapsedTime = toc;
fprintf('Ellapsed time: %d (sec)\n', ellapsedTime);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('approx_tests/test01.mat', 'C', 'Q', 'K', 'D', 'IDX');

%% Create test for 3D points

C = rand(10000, 3);
Q = rand(2, 3);
K = int32(5);

tic;
[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
ellapsedTime = toc;
fprintf('Ellapsed time: %d (sec)\n', ellapsedTime);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('approx_tests/test02.mat', 'C', 'Q', 'K', 'D', 'IDX');

disp('Approx test files generated successfully');
