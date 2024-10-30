clc, clearvars, close all;

% some edge case tests
C = 7.2;
Q = 3.7;
K = int32(1);

% Create directory if it doesn't exist
if ~exist('validity_tests', 'dir')
    mkdir('validity_tests');
end

[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('validity_tests/test01.mat', 'C', 'Q', 'K', 'D', 'IDX');

C = [4.7, 5.2, 4.9; 0, 1.1, 2; 2.4, 6.7, 3.3];
Q = [3.7, 1.2, 4.6];
K = int32(2);

[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('validity_tests/test02.mat', 'C', 'Q', 'K', 'D', 'IDX');

MAX_SIZE = 1000;
NTESTS = 50;
for i = 3:NTESTS
    M = randi(MAX_SIZE);
    N = randi(MAX_SIZE);
    L = randi(MAX_SIZE);
    C = rand(N, L);
    Q = rand(M, L);
    K = int32(randi(N));

    [IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

    % Convert IDX to int32 before saving
    IDX = int32(IDX);
    if i < 10
        save(sprintf('validity_tests/test0%d.mat', i), 'C', 'Q', 'K', 'D', 'IDX');
    else
        save(sprintf('validity_tests/test%d.mat', i), 'C', 'Q', 'K', 'D', 'IDX');
    end
end

%% Test over very large data
Q = rand(100, 2);
C = rand(10000000, 2);
K = int32(10);
tic;
[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);
ellapsed_time = toc;
fprintf("Ellapsed time: %f sec\n", ellapsed_time);
IDX = int32(IDX);
save('validity_tests/test51.mat', 'C', 'Q', 'K', 'D', 'IDX');

disp('Validity test files generated successfully');
