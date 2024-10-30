clc, clearvars, close all;

M = 10; % the queries size
L = 2; % the dimension
K = int32(8);
NTESTS = 22;

N = zeros(1, NTESTS); % the corpus size
matlab_execution_times = zeros(1, NTESTS);

% Create directory if it doesn't exist
if ~exist('performance_tests', 'dir')
    mkdir('performance_tests');
end

for i = 1:NTESTS
    N(i) = 2^i;
    C = rand(N(i), L);
    Q = rand(M, L);

    tic;
    [IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
    matlab_execution_times(i) = toc;

    if i < 10
        save(sprintf('performance_tests/test0%d.mat', i), 'C', 'Q', 'K');
    else
        save(sprintf('performance_tests/test%d.mat', i), 'C', 'Q', 'K');
    end
end

save('matlab_execution_times.mat', 'matlab_execution_times', 'N', 'K', 'M', 'L');
disp('Performance test files generated successfully');
