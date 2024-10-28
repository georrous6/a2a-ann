clc, clearvars, close all;

M = 1000; % the queries size
N = 1000; % the corpus size

L = 10:10:500;
matlab_execution_times = zeros(1, length(L));

% Create directory if it doesn't exist
if ~exist('performance_tests', 'dir')
    mkdir('performance_tests');
end

for i = 1:length(L)
    C = rand(N, L(i));
    Q = rand(M, L(i));
    K = int32(N);

    tic;
    [IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
    matlab_execution_times(i) = toc;

    if i < 10
        save(sprintf('performance_tests/test0%d.mat', i), 'C', 'Q', 'K');
    else
        save(sprintf('performance_tests/test%d.mat', i), 'C', 'Q', 'K');
    end
end

save('matlab_execution_times.mat', 'matlab_execution_times', 'L');
disp('Performance test files generated successfully');
