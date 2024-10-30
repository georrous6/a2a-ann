clc; clearvars;

% Load execution times from .mat files
my_data = load('my_execution_times.mat', 'my_execution_times');
matlab_data = load('matlab_execution_times.mat', 'matlab_execution_times', 'L', 'K', 'N', 'M');


my_execution_times = my_data.my_execution_times;
matlab_execution_times = matlab_data.matlab_execution_times;
N = matlab_data.N;
L = matlab_data.L;
K = matlab_data.K;
M = matlab_data.M;


figure;
plot(N, matlab_execution_times, 'DisplayName', 'MATLAB Execution Times');
set(gca, 'XScale', 'log');
hold on;
plot(N, my_execution_times, 'DisplayName', 'My Execution Times');

% Add legend and labels
legend show; % This displays the legend with names set in 'DisplayName'
xlabel('Corpus Size');
ylabel('Execution Times (sec)');
title(sprintf('Comparison of Execution Times (M=%d, K=%d, L=%d)', M, K, L));
