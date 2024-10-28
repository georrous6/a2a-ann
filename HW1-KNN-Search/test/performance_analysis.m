clc; clearvars; close all;

% Load execution times from .mat files
my_data = load("my_execution_times.mat", "my_execution_times");
matlab_data = load("matlab_execution_times.mat", "matlab_execution_times", "L");


my_execution_times = my_data.my_execution_times;
matlab_execution_times = matlab_data.matlab_execution_times;
L = matlab_data.L;


figure;
plot(L, matlab_execution_times, 'DisplayName', 'MATLAB Execution Times');
hold on;
plot(L, my_execution_times, 'DisplayName', 'My Execution Times');
hold off;

% Add legend and labels
legend show; % This displays the legend with names set in 'DisplayName'
xlabel('Dimension');
ylabel('Execution Times (sec)');
title('Comparison of Execution Times');
