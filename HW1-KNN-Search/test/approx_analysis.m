clc; clearvars;

% Load execution times from .mat files
my_data = load('approx_tests/my_output.mat', 'approx_neighbors');
matlab_data = load('approx_tests/test01.mat', 'IDX', 'Q', 'C');


approx_neighbors = my_data.approx_neighbors;
IDX = matlab_data.IDX;
Q = matlab_data.Q;
C = matlab_data.C;


figure;
scatter(C(:,1), C(:,2), 'b', 'o', 'DisplayName', 'Coprus');
hold on;
scatter(approx_neighbors(:,1), approx_neighbors(:,2), 'r', 'o', 'DisplayName', 'Approximate neighbors');
scatter(Q(:,1), Q(:,2), 'g', '+', 'DisplayName', 'Queries');

% Plot the points in C specified by IDX
knn = C(IDX(:), :); % Flatten IDX and get the exact k-nearest neighbors
scatter(knn(:,1), knn(:,2), 'y', '*', 'DisplayName', 'Exact nearest neighbors');

% Add legend and labels
legend show; % This displays the legend with names set in 'DisplayName'
xlabel('X');
ylabel('Y');
title('Approximate Nearest Neighbors');
