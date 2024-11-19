%% Generate 2D test file
clc, clearvars, close all;

% rng(10); % Add random seed for reproducibility

Q = rand(1000000, 2);
K = int32(3);
C = Q;

[IDX, ~] = knnsearch(Q, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('approx_tests/test01.mat', 'C', 'Q', 'K', 'IDX');

%% Plot 2D points
clc, clearvars, close all;

% Load execution times from .mat files
my_data = load('approx_tests/my_output.mat', 'IDX');
matlab_data = load('approx_tests/test01.mat', 'IDX', 'Q');

IDX_approx = my_data.IDX + 1;
IDX = matlab_data.IDX;
Q = matlab_data.Q;

p=randi(size(Q, 1));

figure;
scatter(Q(:,1), Q(:,2), 'b', 'o', 'DisplayName', 'Coprus');
hold on;
approx_knn = Q(IDX_approx(p,:), :);
scatter(approx_knn(:,1), approx_knn(:,2), 'r', 'o', 'DisplayName', 'Approximate neighbors');

% Plot the points in C specified by IDX
knn = Q(IDX(p,:), :); % Flatten IDX and get the exact k-nearest neighbors
scatter(knn(:,1), knn(:,2), 'y', '*', 'DisplayName', 'Exact nearest neighbors');
scatter(Q(p,1), Q(p,2), 'g', '+', 'DisplayName', 'Queries');
hold off;

% Add legend and labels
legend show; % This displays the legend with names set in 'DisplayName'
xlabel('X');
ylabel('Y');
title('Approximate Nearest Neighbors');


%% Plot 3D points
clc, clearvars;

% Load execution times from .mat files
my_data = load('approx_tests/my_output.mat', 'IDX_approx');
matlab_data = load('approx_tests/test02.mat', 'IDX', 'Q', 'C');

% Extract variables from the loaded data
IDX_approx = my_data.IDX_approx;
IDX = matlab_data.IDX;
Q = matlab_data.Q;
C = matlab_data.C;

% Plot the corpus points, approximate neighbors, queries, and exact nearest neighbors
figure;
scatter3(C(:,1), C(:,2), C(:,3), 'b', 'o', 'DisplayName', 'Corpus');
hold on;
approx_knn = C(IDX_approx(:), :);
scatter3(approx_knn(:,1), approx_knn(:,2), approx_knn(:,3), 'r', 'o', 'DisplayName', 'Approximate neighbors');
scatter3(Q(:,1), Q(:,2), Q(:,3), 'g', '+', 'DisplayName', 'Queries');

% Plot the exact k-nearest neighbors using IDX
knn = C(IDX(:), :); % Flatten IDX and get the exact k-nearest neighbors
scatter3(knn(:,1), knn(:,2), knn(:,3), 'y', '*', 'DisplayName', 'Exact nearest neighbors');

% Add legend and labels
legend show; % This displays the legend with names set in 'DisplayName'
xlabel('X');
ylabel('Y');
zlabel('Z'); % Label for the third dimension
title('Approximate Nearest Neighbors (3D)');

% Set up the view for 3D visualization
view(3); % Sets the default 3D view angle
grid on; % Enables grid for better visual clarity in 3D

