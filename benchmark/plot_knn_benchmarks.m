function plot_knn_benchmarks(matFilePath)
% plotQueriesPerSec - Plots queries_per_sec vs nthreads from a .mat file.
%
% Usage:
%   plot_knn_benchmarks('benchmark-filename.mat')

    % Load the .mat file
    data = load(matFilePath);
    
    % Make sure variables exist
    if ~isfield(data, 'nthreads') || ~isfield(data, 'queries_per_sec')
        error('The .mat file must contain variables "nthreads" and "queries_per_sec".');
    end
    
    % Extract variables
    nthreads = data.nthreads;
    queries_per_sec = data.queries_per_sec;
    
    % Validate that they are vectors of same length
    if numel(nthreads) ~= numel(queries_per_sec)
        error('nthreads and queries_per_sec must be vectors of the same length.');
    end
    
    % Create the plot
    figure;
    plot(nthreads, queries_per_sec, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;
    
    xlabel('Number of Threads');
    ylabel('Queries per Second');
    title('Queries per Second vs Number of Threads');
    
    saveas(gcf, 'knn_benchmarks.png');
end
