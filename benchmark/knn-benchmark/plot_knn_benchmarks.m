function plot_knn_benchmarks(matFilePath)
% plot_knn_benchmarks - Plots queries_per_sec vs nthreads from a .mat file.
%
% Usage:
%   plot_knn_benchmarks('benchmark-filename.mat')

    % Load the .mat file
    data = load(matFilePath);
    
    % Make sure variables exist
    varnames = {'nthreads', 'queries_per_sec'};
    for i = 1:length(varnames)
        if ~isfield(data, varnames{i})
            error('Missing variable "%s" from "%s"', varnames{i}, matFilePath);
        end
    end
    
    % Extract variables
    nthreads = data.nthreads;
    queries_per_sec = data.queries_per_sec;
    
    % Validate that they are vectors of same length
    if numel(nthreads) ~= numel(queries_per_sec)
        error('nthreads and queries_per_sec must be vectors of the same length.');
    end

    [~, sysinfo] = system('nproc --all');
    n_cores = str2double(strtrim(sysinfo));
    
    n = length(nthreads) - 1;
    exponents = 0:n-1;
    
    % Plot Queries per Second vs Number of threads
    figure; hold on;
    plot([0, n-1], queries_per_sec(1) * [1, 1], '--r', 'LineWidth', 2);
    plot(exponents, queries_per_sec(2:end), '-ob', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;

    set(gca, 'XTick', exponents);
    set(gca, 'XTickLabel', arrayfun(@(n) sprintf('%d', 2^n), exponents, 'UniformOutput', false));
    
    xlabel('Number of Threads');
    ylabel('Queries per Second');
    title(sprintf('Queries per Second vs Number of Threads (System Cores: %d)', n_cores));
    legend({sprintf('CBLAS threads: %d', n_cores), 'CBLAS threads: 1'}, 'Location', 'southeast');

    outputDir = fullfile('..', '..', 'docs', 'figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    outputFile = fullfile(outputDir, 'knn_throughput_vs_threads.png');
    
    saveas(gcf, outputFile);
end
