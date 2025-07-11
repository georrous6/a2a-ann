function plot_knn_benchmarks(matFilePath)
% plot_knn_benchmarks - Plots queries_per_sec vs nthreads from a .mat file.
%
% Usage:
%   plot_knn_benchmarks('benchmark-filename.mat')

    % Load the .mat file
    data = load(matFilePath);
    
    % Make sure variables exist
    varnames = {'knn_nthreads'; 'knn_queries_per_sec'; 'knn_recall'; 'ann_nthreads';
        'ann_queries_per_sec'; 'ann_recall', 'ann_num_clusters'};
    for i = 1:length(varnames)
        if ~isfield(data, varnames{i})
            error('Missing variable "%s" from "%s"', varnames{i}, matFilePath);
        end
    end
    
    %% KNN: Plot Queries per Second vs Number of threads
    % Extract variables
    knn_nthreads = data.knn_nthreads;
    knn_queries_per_sec = data.queries_per_sec;
    
    % Validate that they are vectors of same length
    if numel(knn_nthreads) ~= numel(knn_queries_per_sec)
        error('nthreads and queries_per_sec must be vectors of the same length.');
    end

    [~, sysinfo] = system('nproc --all');
    n_cores = str2double(strtrim(sysinfo));
    
    n = length(knn_nthreads) - 1;
    exponents = 0:n-1;
    % Create the plot
    figure; hold on;
    plot([0, n-1], knn_queries_per_sec(1) * [1, 1], '--r', 'LineWidth', 2);
    plot(exponents, knn_queries_per_sec(2:end), '-ob', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;

    set(gca, 'XTick', exponents);
    set(gca, 'XTickLabel', arrayfun(@(n) sprintf('%d', 2^n), exponents, 'UniformOutput', false));
    
    xlabel('Number of Threads');
    ylabel('Queries per Second');
    title(sprintf('Queries per Second vs Number of Threads (System Cores: %d)', n_cores));
    legend({sprintf('CBLAS threads: %d', n_cores), 'CBLAS threads: 1'}, 'Location', 'southeast');

    outputDir = fullfile('..', 'docs', 'figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    outputFile = fullfile(outputDir, 'knn_throughput_vs_threads.png');
    
    saveas(gcf, outputFile);

    %% ANN Plot 1: Queries per Second vs Number of Threads for different cluster counts
    n_clusters = data.ann_num_clusters;
    ann_nthreads = data.ann_nthreads;
    ann_queries_per_sec = data.ann_queries_per_sec;
    ann_recall = data.ann_recall;
    num_clusters = length(n_clusters);

    figure; hold on;

    colors = lines(num_clusters); % for different clusters

    for c = 1:num_clusters
        plot(ann_nthreads, ann_queries_per_sec(:, c), '-o', ...
            'Color', colors(c,:), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', sprintf('%d clusters', n_clusters(c)));
    end

    grid on;
    xlabel('Number of Threads');
    ylabel('Queries per Second');
    title('ANN: Queries per Second vs Number of Threads');
    legend('Location', 'northwest');

    outputFile1 = fullfile(outputDir, 'ann_throughput_vs_threads.png');
    saveas(gcf, outputFile1);


    %% ANN Plot 2: Recall vs Number of Clusters for different thread counts
    figure; hold on;

    colors = lines(num_threads); % for different thread counts

    for t = 1:num_threads
        plot(n_clusters, ann_recall(t, :), '-s', ...
            'Color', colors(t,:), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', sprintf('%d threads', ann_nthreads(t)));
    end

    grid on;
    xlabel('Number of Clusters');
    ylabel('Recall (%)');
    title('ANN: Recall vs Number of Clusters');
    legend('Location', 'southeast');

    outputFile2 = fullfile(outputDir, 'ann_recall_vs_clusters.png');
    saveas(gcf, outputFile2);

    fprintf('Saved ANN benchmark plots to:\n - %s\n - %s\n', outputFile1, outputFile2);
end
