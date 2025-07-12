function plot_ann_benchmarks(matFilePath)

    % Load the .mat file
    data = load(matFilePath);
    
    % Make sure variables exist
    varnames = {'nthreads', 'queries_per_sec', 'recall', 'num_clusters'};
    for i = 1:length(varnames)
        if ~isfield(data, varnames{i})
            error('Missing variable "%s" from "%s"', varnames{i}, matFilePath);
        end
    end

    % Create output directory if it does not exist
    outputDir = fullfile('..', '..', 'docs', 'figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    %% Plot 1: Queries per Second vs Number of Threads for different cluster counts
    n_clusters = data.num_clusters;
    nthreads = data.nthreads;
    queries_per_sec = data.queries_per_sec;
    recall = data.recall;
    clusters = length(n_clusters);

    figure; hold on;

    colors = lines(clusters); % for different clusters

    for c = 1:clusters
        plot(nthreads, queries_per_sec(:, c), '-o', ...
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


    %% Plot 2: Recall vs Number of Clusters for different thread counts
    figure; hold on;

    colors = lines(num_threads); % for different thread counts

    for t = 1:num_threads
        plot(n_clusters, recall(t, :), '-s', ...
            'Color', colors(t,:), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', sprintf('%d threads', nthreads(t)));
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
