function compute_all_to_all_knn(hdf5_file)
    % Ensure the script uses single precision for data and int32 for indices
    assert(ischar(hdf5_file) || isstring(hdf5_file), 'Input must be a filename');
    
    % Check if 'train_test' and 'train_test_neighbors' datasets already exist in the HDF5 file
    info = h5info(hdf5_file);
    existing_datasets = {info.Datasets.Name};

    % Helper to check recursively for datasets in groups
    function exists = datasetExists(groupInfo, datasetName)
        exists = false;
        if any(strcmp({groupInfo.Datasets.Name}, datasetName))
            exists = true;
            return;
        end
        for g = groupInfo.Groups
            if datasetExists(g, datasetName)
                exists = true;
                return;
            end
        end
    end

    % The target datasets are at root '/', so just check root group
    if any(strcmp(existing_datasets, 'train_test')) && any(strcmp(existing_datasets, 'train_test_neighbors'))
        fprintf('Datasets "train_test" and "train_test_neighbors" already exist in %s. Exiting early.\n', hdf5_file);
        return;
    end

    % Load train, test, neighbors from HDF5 as single precision
    train = single(h5read(hdf5_file, '/train'));
    test  = single(h5read(hdf5_file, '/test'));
    neighbors = int32(h5read(hdf5_file, '/neighbors'));  % neighbors as int32

    % Combine train and test into train_test (single precision)
    train_test = [train test];
    
    % K is number of neighbors (columns of neighbors)
    K = size(neighbors, 1);  

    % Parameters
    total_points = size(train_test, 2);
    batch_size = 1000;  % Tune for memory/performance tradeoff

    % Preallocate train_test_neighbors with int32 type for neighbor indices
    train_test_neighbors = zeros(K, total_points, 'int32');

    fprintf('Starting all-to-all ANN search...\n');

    for i = 1:batch_size:total_points
        i_end = min(i + batch_size - 1, total_points);
        query_block = train_test(:, i:i_end);

        % Precompute squared norms
        C_norms = sum(train_test.^2, 1, 'native');               % 1 x N
        Q_norms = sum(query_block.^2, 1, 'native');              % 1 x B
        dot_products = train_test' * query_block;                % N x B

        % Compute squared Euclidean distances: (broadcasted)
        dists = bsxfun(@plus, Q_norms, C_norms') - 2 * dot_products;  % N x B

        % Numerical stability
        dists = max(dists, 0);

        % Find K+1 nearest neighbors (include self)
        [~, idx_block] = mink(dists, K + 1, 1);  % (K+1) x batch_size

        % Remove self-match
        final_block = zeros(K, size(idx_block, 2), 'int32');
        for j = 1:size(idx_block, 2)
            self_idx = find(idx_block(:, j) == (i + j - 1), 1);
            if isempty(self_idx)
                final_block(:, j) = int32(idx_block(1:K, j));
            else
                tmp = idx_block(:, j);
                tmp(self_idx) = [];
                final_block(:, j) = int32(tmp(1:K));
            end
        end

        % Store neighbors indices for this block
        train_test_neighbors(:, i:i_end) = final_block;
        fprintf('Processed %d/%d points\n', i_end, total_points);
    end

    % Write train_test and train_test_neighbors back to the input HDF5 file

    fprintf('Saving datasets "train_test" and "train_test_neighbors" to %s\n', hdf5_file);

    % Use chunking and compression for efficient storage
    chunkSize_train_test = [size(train_test,1), min(1000, size(train_test,2))];
    chunkSize_neighbors = [size(train_test_neighbors,1), min(1000, size(train_test_neighbors,2))];

    h5create(hdf5_file, '/train_test', size(train_test), 'Datatype', 'single', 'ChunkSize', chunkSize_train_test, 'Deflate', 5);
    h5write(hdf5_file, '/train_test', train_test);

    h5create(hdf5_file, '/train_test_neighbors', size(train_test_neighbors), 'Datatype', 'int32', 'ChunkSize', chunkSize_neighbors, 'Deflate', 5);
    h5write(hdf5_file, '/train_test_neighbors', train_test_neighbors);

    fprintf('Done saving datasets.\n');
end
