function generate_tests(outputDir)

    rng(0); % Add random seed for reproducibility

    % some edge case tests
    C = 7.2;
    Q = 3.7;
    K = int32(1);
    
    % Create directory if it doesn't exist
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);
    
    % Convert IDX to int32 before saving
    test_IDX = int32(test_IDX - 1);
    
    test_name = "Test 1";
    save(fullfile(outputDir, 'test01.mat'), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
    
    C = [4.7, 5.2, 4.9; 0, 1.1, 2; 2.4, 6.7, 3.3];
    Q = [3.7, 1.2, 4.6];
    K = int32(2);
    
    [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);
    
    % Convert IDX to int32 before saving
    test_IDX = int32(test_IDX - 1);
    
    test_name = "Test 2";
    save(fullfile(outputDir, 'test02.mat'), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
    
    MAX_SIZE = 2000;
    NTESTS = 20;
    for i = 3:NTESTS
        M = randi(MAX_SIZE);
        N = randi(MAX_SIZE);
        L = randi(MAX_SIZE);
        C = rand(N, L);
        Q = rand(M, L);
        K = int32(randi(N));
    
        [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);
    
        % Convert IDX to int32 before saving
        test_IDX = int32(test_IDX - 1);
        test_name = sprintf("Test %d", i);
        if i < 10
            save(fullfile(outputDir, sprintf('test0%d.mat', i)), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
        else
            save(fullfile(outputDir, sprintf('test%d.mat', i)), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
        end
    end
end

