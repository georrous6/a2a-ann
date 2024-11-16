function generate_knn_tests()

    % some edge case tests
    C = 7.2;
    Q = 3.7;
    K = int32(1);
    
    % Create directory if it doesn't exist
    if ~exist('knn_tests', 'dir')
        mkdir('knn_tests');
    end
    
    [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
    
    % Convert IDX to int32 before saving
    test_IDX = int32(test_IDX);
    
    test_name = "Test 1";
    save('knn_tests/test01.mat', 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
    
    C = [4.7, 5.2, 4.9; 0, 1.1, 2; 2.4, 6.7, 3.3];
    Q = [3.7, 1.2, 4.6];
    K = int32(2);
    
    [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
    
    % Convert IDX to int32 before saving
    test_IDX = int32(test_IDX);
    
    test_name = "Test 2";
    save('knn_tests/test02.mat', 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
    
    MAX_SIZE = 1000;
    NTESTS = 2;
    for i = 3:NTESTS
        M = randi(MAX_SIZE);
        N = randi(MAX_SIZE);
        L = randi(MAX_SIZE);
        C = rand(N, L);
        Q = rand(M, L);
        K = int32(randi(N));
    
        [test_IDX, test_D] = knnsearch(C, Q, 'K', K, 'SortIndices', false);
    
        % Convert IDX to int32 before saving
        test_IDX = int32(test_IDX);
        test_name = sprintf("Test %d", i);
        if i < 10
            save(sprintf('knn_tests/test0%d.mat', i), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
        else
            save(sprintf('knn_tests/test%d.mat', i), 'C', 'Q', 'K', 'test_D', 'test_IDX', 'test_name');
        end
    end
end

