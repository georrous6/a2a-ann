% some edge case tests
C = 7.2;
Q = 3.7;
K = int32(1);

[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('test1.mat', 'C', 'Q', 'K', 'D', 'IDX');

C = [4.7, 5.2, 4.9; 0, 1.1, 2; 2.4, 6.7, 3.3];
Q = [3.7, 1.2, 4.6];
K = int32(2);

[IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

% Convert IDX to int32 before saving
IDX = int32(IDX);

save('test2.mat', 'C', 'Q', 'K', 'D', 'IDX');

MAX_SIZE = 1000;
for i = 3:50
    M = randi(MAX_SIZE);
    N = randi(MAX_SIZE);
    L = randi(MAX_SIZE);
    C = rand(N, L);
    Q = rand(M, L);
    K = int32(randi(N));

    [IDX, D] = knnsearch(C, Q, 'K', K, 'SortIndices', true);

    % Convert IDX to int32 before saving
    IDX = int32(IDX);
    save(sprintf('test%d.mat', i), 'C', 'Q', 'K', 'D', 'IDX');
end

disp('Test files generated successfully');
