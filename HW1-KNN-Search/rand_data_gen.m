C = [7, 6];
Q = [4, 6; 1, 1];

save('test/data.mat', 'C', 'Q');

[Idx, D] = knnsearch(C, Q, 'K', 1, 'SortIndices', true);

% Display the distance matrix
disp('Distance Matrix D:');
disp(D);

disp('Index matrix Idx:');
disp(Idx);