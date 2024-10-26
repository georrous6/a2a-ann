C = [1, 2, 3; 4, 5, 6; 7, 8, 10];
Q = [4, 5, 6];

save('test/data.mat', 'C', 'Q');

[Idx, D] = knnsearch(C, Q, 'K', 3, 'SortIndices', false);

% Display the distance matrix
disp('Distance Matrix D:');
disp(D);

disp('Index matrix Idx:');
disp(Idx);