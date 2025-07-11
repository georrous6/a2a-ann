% Replace 'your_file.hdf5' with your actual HDF5 filename
hdf5_file = fullfile("datasets/fashion-mnist-784-euclidean.hdf5");

% Load train_test dataset
train_test = h5read(hdf5_file, '/train_test');

% Load train_test_neighbors dataset
train_test_neighbors = h5read(hdf5_file, '/train_test_neighbors');

% Display size of loaded data
fprintf('Loaded train_test with size: %dx%d\n', size(train_test,1), size(train_test,2));
fprintf('Loaded train_test_neighbors with size: %dx%d\n', size(train_test_neighbors,1), size(train_test_neighbors,2));
