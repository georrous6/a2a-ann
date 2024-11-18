clc;

info = h5info('fashion-mnist-784-euclidean.hdf5');
disp(info);

train_data = h5read('fashion-mnist-784-euclidean.hdf5', '/train');
test_data = h5read('fashion-mnist-784-euclidean.hdf5', '/test');
distances = h5read('fashion-mnist-784-euclidean.hdf5', '/distances'); % If needed
neighbors = h5read('fashion-mnist-784-euclidean.hdf5', '/neighbors'); % If needed

