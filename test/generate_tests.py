import numpy as np
import os
import h5py
from sklearn.neighbors import NearestNeighbors

# === Configuration ===
output_dir = "data"
np.random.seed(0)  # For reproducibility
os.makedirs(output_dir, exist_ok=True)

def save_test_hdf5(train, test, distances, neighbors, test_name, test_index):
    if test_index < 10:
        filename = f"test0{test_index}.hdf5"
    else:
        filename = f"test{test_index}.hdf5"
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('test', data=test)
        f.create_dataset('distances', data=distances)
        f.create_dataset('neighbors', data=neighbors.astype(np.int32))

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('test_name', data=test_name, dtype=dt)

def run_and_save(train, test, K, test_name, test_index):
    train = np.atleast_2d(train)
    test = np.atleast_2d(test)
    K = int(K)

    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(train)
    distances, neighbors = knn.kneighbors(test, return_distance=True)

    save_test_hdf5(train, test, distances, neighbors, test_name, test_index)

# === Test 1 ===
run_and_save(train=7.2, test=3.7, K=1, test_name="Test 1", test_index=1)

# === Test 2 ===
train = np.array([[4.7, 5.2, 4.9],
               [0.0, 1.1, 2.0],
               [2.4, 6.7, 3.3]])
test = np.array([3.7, 1.2, 4.6])
run_and_save(train, test, 2, "Test 2", 2)

# === Randomized Tests ===
MAX_SIZE = 2000
NTESTS = 20

for i in range(3, NTESTS + 1):
    M = np.random.randint(1, MAX_SIZE + 1)
    N = np.random.randint(1, MAX_SIZE + 1)
    L = np.random.randint(1, MAX_SIZE + 1)

    train = np.random.rand(N, L)
    test = np.random.rand(M, L)
    K = np.random.randint(1, N + 1)

    run_and_save(train, test, K, f"Test {i}", i)

print(f"Generated {NTESTS} tests in: {output_dir}")
