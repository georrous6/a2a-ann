import numpy as np
import os
import h5py
from sklearn.neighbors import NearestNeighbors

# === Configuration ===
output_dir = "data"
np.random.seed(0)  # For reproducibility
os.makedirs(output_dir, exist_ok=True)

def save_test_hdf5(C, Q, K, test_D, test_IDX, test_name, test_index):
    if test_index < 10:
        filename = f"test0{test_index}.hdf5"
    else:
        filename = f"test{test_index}.hdf5"
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('C', data=C)
        f.create_dataset('Q', data=Q)
        # Store K as 2D matrix (1x1)
        f.create_dataset('K', data=np.array(K, dtype=np.int32).reshape(1, 1))
        f.create_dataset('test_D', data=test_D)
        f.create_dataset('test_IDX', data=test_IDX.astype(np.int32))
        f.create_dataset('test_name', data=np.string_(test_name))

def run_and_save(C, Q, K, test_name, test_index):
    C = np.atleast_2d(C)
    Q = np.atleast_2d(Q)
    K = int(K)

    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(C)
    test_D, test_IDX = knn.kneighbors(Q, return_distance=True)

    save_test_hdf5(C, Q, K, test_D, test_IDX, test_name, test_index)

# === Test 1 ===
run_and_save(C=7.2, Q=3.7, K=1, test_name="Test 1", test_index=1)

# === Test 2 ===
C2 = np.array([[4.7, 5.2, 4.9],
               [0.0, 1.1, 2.0],
               [2.4, 6.7, 3.3]])
Q2 = np.array([3.7, 1.2, 4.6])
run_and_save(C2, Q2, 2, "Test 2", 2)

# === Randomized Tests ===
MAX_SIZE = 2000
NTESTS = 20

for i in range(3, NTESTS + 1):
    M = np.random.randint(1, MAX_SIZE + 1)
    N = np.random.randint(1, MAX_SIZE + 1)
    L = np.random.randint(1, MAX_SIZE + 1)

    C = np.random.rand(N, L)
    Q = np.random.rand(M, L)
    K = np.random.randint(1, N + 1)

    run_and_save(C, Q, K, f"Test {i}", i)

print(f"Generated {NTESTS} tests in: {output_dir}")
