import numpy as np
import os
import h5py
from sklearn.neighbors import NearestNeighbors
import sys

# === Argument Parsing ===
if len(sys.argv) != 2:
    print("Usage: python script.py <output_file.hdf5>")
    sys.exit(1)

output_path = os.path.abspath(sys.argv[1])

# === Exit if file already exists ===
if os.path.exists(output_path):
    print(f"File '{output_path}' already exists. Exiting.")
    sys.exit(0)

# === Configuration ===
np.random.seed(0)
K = 10
M, N, L = 5000, 2000, 100  # M: test samples, N: train samples, L: dimensions

# === Generate Data ===
train = np.random.rand(N, L).astype(np.float32)
test = np.random.rand(M, L).astype(np.float32)

# === Nearest Neighbors (test -> train) ===
knn = NearestNeighbors(n_neighbors=K)
knn.fit(train)
distances, neighbors = knn.kneighbors(test, return_distance=True)

# === Save train/test/test->train knn ===
with h5py.File(output_path, 'w') as f:
    f.create_dataset('train', data=train)
    f.create_dataset('test', data=test)
    f.create_dataset('distances', data=distances)
    f.create_dataset('neighbors', data=neighbors.astype(np.int32))
    f.create_dataset('test_name', data="Single Random Test", dtype=h5py.string_dtype('utf-8'))

# === All-to-All Nearest Neighbors ===
train_test = np.vstack([train, test])
total_points = train_test.shape[0]
corpus_norms = np.sum(train_test ** 2, axis=1)
batch_size = 200

all_to_all_neighbors = np.zeros((total_points, K), dtype=np.int32)

for i in range(0, total_points, batch_size):
    i_end = min(i + batch_size, total_points)
    query_block = train_test[i:i_end]

    query_norms = np.sum(query_block ** 2, axis=1)
    dot_products = query_block @ train_test.T
    dists = query_norms[:, None] + corpus_norms[None, :] - 2 * dot_products
    dists = np.maximum(dists, 0)

    idx_block = np.argpartition(dists, K + 1, axis=1)[:, :K + 1]

    for j in range(i_end - i):
        query_index = i + j
        indices = idx_block[j]
        indices = indices[indices != query_index]  # remove self
        all_to_all_neighbors[query_index, :] = indices[:K]

    print(f'Processed {i_end}/{total_points} points')

# === Save All-to-All Neighbors ===
with h5py.File(output_path, 'a') as f:
    f.create_dataset('all_to_all_neighbors', data=all_to_all_neighbors,
                     dtype='int32',
                     chunks=(min(1000, total_points), K),
                     compression='gzip', compression_opts=5)

print(f"All data saved to: {output_path}")
