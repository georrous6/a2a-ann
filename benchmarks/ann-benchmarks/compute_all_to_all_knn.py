import h5py
import numpy as np
import sys
import os

# === Config ===
if len(sys.argv) != 2:
    print("Usage: python compute_all_to_all_knn.py <hdf5_file>")
    sys.exit(1)

hdf5_file = sys.argv[1]

if not os.path.isfile(hdf5_file):
    print(f"File '{hdf5_file}' not found.")
    sys.exit(1)

# === Main Logic ===
with h5py.File(hdf5_file, 'r+') as f:
    if 'all_to_all_neighbors' in f:
        print(f'Dataset "all_to_all_neighbors" already exists in {hdf5_file}. Exiting early.')
        sys.exit(0)

    # Load data (assumes row-major order in HDF5)
    train = f['train'][:].astype(np.float32)
    test = f['test'][:].astype(np.float32)
    neighbors = f['neighbors'][:].astype(np.int32)

    # Concatenate train and test
    train_test = np.vstack([train, test]).astype(np.float32)

    K = neighbors.shape[1]
    total_points = train_test.shape[0]
    dim = train_test.shape[1]
    batch_size = 200

    # Preallocate: shape (N, K)
    all_to_all_neighbors = np.zeros((total_points, K), dtype=np.int32)

    print('Starting all-to-all ANN search...')

    # Precompute corpus squared norms
    corpus_norms = np.sum(train_test ** 2, axis=1)

    for i in range(0, total_points, batch_size):
        i_end = min(i + batch_size, total_points)
        query_block = train_test[i:i_end]

        query_norms = np.sum(query_block ** 2, axis=1)
        dot_products = query_block @ train_test.T
        dists = query_norms[:, None] + corpus_norms[None, :] - 2 * dot_products
        dists = np.maximum(dists, 0)

        idx_block = np.argpartition(dists, K + 1, axis=1)[:, :K + 1]

        for j in range(i_end - i):
            indices = idx_block[j]
            query_index = i + j
            indices = indices[indices != query_index]
            all_to_all_neighbors[query_index, :] = indices[:K]

        print(f'Processed {i_end}/{total_points} points')

    print(f'Saving dataset "all_to_all_neighbors" to {hdf5_file}')

    chunk_size = (min(1000, total_points), K)
    f.create_dataset('/all_to_all_neighbors', data=all_to_all_neighbors,
                     dtype='int32', chunks=chunk_size,
                     compression='gzip', compression_opts=5)

    print('Done saving dataset.')
