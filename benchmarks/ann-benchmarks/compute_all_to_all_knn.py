import h5py
import numpy as np
import sys
import os
import psutil
from joblib import Parallel, delayed

# === CONFIG ===
MEMORY_RATIO = 0.1  # Ratio of available memory to use for batch size estimation

# === MEMORY FUNCTIONS ===
def available_memory_bytes(memory_ratio):
    return int(psutil.virtual_memory().available * memory_ratio)

def estimate_batch_size(total_points, dim, memory_ratio, dtype=np.float32):
    bytes_per_float = np.dtype(dtype).itemsize
    # Distance matrix + norms + dot products (approximation)
    bytes_per_point = (total_points * bytes_per_float + dim * bytes_per_float) * 3
    max_batch = max(1, available_memory_bytes(memory_ratio) // bytes_per_point)
    return min(max_batch, total_points, 10000)

# === MAIN COMPUTATION ===
def process_block(i, batch_size, K, train_test, corpus_norms, total_points):
    i_end = min(i + batch_size, total_points)
    query_block = train_test[i:i_end]
    query_norms = np.einsum('ij,ij->i', query_block, query_block)
    dists = query_norms[:, None] + corpus_norms[None, :] - 2 * (query_block @ train_test.T)
    np.maximum(dists, 0, out=dists)

    idx_block = np.argpartition(dists, K + 1, axis=1)[:, :K + 1]

    local_neighbors = np.empty((i_end - i, K), dtype=np.int32)
    for j in range(i_end - i):
        indices = idx_block[j]
        query_index = i + j
        indices = indices[indices != query_index]
        local_neighbors[j] = indices[:K]

    print(f'Processed {i_end}/{total_points} points')
    return i, i_end, local_neighbors

# === SCRIPT START ===
if len(sys.argv) != 2:
    print("Usage: python compute_all_to_all_knn_parallel.py <hdf5_file>")
    sys.exit(1)

hdf5_file = sys.argv[1]
if not os.path.isfile(hdf5_file):
    print(f"File '{hdf5_file}' not found.")
    sys.exit(1)

with h5py.File(hdf5_file, 'r+') as f:
    if 'all_to_all_neighbors' in f:
        print(f'Dataset "all_to_all_neighbors" already exists. Exiting.')
        sys.exit(0)
    else:
        print(f'Creating dataset "all_to_all_neighbors" in {hdf5_file}')

    train = f['train'][:].astype(np.float32)
    test = f['test'][:].astype(np.float32)
    neighbors = f['neighbors'][:].astype(np.int32)

    train_test = np.vstack([train, test])
    total_points, dim = train_test.shape
    K = neighbors.shape[1]

    batch_size = estimate_batch_size(total_points, dim, MEMORY_RATIO)
    print(f'Batch size: {batch_size} (MEMORY_RATIO={MEMORY_RATIO})')

    corpus_norms = np.einsum('ij,ij->i', train_test, train_test)

    num_jobs = os.cpu_count()
    print(f'Using {num_jobs} parallel workers...')

    all_to_all_neighbors = np.empty((total_points, K), dtype=np.int32)

    results = Parallel(n_jobs=num_jobs, prefer="threads")(
        delayed(process_block)(i, batch_size, K, train_test, corpus_norms, total_points)
        for i in range(0, total_points, batch_size)
    )

    for i, i_end, local_neighbors in results:
        all_to_all_neighbors[i:i_end] = local_neighbors

    print(f'Saving dataset "all_to_all_neighbors" to {hdf5_file}')
    chunk_size = (min(1000, total_points), K)
    f.create_dataset('all_to_all_neighbors', data=all_to_all_neighbors,
                     dtype='int32', chunks=chunk_size,
                     compression='gzip', compression_opts=5)

    print('Done saving dataset.')
