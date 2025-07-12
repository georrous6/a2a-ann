import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# === Check command-line arguments ===
if len(sys.argv) != 2:
    print("Usage: python plot_knn_benchmarks.py benchmark-file.hdf5")
    sys.exit(1)

# === Load the .hdf5 file ===
hdf5_file_path = sys.argv[1]
if not os.path.exists(hdf5_file_path):
    print(f"Error: File '{hdf5_file_path}' does not exist.")
    sys.exit(1)

with h5py.File(hdf5_file_path, 'r') as f:
    required_vars = ['nthreads', 'queries_per_sec']
    for var in required_vars:
        if var not in f:
            raise KeyError(f'Missing dataset "{var}" in "{hdf5_file_path}"')
    
    nthreads = f['nthreads'][:].flatten()
    queries_per_sec = f['queries_per_sec'][:].flatten()

# === Validate shape ===
if nthreads.size != queries_per_sec.size:
    raise ValueError("nthreads and queries_per_sec must be vectors of the same length.")

# === Get number of CPU cores ===
n_cores = os.cpu_count()

# === Prepare data for plotting ===
n = len(nthreads) - 1
exponents = np.arange(0, n)

# === Plot ===
plt.figure()
plt.plot([0, n - 1], [queries_per_sec[0]] * 2, '--r', linewidth=2, label=f'CBLAS threads: {n_cores}')
plt.plot(exponents, queries_per_sec[1:], '-ob', linewidth=2, markersize=8, label='CBLAS threads: 1')

plt.grid(True)
plt.xticks(exponents, [f'{2**i}' for i in exponents])
plt.xlabel('Number of Threads')
plt.ylabel('Queries per Second')
plt.title(f'KNN: Throughput vs Number of Threads (System Cores: {n_cores})')
plt.legend(loc='lower right')

# === Save figure ===
output_dir = os.path.join('..', '..', 'docs', 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'knn_throughput_vs_threads.png')
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")
