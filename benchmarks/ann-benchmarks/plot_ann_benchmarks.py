import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# === Check arguments ===
if len(sys.argv) != 2:
    print("Usage: python plot_ann_benchmarks.py benchmark-file.hdf5")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

# === Load HDF5 file ===
with h5py.File(file_path, 'r') as f:
    required = ['nthreads', 'queries_per_sec', 'recall', 'num_clusters']
    for var in required:
        if var not in f:
            raise KeyError(f"Missing dataset '{var}' in HDF5 file.")
    
    nthreads = f['nthreads'][:].flatten()
    queries_per_sec = f['queries_per_sec'][:]
    recall = f['recall'][:]
    num_clusters = f['num_clusters'][:].flatten()

# === Derived info ===
num_thread_levels = len(nthreads)
num_cluster_levels = len(num_clusters)

# === Output directory ===
output_dir = os.path.join('..', '..', 'docs', 'figures')
os.makedirs(output_dir, exist_ok=True)

# === Plot 1: Queries per Second vs Number of Threads ===
plt.figure()
colors = plt.cm.get_cmap('tab10', num_cluster_levels)

for c in range(num_cluster_levels):
    plt.plot(nthreads, queries_per_sec[:, c], '-o',
             label=f'{num_clusters[c]} clusters',
             color=colors(c), linewidth=2, markersize=6)

plt.xscale('log', base=2)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlabel('Number of Threads')
plt.ylabel('Queries per Second')
plt.title('ANN: Queries per Second vs Number of Threads')
plt.legend(loc='upper right')
plt.xticks(nthreads, [str(int(n)) for n in nthreads])  # Show exact thread counts as ticks

output_file1 = os.path.join(output_dir, 'ann_throughput_vs_threads.png')
plt.savefig(output_file1)

# === Plot 2: Recall vs Number of Clusters ===
plt.figure()
colors = plt.cm.get_cmap('tab10', num_thread_levels)

for t in range(num_thread_levels):
    plt.plot(num_clusters, recall[t, :], '-s',
             label=f'{nthreads[t]} threads',
             color=colors(t), linewidth=2, markersize=6)

plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Recall (%)')
plt.title('ANN: Recall vs Number of Clusters')
plt.legend(loc='upper right')

output_file2 = os.path.join(output_dir, 'ann_recall_vs_clusters.png')
plt.savefig(output_file2)

print("Saved ANN benchmark plots to:")
print(f" - {output_file1}")
print(f" - {output_file2}")
