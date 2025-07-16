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
    if 'nthreads' not in f:
        raise KeyError(f'Missing dataset "nthreads" in "{hdf5_file_path}"')
    nthreads = f['nthreads'][:].flatten()

    # Collect all queries_per_sec_* datasets
    queries_data = {}
    for key in f.keys():
        if key.startswith('queries_per_sec'):
            queries_data[key] = f[key][:].flatten()

# === Validate data ===
for key, values in queries_data.items():
    if nthreads.size != values.size:
        raise ValueError(f"Mismatch: 'nthreads' and '{key}' have different lengths.")

# === Get number of CPU cores ===
n_cores = os.cpu_count()

# === Plot ===
plt.figure(figsize=(8, 6))

markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']

for idx, (key, values) in enumerate(queries_data.items()):
    label = key.replace('queries_per_sec_', '').upper()
    marker = markers[idx % len(markers)]
    color = plt.cm.tab10(idx % 10)  # consistent color

    # Plot dashed horizontal line for the first value (excluding from legend)
    plt.axhline(y=values[0], color=color, linestyle='--', linewidth=1.5)

    # Plot the rest of the data (excluding first element)
    plt.plot(nthreads[1:], values[1:], marker=marker, label=label,
             color=color, linewidth=2, markersize=8)

plt.grid(True)
plt.xscale('log', base=2)
plt.xticks(nthreads[1:], [str(int(x)) for x in nthreads[1:]])
plt.xlabel('Number of Threads')
plt.ylabel('Queries per Second')
plt.title(f'KNN: Throughput vs Threads (System Cores: {n_cores})')
plt.legend(loc='best')

# === Save figure ===
output_dir = os.path.join('..', '..', 'docs', 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'knn_throughput_vs_threads.png')
plt.tight_layout()
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")
