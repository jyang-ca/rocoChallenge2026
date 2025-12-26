import h5py
import os
import glob
import sys

dataset_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'

if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]

print(f"Checking HDF5 files in {dataset_dir}...")

files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))
files.sort()

if not files:
    print("No HDF5 files found.")
    sys.exit(0)

print(f"Found {len(files)} files.")

corrupted_files = []
valid_count = 0

for filepath in files:
    filename = os.path.basename(filepath)
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to read a key to ensure it's readable
            keys = list(f.keys())
            valid_count += 1
    except Exception as e:
        print(f"CORRUPTED: {filename} - {e}")
        corrupted_files.append(filepath)

print(f"\nSummary:")
print(f"Valid files: {valid_count}")
print(f"Corrupted files: {len(corrupted_files)}")
if corrupted_files:
    print("List of corrupted files:")
    for f in corrupted_files:
        print(f)
