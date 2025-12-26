import h5py
import os
import glob

dataset_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
num_episodes = 191

print(f"Checking {num_episodes} files in {dataset_dir}...")

corrupted_files = []
valid_count = 0

for i in range(num_episodes):
    filename = f'episode_{i}.hdf5'
    filepath = os.path.join(dataset_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"MISSING: {filename}")
        continue
        
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to read a key to ensure it's readable
            keys = list(f.keys())
            valid_count += 1
    except Exception as e:
        print(f"CORRUPTED: {filename} - {e}")
    if i >= 40: break
    filename = f'episode_{i}.hdf5'
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
