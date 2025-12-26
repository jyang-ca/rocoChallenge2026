import os
import glob
import shutil

old_dataset_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
new_dataset_dir = '/root/gearboxAssembly/data/datasets/temp_new_dataset'

def get_max_episode_id(directory):
    files = glob.glob(os.path.join(directory, 'episode_*.hdf5'))
    max_id = -1
    for f in files:
        basename = os.path.basename(f)
        try:
            # Assumes format episode_{N}.hdf5
            idx = int(basename.split('_')[1].split('.')[0])
            if idx > max_id:
                max_id = idx
        except:
            pass
    return max_id

print(f"Scanning {old_dataset_dir}...")
max_id = get_max_episode_id(old_dataset_dir)
print(f"Max episode ID in old dataset: {max_id}")

start_id = max_id + 1
print(f"New episodes will start from: {start_id}")

# Find new files
# Note: The new dataset might have arbitrary names or nested structure.
# We'll search recursively.
new_files = glob.glob(os.path.join(new_dataset_dir, '**/*.hdf5'), recursive=True)
print(f"Found {len(new_files)} files in new dataset.")

# Sort new files if possible to maintain some order, though not strictly required
new_files.sort()

# Move and rename
if len(new_files) == 0:
    print("No files to merge!")
    exit(1)

count = 0
for i, src_path in enumerate(new_files):
    new_id = start_id + i
    dst_name = f'episode_{new_id}.hdf5'
    dst_path = os.path.join(old_dataset_dir, dst_name)
    
    print(f"Moving {src_path} -> {dst_path}")
    shutil.move(src_path, dst_path)
    count += 1

print(f"Merged {count} files.")
print(f"New total episodes: {start_id + count}")
