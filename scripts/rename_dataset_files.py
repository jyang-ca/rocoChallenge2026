import os
import glob
import shutil

dataset_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'

# Get all hdf5 files
files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))

# Filter out files that already match the pattern episode_{i}.hdf5 to avoid double renaming if run multiple times
# But here we want to rename EVERYTHING to a clean sequence.
# To avoid conflicts (e.g. renaming 13.hdf5 to episode_0.hdf5 when episode_0.hdf5 might already exist),
# we will first rename everything to a temporary name.

print(f"Found {len(files)} files.")

# Sort files to ensure deterministic order (numerically if possible)
def extract_number(f):
    basename = os.path.basename(f)
    name_without_ext = os.path.splitext(basename)[0]
    # Try to extract number from "13.hdf5" or "episode_13.hdf5"
    if 'episode_' in name_without_ext:
        return int(name_without_ext.split('_')[1])
    elif name_without_ext.isdigit():
        return int(name_without_ext)
    else:
        return 999999 # fallback

files.sort(key=extract_number)

# Renaming
for i, file_path in enumerate(files):
    new_name = f'episode_{i}.hdf5'
    new_path = os.path.join(dataset_dir, new_name)
    
    if file_path == new_path:
        print(f"Skipping {os.path.basename(file_path)} (already correct)")
        continue
        
    # If target exists, we have a collision risk. 
    # Since we are renaming strictly to episode_0...N, and the original files are likely random numbers or unordered,
    # collision can happen if we are not careful.
    # Strategy: Rename to temporary, then rename to final.
    temp_path = os.path.join(dataset_dir, f'temp_{i}.hdf5')
    os.rename(file_path, temp_path)

# Second pass: temp to final
temp_files = glob.glob(os.path.join(dataset_dir, 'temp_*.hdf5'))
# We need to sort them by the index in the temp name to keep the order from the first pass
temp_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

for i, temp_path in enumerate(temp_files):
    final_name = f'episode_{i}.hdf5'
    final_path = os.path.join(dataset_dir, final_name)
    os.rename(temp_path, final_path)
    print(f"Renamed to {final_name}")

print("Renaming complete.")
