import os
import glob
import shutil

def merge_datasets():
    # Source and Destination
    source_dir = '/root/gearboxAssembly/data/datasets/temp_new_dataset'
    dest_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
    
    # 1. Get existing episode indices in destination
    current_files = glob.glob(os.path.join(dest_dir, 'episode_*.hdf5'))
    indices = []
    for f in current_files:
        try:
            filename = os.path.basename(f)
            idx = int(filename.split('_')[1].split('.')[0])
            indices.append(idx)
        except:
            pass
    
    start_idx = max(indices) + 1 if indices else 0
    print(f"Destination currently has {len(indices)} files.")
    print(f"New files will start from index: {start_idx}")
    
    # 2. Get files to copy
    source_files = glob.glob(os.path.join(source_dir, '*.hdf5'))
    source_files.sort() # Sort to keep order deterministic
    
    if not source_files:
        print("No files found in source directory.")
        return

    print(f"Found {len(source_files)} files to copy.")
    
    # 3. Copy and Rename
    for i, src_path in enumerate(source_files):
        new_idx = start_idx + i
        dst_filename = f"episode_{new_idx}.hdf5"
        dst_path = os.path.join(dest_dir, dst_filename)
        
        print(f"Copying {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)

    print("\nMerge Complete!")
    print(f"Total files in destination should be: {start_idx + len(source_files)}")
    print(f"Final episode index is: {start_idx + len(source_files) - 1}")
    
    # Also update constants.py num_episodes automatically if needed?
    # No, I should print it so the user/agent can update it.
    print(f"\n[IMPORTANT] Please update 'constants.py' num_episodes to: {start_idx + len(source_files)}")

if __name__ == "__main__":
    merge_datasets()
