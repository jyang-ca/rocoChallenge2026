import os
import glob
import shutil

def symlink_datasets():
    # Source and Destination
    source_dir = '/root/gearboxAssembly/data/datasets/temp_new_dataset'
    dest_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
    
    # 1. Clean up potential partial copies from the killed process
    # The copy started at index 192. We should clean up anything from 192 onwards just to be safe.
    start_idx = 192
    print(f"Cleaning up potentially incomplete files starting from index {start_idx}...")
    
    existing_files = glob.glob(os.path.join(dest_dir, 'episode_*.hdf5'))
    for f in existing_files:
        try:
            filename = os.path.basename(f)
            idx = int(filename.split('_')[1].split('.')[0])
            if idx >= start_idx:
                os.remove(f)
                # print(f"Removed {f}")
        except:
            pass
    print("Cleanup complete.")

    # 2. Get files to link
    source_files = glob.glob(os.path.join(source_dir, '*.hdf5'))
    source_files.sort()
    
    if not source_files:
        print("No files found in source directory.")
        return

    print(f"Found {len(source_files)} files to link.")
    
    # 3. Create Symlinks
    for i, src_path in enumerate(source_files):
        new_idx = start_idx + i
        dst_filename = f"episode_{new_idx}.hdf5"
        dst_path = os.path.join(dest_dir, dst_filename)
        
        # print(f"Linking {src_path} -> {dst_path}")
        os.symlink(src_path, dst_path)

    print("\nSymlink Complete!")
    print(f"Total files in destination: {start_idx + len(source_files)}")
    
    # Verify a few
    print("\nVerifying integration...")
    print(f"Check episode_192: {os.path.exists(os.path.join(dest_dir, 'episode_192.hdf5'))}")
    print(f"Check episode_{start_idx + len(source_files) - 1}: {os.path.exists(os.path.join(dest_dir, f'episode_{start_idx + len(source_files) - 1}.hdf5'))}")

if __name__ == "__main__":
    symlink_datasets()
