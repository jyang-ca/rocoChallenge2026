import h5py
import os
import glob
import numpy as np

def check_dataset_lengths(dataset_dirs):
    max_len_global = 0
    file_with_max_len = ""
    
    for data_dir in dataset_dirs:
        print(f"Scanning directory: {data_dir}")
        search_pattern = os.path.join(data_dir, '*.hdf5')
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Warning: No .hdf5 files found in {data_dir}")
            continue

        for file_path in files:
            try:
                with h5py.File(file_path, 'r') as root:
                    # Check for different formats
                    if 'action' in root:
                        episode_len = root['/action'].shape[0]
                    elif 'actions' in root:
                         # ROCO format often has dict under actions
                        if 'left_arm_action' in root['actions']:
                            episode_len = root['actions']['left_arm_action'].shape[0]
                        else:
                             # Fallback or different structure
                             print(f"Skipping {file_path}: efficient key search failed")
                             continue
                    else:
                        print(f"Skipping {file_path}: unknown format")
                        continue
                    
                    if episode_len > max_len_global:
                        max_len_global = episode_len
                        file_with_max_len = file_path
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print("\n" + "="*50)
    print("SCAN RESULTS")
    print("="*50)
    print(f"Maximum Episode Length Found: {max_len_global}")
    print(f"File with Max Length: {file_with_max_len}")
    
    if max_len_global > 590:
        print("\n[WARNING] Max length exceeds current setting (590)!")
        print(f"You should update 'constants.py' episode_len to at least {max_len_global}")
    else:
        print("\n[OK] Current setting (590) is safe. No truncation will occur.")

if __name__ == "__main__":
    # List all dataset directories you want to train on
    dirs_to_check = [
        '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated',
        '/root/gearboxAssembly/data/datasets/temp_new_dataset'
    ]
    check_dataset_lengths(dirs_to_check)
