import h5py
import os
import numpy as np
import glob

# Paths
old_dataset_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
new_dataset_dir = '/root/gearboxAssembly/data/datasets/temp_new_dataset'

def get_first_hdf5(directory):
    files = glob.glob(os.path.join(directory, '*.hdf5'))
    if not files:
        files = glob.glob(os.path.join(directory, '**/*.hdf5'), recursive=True)
    return files[0] if files else None

def inspect_file(filepath):
    info = {}
    try:
        with h5py.File(filepath, 'r') as f:
            if 'observations/qpos' in f:
                info['qpos_shape'] = f['observations/qpos'].shape
                info['action_shape'] = f['action'].shape
                info['qpos_mean'] = np.mean(f['observations/qpos'][()])
                info['action_mean'] = np.mean(f['action'][()])
                info['camera_names'] = list(f['observations/images'].keys())
            elif 'observations' in f: # ROCO format
                # Construct comparable shapes
                obs = f['observations']
                acts = f['actions']
                qpos_len = len(obs['left_arm_joint_pos'])
                # Assuming 14 dim qpos and action
                info['episode_len'] = qpos_len
                info['is_roco_format'] = True
                
                # Check keys
                info['obs_keys'] = list(obs.keys())
                # Check images
                if 'images' in obs:
                     info['camera_names'] = list(obs['images'].keys())
                else:
                     info['camera_names'] = [k for k in obs.keys() if 'rgb' in k or 'image' in k]

    except Exception as e:
        info['error'] = str(e)
    return info

print("--- Existing Dataset ---")
old_file = get_first_hdf5(old_dataset_dir)
if old_file:
    print(f"Sample file: {old_file}")
    print(inspect_file(old_file))
else:
    print("No HDF5 files found in old dataset dir")

print("\n--- New Dataset ---")
new_file = get_first_hdf5(new_dataset_dir)
if new_file:
    print(f"Sample file: {new_file}")
    print(inspect_file(new_file))
else:
    print("No HDF5 files found in new dataset dir (yet)")
