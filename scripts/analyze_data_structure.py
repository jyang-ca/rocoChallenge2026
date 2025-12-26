import h5py
import numpy as np
import sys

def print_structure(name, obj):
    indent = "  " * (name.count('/') + 1)
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}, size={obj.size * obj.dtype.itemsize / 1024 / 1024:.2f} MB")
        # Check for compression
        if obj.compression:
             print(f"{indent}  (Compressed: {obj.compression}, Opts: {obj.compression_opts})")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name} (Group)")

def analyze_file(file_path):
    print(f"\nAnalyzing: {file_path}")
    print("=" * 60)
    try:
        with h5py.File(file_path, 'r') as f:
            f.visititems(print_structure)
            
            # Additional logic to check specific heavy items
            print("-" * 60)
            print("Detailed Sampling:")
            
            # Check image resolution and type
            if 'observations/images/head_rgb' in f:
                img = f['observations/images/head_rgb'][0]
                print(f"  head_rgb sample shape: {img.shape}, min={np.min(img)}, max={np.max(img)}")
            elif 'observations' in f and 'head_rgb' in f['observations']:
                 img = f['observations']['head_rgb'][0]
                 print(f"  head_rgb sample shape: {img.shape}, min={np.min(img)}, max={np.max(img)}")
            
            # Check length again
            if 'action' in f:
                print(f"  Episode Length (action): {f['action'].shape[0]}")
            elif 'actions' in f:
                # Try finding a key in actions
                keys = list(f['actions'].keys())
                if keys:
                    print(f"  Episode Length (actions/{keys[0]}): {f['actions'][keys[0]].shape[0]}")

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    files_to_check = [
        # Small file from temp (likely corrupted one or short one) -> let's pick a big one
        '/root/gearboxAssembly/data/datasets/temp_new_dataset/data_20251222_233128.hdf5',
        # Standard file from roco
        '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated/episode_0.hdf5'
    ]
    
    for fp in files_to_check:
        analyze_file(fp)
