import h5py
import numpy as np
import sys

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"\nGroup: {name}")
        for key, val in obj.attrs.items():
            print(f"  Attribute: {key} = {val}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
        # Print range for numerical data
        if np.issubdtype(obj.dtype, np.number):
             data = obj[:]
             if data.size > 0:
                 print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        for key, val in obj.attrs.items():
            print(f"  Attribute: {key} = {val}")

def inspect_file(file_path):
    print(f"Inspecting file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nRoot Attributes:")
            for key, val in f.attrs.items():
                print(f"  {key}: {val}")
            
            f.visititems(print_hdf5_structure)
            
            # Specific check for timestamps to calculate FPS if not explicitly stated
            times = None
            if 'time' in f:
                times = f['time'][:]
            elif 'timestamp' in f:
                times = f['timestamp'][:]
            elif 'observations/timestamp' in f:
                times = f['observations/timestamp'][:]
            elif 'current_time' in f:
                times = f['current_time'][:]
            
            if times is not None and len(times) > 1:
                dt_list = np.diff(times)
                avg_dt = np.mean(dt_list)
                fps = 1.0 / avg_dt
                print(f"\nCalculated FPS from timestamps: {fps:.2f} (avg dt: {avg_dt:.4f}s)")
            
            # Check for other common locations
            if 'sim_freq' in f.attrs:
                print(f"Sim Frequency from attrs: {f.attrs['sim_freq']}")
            
    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_file(sys.argv[1])
    else:
        print("Please provide a file path.")
