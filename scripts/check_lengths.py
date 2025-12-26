import h5py
import os
import glob
import sys

dataset_dir = sys.argv[1]
files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))
files.sort()

print(f"Checking {len(files)} files in {dataset_dir}")
for i, f in enumerate(files):
    if i >= 10: break
    try:
        with h5py.File(f, 'r') as root:
            if 'action' in root:
                l = root['action'].shape[0]
            elif 'actions' in root:
                l = root['actions']['left_arm_action'].shape[0]
            else:
                l = 'Unknown'
            print(f"{os.path.basename(f)}: {l}")
    except Exception as e:
        print(f"{os.path.basename(f)}: Error - {e}")
