import os
import requests
from huggingface_hub import list_repo_files
import time

repo_id = 'jonhpark/rocochallenge_2026_score_3'
local_dir = '/root/gearboxAssembly/data/datasets/temp_new_dataset'

# Get remote files
print("Fetching remote file list...")
all_files = list_repo_files(repo_id, repo_type='dataset')
hdf5_files = [f for f in all_files if f.endswith('.hdf5')]
print(f"Found {len(hdf5_files)} HDF5 files remotely.")

# Get local files
local_files = [f for f in os.listdir(local_dir) if f.endswith('.hdf5')]
print(f"Found {len(local_files)} HDF5 files locally.")

# Find missing
missing_files = [f for f in hdf5_files if f not in local_files]
print(f"Missing {len(missing_files)} files: {missing_files}")

if not missing_files:
    print("All files present.")
    exit(0)

# Download loop
base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"

for i, filename in enumerate(missing_files):
    url = f"{base_url}/{filename}"
    local_path = os.path.join(local_dir, filename)
    
    print(f"[{i+1}/{len(missing_files)}] Downloading {filename}...")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        # Clean up partial file
        if os.path.exists(local_path):
            os.remove(local_path)
    
    # Optional: check disk space
    s = os.statvfs('/')
    free_gb = (s.f_bavail * s.f_frsize) / (1024**3)
    print(f"Free space: {free_gb:.2f} GB")
    if free_gb < 10:
        print("DISK SPACE CRITICAL! Stopping download.")
        break
