import os

src_dir = '/root/gearboxAssembly/data/datasets/rocochallenge2025/gearbox_assembly_demos_updated'
files = [f for f in os.listdir(src_dir) if f.endswith('.hdf5')]
valid_files = []
for f in files:
    path = os.path.join(src_dir, f)
    if os.path.getsize(path) > 50 * 1024 * 1024: # > 50MB just to be safe (they are ~900MB)
        valid_files.append(f)

print(f"Found {len(valid_files)} valid files.")
# Sort by the numeric value if possible, else string
# Original names are like '1.hdf5', '10.hdf5'. 
# We should probably sort them numerically to keep some order, though not strictly required.
def parse_int(name):
    try:
        return int(name.split('.')[0])
    except:
        return 999999

valid_files.sort(key=parse_int)

for idx, f in enumerate(valid_files):
    src = os.path.join(src_dir, f)
    dst = os.path.join(src_dir, f'episode_{idx}.hdf5')
    if src != dst:
         # print(f'Renaming {f} -> episode_{idx}.hdf5')
         os.rename(src, dst)

print(f"Renamed {len(valid_files)} files to episode_0..{len(valid_files)-1}")
