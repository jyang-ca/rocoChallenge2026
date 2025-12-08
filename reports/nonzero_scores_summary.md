## HDF5 score scan (rocochallenge2025/rocochallenge2025)

- Command: `python scripts/find_nonzero_scores.py --max-keep 10`
- Cache dir: `data/cache`
- Kept first 10 files with any non-zero score; zero-only files were deleted after inspection.

### Kept (non-zero scores)
| file | frames | min | max |
| --- | --- | --- | --- |
| gearbox_assembly_demos/data_20251127_213235.hdf5 | 664 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251127_222735.hdf5 | 664 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251127_233844.hdf5 | 664 | 0.0 | 4.0 |
| gearbox_assembly_demos/data_20251128_001933.hdf5 | 664 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251128_003957.hdf5 | 664 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251128_013056.hdf5 | 639 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251128_014108.hdf5 | 640 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251128_025233.hdf5 | 641 | 0.0 | 3.0 |
| gearbox_assembly_demos/data_20251128_035345.hdf5 | 643 | 0.0 | 2.0 |
| gearbox_assembly_demos/data_20251128_050509.hdf5 | 662 | 0.0 | 3.0 |

### Removed (score all zero)
| file | frames |
| --- | --- |
| gearbox_assembly_demos/data_20251127_212217.hdf5 | 585 |
| gearbox_assembly_demos/data_20251127_235930.hdf5 | 585 |
| gearbox_assembly_demos/data_20251128_000935.hdf5 | 585 |
| gearbox_assembly_demos/data_20251128_032309.hdf5 | 585 |
| gearbox_assembly_demos/data_20251128_034348.hdf5 | 585 |

No files were missing the `score` dataset in this scan.
