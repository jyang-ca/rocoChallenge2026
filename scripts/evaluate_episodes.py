#!/usr/bin/env python3
"""
Episode Evaluation Script for RoCoChallenge 2025

This script evaluates episodes from the rocochallenge2025 HuggingFace dataset
by replaying them in IsaacSim headless mode and computing the assembly score.

Usage:
    # Evaluate all episodes with chunked processing
    python scripts/evaluate_episodes.py \
        --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
        --task Isaac-GalaxeaLabAgent-Gearbox-Direct-v0 \
        --headless \
        --chunk-size 5

    # Evaluate specific episode range
    python scripts/evaluate_episodes.py \
        --input https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025 \
        --task Isaac-GalaxeaLabAgent-Gearbox-Direct-v0 \
        --headless \
        --start-episode 0 --end-episode 10 \
        --chunk-size 5

Author: AI Assistant
Date: 2025-12-24
"""

import argparse
import sys
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# Add argument parsing BEFORE IsaacLab imports
parser = argparse.ArgumentParser(description="Evaluate episodes from RoCoChallenge dataset")

# Input/Output arguments
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="HuggingFace repo URL or local directory path"
)
parser.add_argument(
    "--output",
    type=str,
    default="evaluation_results.txt",
    help="Output file for valid episode list"
)
parser.add_argument(
    "--local-dir",
    type=str,
    default="data/eval_cache",
    help="Local cache directory for downloaded files"
)

# Episode range arguments
parser.add_argument(
    "--start-episode",
    type=int,
    default=0,
    help="Start episode index (0-based, inclusive)"
)
parser.add_argument(
    "--end-episode",
    type=int,
    default=None,
    help="End episode index (exclusive). None = all remaining"
)

# Processing arguments
parser.add_argument(
    "--chunk-size",
    type=int,
    default=5,
    help="Number of episodes to process before cleanup"
)
parser.add_argument(
    "--min-score",
    type=int,
    default=1,
    help="Minimum score threshold for valid episodes"
)

# Isaac Lab arguments
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-GalaxeaLabAgent-Gearbox-Direct-v0",
    help="Name of the task environment"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments"
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations"
)

# Import AppLauncher and add its arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# Force headless mode for efficiency
args.headless = True
args.enable_cameras = True  # Need cameras for observation but not display

# Launch IsaacSim
print(f"\n{'='*80}")
print("Launching IsaacSim in headless mode...")
print(f"{'='*80}\n")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else after IsaacSim is launched
import gymnasium as gym
import torch
import numpy as np
import h5py
from huggingface_hub import HfApi, hf_hub_download

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks


def natural_sort_key(path: Path):
    """Natural sort key for filenames (1, 2, 10 order instead of 1, 10, 2)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem if isinstance(path, Path) else path)]


def extract_repo_id(path_or_url: str) -> str:
    """Extract repo_id from HuggingFace URL"""
    if path_or_url.startswith("http"):
        parts = path_or_url.rstrip("/").split("/")
        if "huggingface.co" in path_or_url and "datasets" in parts:
            idx = parts.index("datasets")
            if idx + 2 < len(parts):
                return f"{parts[idx+1]}/{parts[idx+2]}"
    return path_or_url


def is_remote_repo(path_or_id: str) -> bool:
    """Check if path is a remote HuggingFace repo"""
    if Path(path_or_id).exists():
        return False
    if path_or_id.startswith(("http://", "https://")):
        return True
    return "/" in path_or_id and not path_or_id.startswith(("/", "./", "../"))


def get_remote_hdf5_file_list(repo_id: str) -> List[str]:
    """Get sorted list of HDF5 files from remote HF repo"""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    hdf5_files = [f for f in files if f.endswith(".hdf5")]
    hdf5_files.sort(key=lambda x: natural_sort_key(Path(x)))
    return hdf5_files


def download_hdf5_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a single HDF5 file from HuggingFace"""
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir
    ))


def load_actions_from_hdf5(file_path: Path) -> Tuple[np.ndarray, int]:
    """Load action data from HDF5 file
    
    Returns:
        actions: Combined action array (N, 14)
        num_steps: Number of timesteps
    """
    with h5py.File(file_path, 'r') as f:
        left_arm = f['/actions/left_arm_action'][:]  # (N, 6)
        right_arm = f['/actions/right_arm_action'][:]  # (N, 6)
        left_gripper = f['/actions/left_gripper_action'][:]  # (N,)
        right_gripper = f['/actions/right_gripper_action'][:]  # (N,)
        
        # Combine: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
        actions = np.concatenate([
            left_arm,
            right_arm,
            left_gripper[:, np.newaxis],
            right_gripper[:, np.newaxis],
        ], axis=1)
        
        return actions, len(actions)


def evaluate_single_episode(
    env,
    actions: np.ndarray,
    device: str
) -> Tuple[int, float]:
    """Replay actions and compute final score
    
    Returns:
        score: Final assembly score
        time_cost: Time taken in simulation
    """
    env.reset()
    actions_tensor = torch.from_numpy(actions).float().to(device)
    
    num_steps = len(actions)
    
    for step_idx in range(num_steps):
        action = actions_tensor[step_idx:step_idx+1]  # (1, 14)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Get final score from environment
    unwrapped_env = env.unwrapped
    score, time_cost = unwrapped_env.evaluate_score()
    
    # Convert tensor to int if needed
    if isinstance(score, torch.Tensor):
        score = score.item() if score.numel() == 1 else int(score.sum().item())
    
    return int(score), float(time_cost)


def main():
    """Main evaluation loop with chunked processing"""
    
    print(f"\n{'='*80}")
    print("Episode Evaluation Script - RoCoChallenge 2025")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Min score: {args.min_score}")
    print(f"Episode range: {args.start_episode} to {args.end_episode or 'end'}")
    
    # Setup paths
    local_cache_dir = Path(args.local_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file list
    input_path = extract_repo_id(args.input)
    
    if is_remote_repo(args.input):
        print(f"\n📡 Fetching file list from HuggingFace: {input_path}")
        all_files = get_remote_hdf5_file_list(input_path)
    else:
        local_path = Path(args.input)
        all_files = sorted(local_path.glob("**/*.hdf5"), key=natural_sort_key)
        all_files = [str(f) for f in all_files]
    
    print(f"📂 Found {len(all_files)} total HDF5 files")
    
    # Apply episode range filter
    end_idx = args.end_episode if args.end_episode else len(all_files)
    target_files = all_files[args.start_episode:end_idx]
    print(f"🎯 Processing {len(target_files)} files (index {args.start_episode} to {end_idx-1})")
    
    # Create environment
    print(f"\n🎮 Creating environment: {args.task}")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric
    )
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device
    
    # Results storage
    valid_episodes = []
    all_results = []
    
    # Process in chunks
    num_chunks = (len(target_files) + args.chunk_size - 1) // args.chunk_size
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation: {num_chunks} chunks of {args.chunk_size} episodes")
    print(f"{'='*80}\n")
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(target_files))
        chunk_files = target_files[chunk_start:chunk_end]
        
        print(f"\n{'─'*60}")
        print(f"📦 Chunk {chunk_idx + 1}/{num_chunks}: Episodes {args.start_episode + chunk_start} to {args.start_episode + chunk_end - 1}")
        print(f"{'─'*60}")
        
        # Process each file in chunk
        for idx, filename in enumerate(chunk_files):
            global_idx = args.start_episode + chunk_start + idx
            file_basename = Path(filename).name
            
            print(f"\n[{global_idx}] 📥 Processing: {file_basename}")
            
            try:
                # Download if remote
                if is_remote_repo(args.input):
                    local_file = download_hdf5_file(input_path, filename, local_cache_dir)
                else:
                    local_file = Path(filename)
                
                # Load actions
                actions, num_steps = load_actions_from_hdf5(local_file)
                print(f"    📊 Loaded {num_steps} timesteps")
                
                # Evaluate
                print(f"    ⏳ Running simulation...")
                score, time_cost = evaluate_single_episode(env, actions, device)
                
                # Store result
                result = {
                    'index': global_idx,
                    'filename': file_basename,
                    'score': score,
                    'time_cost': time_cost,
                    'num_steps': num_steps
                }
                all_results.append(result)
                
                if score >= args.min_score:
                    valid_episodes.append(result)
                    print(f"    ✅ Score: {score} (time: {time_cost:.2f}s) - VALID")
                else:
                    print(f"    ❌ Score: {score} (time: {time_cost:.2f}s) - Invalid")
                
                # Delete downloaded file immediately if remote
                if is_remote_repo(args.input) and local_file.exists():
                    local_file.unlink()
                    
            except Exception as e:
                print(f"    ⚠️ Error: {str(e)}")
                all_results.append({
                    'index': global_idx,
                    'filename': file_basename,
                    'score': -1,
                    'error': str(e)
                })
        
        # Cleanup chunk cache directory
        if is_remote_repo(args.input):
            chunk_cache = local_cache_dir / input_path.replace("/", "_")
            if chunk_cache.exists():
                try:
                    shutil.rmtree(chunk_cache)
                    print(f"    🗑️ Cleaned up chunk cache")
                except Exception as e:
                    print(f"    ⚠️ Cache cleanup failed: {e}")
    
    # Close environment
    env.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Total episodes processed: {len(all_results)}")
    print(f"✅ Valid episodes (score >= {args.min_score}): {len(valid_episodes)}")
    print(f"❌ Invalid episodes: {len(all_results) - len(valid_episodes)}")
    
    if valid_episodes:
        print(f"\n📋 Valid Episodes:")
        print("-" * 60)
        for ep in valid_episodes:
            print(f"  [{ep['index']:3d}] {ep['filename']:40s} score={ep['score']}")
    
    # Save results to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write("# RoCoChallenge 2025 Episode Evaluation Results\n")
        f.write(f"# Input: {args.input}\n")
        f.write(f"# Min score: {args.min_score}\n")
        f.write(f"# Total processed: {len(all_results)}\n")
        f.write(f"# Valid episodes: {len(valid_episodes)}\n\n")
        
        f.write("## Valid Episodes\n")
        for ep in valid_episodes:
            f.write(f"{ep['index']},{ep['filename']},{ep['score']},{ep.get('time_cost', 0):.2f}\n")
        
        f.write("\n## All Results\n")
        for ep in all_results:
            status = "VALID" if ep.get('score', -1) >= args.min_score else "INVALID"
            f.write(f"{ep['index']},{ep['filename']},{ep.get('score', -1)},{status}\n")
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return len(valid_episodes)


if __name__ == "__main__":
    try:
        num_valid = main()
        print(f"\n✅ Evaluation completed successfully. Found {num_valid} valid episodes.")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
