#!/usr/bin/env python3
"""
Standardized Policy Deployment Script for Competition
"""

import argparse
from pathlib import Path
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Standardized Policy Deployment for Competition")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-Agent-Direct-v0", help="Task name")
parser.add_argument("--policy_type", type=str, required=True, choices=['act', 'diffusion', 'bc', 'replay'], 
                    help="Policy type (act/diffusion/bc/replay)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint or data file (for replay)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--save_video", action="store_true", help="Save episode videos")
parser.add_argument("--temporal_agg", action="store_true", default=True, help="Use temporal aggregation (for ACT)")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force enable cameras as they are required for key policies
args_cli.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import gymnasium as gym
import torch
import numpy as np
import time

# Patch numpy for backward compatibility (loading 2.x pickle on 1.x numpy)
try:
    if not hasattr(np, "_core") and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.multiarray.scalars"] = np.core.multiarray
        print("Patched numpy._core for backward compatibility")
except Exception as e:
    print(f"Failed to patch numpy: {e}")

# Fix imports for ACT/DETR
# Add 'act' directory to path so 'detr' can be imported
act_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT/act"))
if act_path not in sys.path:
    sys.path.append(act_path)
    print(f"Added to sys.path: {act_path}")

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks
from Galaxea_Lab_External.VLA.ACT.policy_wrapper import ACTPolicyWrapper, DiffusionPolicyWrapper, BCPolicyWrapper, DataReplayPolicyWrapper

def load_replay_actions(data_path: str):
    import h5py
    print(f"\n{'='*80}")
    print("Loading replay data...")
    print(f"Data path: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        try:
            left_arm = f['/actions/left_arm_action'][:]
            right_arm = f['/actions/right_arm_action'][:]
            left_gripper = f['/actions/left_gripper_action'][:]
            right_gripper = f['/actions/right_gripper_action'][:]
            
            actions = np.concatenate([
                left_arm,
                right_arm,
                left_gripper[:, np.newaxis],
                right_gripper[:, np.newaxis],
            ], axis=1)
        except KeyError:
             print("KeyError reading actions, trying alternative format...")
             actions = f['action'][:]

        print(f"✓ Loaded {len(actions)} actions")
        print(f"  Action shape: {actions.shape}")
        return actions

def load_policy(policy_type: str, checkpoint_path: str, **kwargs):
    if policy_type == 'act':
        return ACTPolicyWrapper(checkpoint_path, temporal_agg=kwargs.get('temporal_agg', True))
    elif policy_type == 'diffusion':
        return DiffusionPolicyWrapper(checkpoint_path)
    elif policy_type == 'bc':
        return BCPolicyWrapper(checkpoint_path)
    elif policy_type == 'replay':
        return DataReplayPolicyWrapper(checkpoint_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def get_observations(env, policy_wrapper):
    device = policy_wrapper.device
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    
    left_arm_pos = robot.data.joint_pos[:, env_unwrapped._left_arm_joint_idx]
    right_arm_pos = robot.data.joint_pos[:, env_unwrapped._right_arm_joint_idx]
    left_gripper_pos = robot.data.joint_pos[:, env_unwrapped._left_gripper_dof_idx]
    right_gripper_pos = robot.data.joint_pos[:, env_unwrapped._right_gripper_dof_idx]
    
    joint_pos = torch.cat([
        left_arm_pos,
        right_arm_pos,
        left_gripper_pos,
        right_gripper_pos
    ], dim=-1)
    
    qpos = joint_pos.clone().to(device)
    
    camera_images = []
    camera_name_mapping = {
        'head_rgb': 'head_camera',
        'left_hand_rgb': 'left_hand_camera',
        'right_hand_rgb': 'right_hand_camera'
    }
    
    for cam_name in policy_wrapper.camera_names:
        sensor_name = camera_name_mapping.get(cam_name)
        if sensor_name is None:
            raise ValueError(f"Unknown camera name: {cam_name}")
        
        rgb_data = env.unwrapped.scene[sensor_name].data.output["rgb"]
        rgb_tensor = rgb_data.clone()
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        
        camera_images.append(rgb_tensor)
    
    images = torch.stack(camera_images, dim=1).to(device)
    return qpos, images

def run_episode(env, policy_wrapper, episode_idx: int, save_video: bool = False):
    if hasattr(policy_wrapper, 'reset'):
        policy_wrapper.reset()
    
    obs, _ = env.reset()
    
    episode_reward = 0.0
    episode_length = 0
    success = False
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    while True:
        qpos, images = get_observations(env, policy_wrapper)
        
        with torch.no_grad():
            action = policy_wrapper.predict(qpos, images)
        
        action_env = action.to(env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action_env)
        
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
            
        if episode_length > 2000:
             print("Max steps reached")
             break

    elapsed = time.time() - start_time
    print(f"\n{'-'*60}")
    print(f"Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}")
    print(f"  - Total Reward: {episode_reward:.2f}")
    print(f"  - Success: {success}")
    print(f"{'-'*60}")
    
    return episode_reward, episode_length, success

def run_replay_episode(env, replay_actions, episode_idx):
    print(f"\n{'='*80}")
    print(f"Episode {episode_idx + 1} - Replaying recorded actions")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    success = False
    start_time = time.time()
    
    for step_idx in range(len(replay_actions)):
        action_np = replay_actions[step_idx]
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.unwrapped.device)
        
        if step_idx % 50 == 0:
            print(f"\n[Replay Step {step_idx}/{len(replay_actions)}]")
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
    
    elapsed = time.time() - start_time
    print(f"\n{'-'*60}")
    print(f"Replay Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}")
    print(f"  - Success: {success}")
    print(f"{'-'*60}")
    return episode_reward, episode_length, success

def main():
    print("\n" + "="*60)
    print("STANDARDIZED POLICY DEPLOYMENT")
    print("="*60)
    
    if args_cli.policy_type == 'replay':
        replay_actions = load_replay_actions(args_cli.checkpoint)
        policy_wrapper = None
    else:
        print("\nLoading policy...")
        policy_wrapper = load_policy(
            args_cli.policy_type,
            args_cli.checkpoint,
            temporal_agg=args_cli.temporal_agg
        )
        replay_actions = None
    
    print(f"\nInitializing environment: {args_cli.task}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    
    if policy_wrapper is not None:
        required_freq = getattr(policy_wrapper, 'required_control_frequency', None)
        if required_freq is not None:
            required_decimation = int(1.0 / (env_cfg.sim.dt * required_freq))
            original_decimation = env_cfg.decimation
            env_cfg.decimation = required_decimation
            print(f"\nControl frequency adaptation: {required_freq} Hz (decimation={required_decimation})")
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"✓ Environment created")
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    try:
        for episode_idx in range(args_cli.num_episodes):
            if policy_wrapper is not None:
                reward, length, success = run_episode(env, policy_wrapper, episode_idx, save_video=args_cli.save_video)
            else:
                reward, length, success = run_replay_episode(env, replay_actions, episode_idx)
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            successes.append(success)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    if len(episode_rewards) > 0:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
        print("="*60)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
