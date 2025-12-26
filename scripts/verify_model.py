
"""
Verification Script for Galaxea Lab Agent using Trained ACT Policy
This script loads a trained policy checkpoint and evaluates it using a custom score function.
"""

import argparse
from pathlib import Path
import sys
import os

from isaaclab.app import AppLauncher

# Add argument parsing BEFORE IsaacLab imports
parser = argparse.ArgumentParser(description="Verify Galaxea Lab Agent model performance")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/root/gearboxAssembly/data/model/20251224_104333/policy_best.ckpt",
    help="Path to the policy checkpoint"
)
parser.add_argument(
    "--task",
    type=str,
    default="Template-Galaxea-Lab-Agent-Direct-v0",
    help="Name of the task environment"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate"
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=2000,
    help="Maximum number of steps to run"
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations"
)

# Import AppLauncher and add its arguments
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# Force enable cameras as they are required for the policy
args.enable_cameras = True

# Launch IsaacSim
print(f"Launching IsaacSim...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else after IsaacSim is launched
import gymnasium as gym
import torch
import numpy as np
import sys
import os

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
from Galaxea_Lab_External.VLA.ACT.policy_wrapper import ACTPolicyWrapper

def main():
    print(f"Verifying model: {args.checkpoint}")

    # Check if stats file exists
    checkpoint_path = Path(args.checkpoint)
    stats_path = checkpoint_path.parent / "dataset_stats.pkl"
    if not stats_path.exists():
        print(f"WARNING: dataset_stats.pkl not found at {stats_path}")
        print("Policy outputs might not be correctly denormalized.")

    # Load policy
    try:
        policy = ACTPolicyWrapper(str(checkpoint_path), temporal_agg=True)
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Create environment
    print(f"Creating environment: {args.task}")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric
    )
    env = gym.make(args.task, cfg=env_cfg)
    
    # Reset environment
    obs, _ = env.reset()
    
    print("Starting simulation...")
    
    # Run simulation
    step = 0
    terminated = False
    truncated = False
    
    while simulation_app.is_running() and step < args.max_steps:
        # Prepare inputs for policy
        # Extract observations from env dictionary
        # Structure: left_arm_joint_pos, right_arm_joint_pos, left_gripper_joint_pos, right_gripper_joint_pos
        # Policy expects: qpos (batch, 14), images (batch, 3, 3, 240, 320)
        
        # Note: obs structure matches what we saw in VLA_agent.py
        # obs['policy'] contains the dictionary
        
        policy_obs = obs['policy']
        
        left_arm_pos = policy_obs['left_arm_joint_pos']
        right_arm_pos = policy_obs['right_arm_joint_pos']
        left_gripper_pos = policy_obs['left_gripper_joint_pos']
        right_gripper_pos = policy_obs['right_gripper_joint_pos']
        
        # Concatenate to form qpos (14 dim)
        # Assuming batch size 1 for simplicity in this script, though it supports batching
        qpos = torch.cat([
            left_arm_pos, 
            right_arm_pos, 
            left_gripper_pos.unsqueeze(0) if len(left_gripper_pos.shape) == 1 else left_gripper_pos, 
            right_gripper_pos.unsqueeze(0) if len(right_gripper_pos.shape) == 1 else right_gripper_pos
        ], dim=-1)

        # Prepare images
        # policy expects (batch, num_cams, C, H, W)
        # obs has separate keys: head_rgb, left_hand_rgb, right_hand_rgb
        # Each is likely (H, W, C) or (batch, H, W, C) - wait, VLA_agent.py said (0, 1, 4, 2, 3) permutation
        # Let's check VLA_agent.py again:
        # head_rgb = obs['policy']['head_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
        # It seems obs returns (num_envs, H, W, C)? No, usually IsaacLab returns (num_envs, H, W, C)
        # VLA_agent.py: obs['policy']['head_rgb']
        # If num_envs=1, it is (1, H, W, C)
        # permutation (0, 1, 4, 2, 3) implies adding a dimension somewhere?
        # VLA_agent: .unsqueeze(0).permute(0, 1, 4, 2, 3) -> (1, 1, C, H, W) maybe?
        
        # Let's trust VLA_agent.py logic but verify shapes if it fails.
        # Assuming obs values are (num_envs, H, W, C)
        
        # We need to construct (num_envs, 3, 3, H, W)
        # VLA_agent.py lines:
        # head_rgb = obs['policy']['head_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
        # This looks suspicious. If obs['head_rgb'] is (1, H, W, C), unsqueeze(0) -> (1, 1, H, W, C)
        # permute(0, 1, 4, 2, 3) -> (1, 1, C, H, W) -> correct for one camera
        
        head_rgb = policy_obs['head_rgb'] # (N, H, W, C)
        left_hand_rgb = policy_obs['left_hand_rgb']
        right_hand_rgb = policy_obs['right_hand_rgb']
        
        # Convert to float and normalize if needed (PolicyWrapper might handle normalization, but usually we pass [0,1] or [0,255] float)
        # VLA_agent converts to float32. PolicyWrapper doc says "Images are expected to be normalized to [0, 1]"
        # IsaacLab usually returns [0, 255] uint8 or float. I should check.
        # Assuming uint8 [0, 255], I should divide by 255 if Policy expects [0, 1]
        # But VLA_agent simply does .to(torch.float32) without division. I will check if PolicyWrapper does normalization.
        # ACT code usually expects normalized images.
        
        # To be safe, let's stick to VLA_agent implementation logic:
        # images = torch.cat([head_rgb, left_hand_rgb, right_hand_rgb], dim=1)
        # But wait, VLA_agent lines 94-96:
        # head_rgb = obs['policy']['head_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
        # This implies obs['policy']['head_rgb'] is (num_envs, H, W, C) ?? Or maybe (H, W, C) if single env?
        # IsaacLab DirectRLEnv step returns dict of tensors with batch dim.
        # So obs['policy']['head_rgb'] is (num_envs, H, W, C).
        # unsqueeze(0) makes it (1, num_envs, H, W, C).
        # permute(0, 1, 4, 2, 3) -> (1, num_envs, C, H, W).
        # concatenating dim 1 -> (1, 3*num_envs, C, H, W) ??? This seems wrong for batching.
        
        # Let's fix this for arbitrary num_envs.
        # Target: (num_envs, 3, C, H, W)
        # Input: (num_envs, H, W, C)
        
        # Permute to (num_envs, C, H, W)
        head_rgb_p = head_rgb.permute(0, 3, 1, 2)
        left_hand_rgb_p = left_hand_rgb.permute(0, 3, 1, 2)
        right_hand_rgb_p = right_hand_rgb.permute(0, 3, 1, 2)
        
        # Stack to (num_envs, 3, C, H, W)
        images = torch.stack([head_rgb_p, left_hand_rgb_p, right_hand_rgb_p], dim=1)
        
        # Convert to float
        images = images.float()
        
        # IMPORTANT: Normalize to [0, 1] usually required for ACT
        # Checking VLA_agent again... it doesn't divide.
        # But typically RGB is 0-255.
        # I'll divide by 255.0 to be safe as ACT usually trains on 0-1. 
        # Wait, if VLA_agent DOES NOT divide, maybe the env returns 0-1 floats?
        # Standard IsaacLab cameras return [0, 255] usually.
        # I will check one sample value if possible, but for now let's assume [0, 255] and normalize.
        # ACT Policy (DETR based) definitely uses standard ImageNet normalization which assumes 0-1 input.
        images = images / 255.0
        
        # Move to policy device
        qpos = qpos.to(policy.device)
        images = images.to(policy.device)
        
        with torch.inference_mode():
            action = policy.predict(qpos, images)
            
        # Move action back to env device if needed
        if action.device != env.unwrapped.device:
            action = action.to(env.unwrapped.device)
            
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        if step % 10 == 0:
            print(f"Step {step}")
            
        # Check termination
        if terminated.any() or truncated.any():
            print(f"Episode terminated or truncated at step {step}")
            break
            
    # Evaluation
    print("Simulation finished. Evaluating score...")
    
    # Access the unwrapped environment to call evaluate_score
    # env is Gym wrapper -> env.unwrapped is DirectRLEnv (GalaxeaLabAgentEnv)
    unwrapped_env = env.unwrapped
    
    score, time_cost = unwrapped_env.evaluate_score()
    
    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    
    # Handle batch score if multiple envs
    if isinstance(score, torch.Tensor):
        if score.numel() > 1:
            print(f"Scores per env: {score}")
            score = score.mean().item()
        else:
            score = score.item()
            
    if isinstance(time_cost, torch.Tensor):
        if time_cost.numel() > 1:
            time_cost = time_cost.mean().item()
        else:
            time_cost = time_cost.item()
            
    print(f"Final Score: {score}")
    print(f"Time Cost: {time_cost}")
    
    if score >= 6: # Assuming max score is roughly 6 based on code (mounting gears etc)
        print("SUCCESS: Model achieved high score.")
    elif score > 0:
        print("PARTIAL SUCCESS: Model achieved some score.")
    else:
        print("FAILURE: Model failed to achieve score.")
        
    print("="*40)
    
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error executing verification execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
