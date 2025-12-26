import torch
import numpy as np
import os
import pickle
import argparse
import sys

# Add ACT module to path
sys.path.append('/root/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT')
sys.path.append('/root/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT/act')

from act.utils import load_data, compute_dict_mean, detach_dict
from act.policy import ACTPolicy

def main():
    # Hardcoded configuration matching the training script
    DATA_DIR = '/root/gearboxAssembly/data/datasets/rocochallenge2025'
    dataset_dir = os.path.join(DATA_DIR, 'gearbox_assembly_demos_updated')
    ckpt_path = '/root/gearboxAssembly/data/model/20251226_051106/policy_epoch_2800_seed_0.ckpt'
    num_episodes = 192
    camera_names = ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
    
    # Policy Config
    policy_config = {
        'lr': 1e-5,
        'num_queries': 100,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names
    }

    print(f"Loading model from: {ckpt_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build policy
    try:
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded: {loading_status}")
        policy.to(device)
        policy.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Data
    print(f"Loading validation dataset from: {dataset_dir}")
    # Using batch_size=32 as in training
    _, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, 32, 32)

    print(f"Starting Offline Verification (Open-Loop)...")
    
    val_history = []
    with torch.inference_mode():
        for batch_idx, data in enumerate(val_dataloader):
            image, qpos, action, is_pad = data
            image, qpos, action, is_pad = image.to(device), qpos.to(device), action.to(device), is_pad.to(device)
            
            # Forward pass (calculates loss similar to training)
            loss_dict = policy(qpos, image, action, is_pad)
            val_history.append(detach_dict(loss_dict))
            
            if batch_idx % 5 == 0:
                print(f"Processed batch {batch_idx}")

    epoch_summary = compute_dict_mean(val_history)
    
    print("\n" + "="*50)
    print("OFFLINE VERIFICATION RESULTS")
    print("="*50)
    print(f"Validation Loss (L1 + KL): {epoch_summary['loss']:.5f}")
    print(f"L1 Loss (Action Error):      {epoch_summary['l1']:.5f}")
    print(f"KL Divergence:               {epoch_summary['kl']:.5f}")
    print("="*50)
    
    if epoch_summary['l1'] < 0.5:
        print("Verdict: Model appears to have learned reliable behavior (Low L1 Error).")
    else:
        print("Verdict: Model error is still relatively high. Further training might be needed.")

if __name__ == '__main__':
    main()
