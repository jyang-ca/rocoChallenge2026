import torch
import numpy as np
import os
import pickle
from copy import deepcopy
from einops import rearrange

# Import model definitions
# Assuming we are running from root or can access these via path
import sys
# sys.path.append("/root/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT")

try:
    from act.policy import ACTPolicy, CNNMLPPolicy
    from act.utils import compute_dict_mean, set_seed, detach_dict
except ImportError:
    # If standard import fails, try relative or adding path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'act'))
    from policy import ACTPolicy, CNNMLPPolicy

class ACTPolicyWrapper:
    def __init__(self, checkpoint_path, temporal_agg=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temporal_agg = temporal_agg
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint and stats
        self.policy, self.stats, self.policy_config = self._load_policy_and_stats()
        self.policy.to(self.device)
        self.policy.eval()
        
        # ACT specific state
        self.query_frequency = self.policy_config['num_queries']
        self.num_queries = self.policy_config['num_queries']
        if self.temporal_agg:
            self.query_frequency = 1
            
        self.all_time_actions = None
        self.step_counter = 0
        
        # Configuration
        self.camera_names = self.policy_config['camera_names']
        # Default freq, can be overridden if known
        self.required_control_frequency = 50 

    def _load_policy_and_stats(self):
        # Load stats
        ckpt_dir = os.path.dirname(self.checkpoint_path)
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        
        # If policy_best, stats might be in same dir
        if not os.path.exists(stats_path):
             # Try going up one level if checkpoint is inside a subdir?
             # Or assume it is there.
             pass

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # Load policy
        # We need to reconstruct config. Ideally it is saved or we infer it.
        # For this specific task, we know the params from user request history or config.json
        
        # Try to load config.json if it exists
        config_path = os.path.join(ckpt_dir, 'config.json')
        policy_config = None
        if os.path.exists(config_path):
             import json
             with open(config_path, 'r') as f:
                 saved_config = json.load(f)
                 # Map saved config to policy config
                 policy_config = {
                    'lr': saved_config.get('lr', 1e-5),
                    'num_queries': saved_config.get('chunk_size', 100),
                    'kl_weight': saved_config.get('kl_weight', 10),
                    'hidden_dim': saved_config.get('hidden_dim', 512),
                    'dim_feedforward': saved_config.get('dim_feedforward', 3200),
                    'lr_backbone': 1e-5,
                    'backbone': 'resnet18',
                    'enc_layers': 4,
                    'dec_layers': 7,
                    'nheads': 8,
                    'camera_names': ['head_camera', 'left_hand_camera', 'right_hand_camera'] # Hardcoded or needs to come from config
                 }
        
        if policy_config is None:
             # Fallback to defaults we know
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
                'camera_names': ['head_camera', 'left_hand_camera', 'right_hand_camera']
             }

        policy = ACTPolicy(policy_config)
        
        # Load weights
        loading_status = policy.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        print(f"Loaded policy from {self.checkpoint_path}")
        print(loading_status)
        
        return policy, stats, policy_config

    def reset(self):
        self.step_counter = 0
        self.all_time_actions = torch.zeros([10000, 10000 + self.num_queries, 14]).to(self.device) # Max steps assumption
        
    def predict(self, qpos, images):
        # qpos: (1, 14)
        # images: (1, num_cams, 3, H, W)
        
        # Preprocess
        qpos_np = qpos.cpu().numpy().squeeze(0)
        qpos_norm = (qpos_np - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos_input = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
        
        # Images are already tensors from wrapper, just ensure shape/device
        # Expected: (1, num_cams, 3, H, W) -> remove batch for rearrange -> (num_cams, 3, H, W) ???
        # BUT ACTPolicy expects: (batch, num_cams, 3, H, W) check forward
        # forward(self, qpos, image, actions=None, is_pad=None)
        # image shape expected: (batch, num_cameras, channel, height, width)
        
        # ACT `get_image` usually stacks them. 
        # Our wrapper `get_observations` returns (batch, num_cameras, 3, H, W).
        # We need to make sure we pass it correctly.
        image_input = images # Already correct
        
        t = self.step_counter
        
        if self.temporal_agg:
            if t % self.query_frequency == 0:
                all_actions = self.policy(qpos_input, image_input) # (1, chunk_size, 14)
                # Store
                self.all_time_actions[[t], t:t+self.num_queries] = all_actions
                
            actions_for_curr_step = self.all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            
        else:
            if t % self.query_frequency == 0:
                self.curr_actions = self.policy(qpos_input, image_input)
            raw_action = self.curr_actions[:, t % self.query_frequency]

        # Post-process
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = raw_action * self.stats['action_std'] + self.stats['action_mean']
        
        self.step_counter += 1
        return torch.from_numpy(action).float().unsqueeze(0).to(self.device)

# Placeholders for other policies
class DiffusionPolicyWrapper:
    def __init__(self, checkpoint_path):
        raise NotImplementedError("Diffusion not implemented yet")

class BCPolicyWrapper:
     def __init__(self, checkpoint_path):
        raise NotImplementedError("BC not implemented yet")

class DataReplayPolicyWrapper:
     def __init__(self, checkpoint_path):
        pass 
