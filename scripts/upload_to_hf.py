import argparse
import json
import os
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoints and config to Hugging Face")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, help="Hugging Face API token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    args = parser.parse_args()

    # Source directory containing the checkpoints
    source_dir = "/root/gearboxAssembly/data/model/20251224_104333"
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Define configuration from user request
    config = {
        "task_name": "sim_gearbox_assembly_demos_filtered",
        "ckpt_dir": "/root/gearboxAssembly/data/model",
        "policy_class": "ACT",
        "kl_weight": 10,
        "chunk_size": 100,
        "hidden_dim": 512,
        "batch_size": 32,
        "dim_feedforward": 3200,
        "num_epochs": 10000,
        "lr": 1e-5,
        "seed": 0,
        "save_interval": 200,
        "early_stopping_patience": 3000
    }

    # Save config.json
    config_path = os.path.join(source_dir, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"✓ Created config.json at {config_path}")
    except Exception as e:
        print(f"Error creating config.json: {e}")
        return

    # Initialize API
    api = HfApi(token=args.token)

    # Create repository
    print(f"Creating/Checking repository: {args.repo_id}")
    try:
        create_repo(
            args.repo_id, 
            token=args.token, 
            exist_ok=True, 
            private=args.private,
            repo_type="model"
        )
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload folder
    print(f"Uploading files from {source_dir} to https://huggingface.co/{args.repo_id} ...")
    try:
        api.upload_folder(
            folder_path=source_dir,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("✓ Upload completed successfully!")
        print(f"View your model at: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    main()
