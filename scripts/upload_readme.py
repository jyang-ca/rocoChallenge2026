import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload README.md to Hugging Face")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    args = parser.parse_args()

    readme_content = """---
tags:
- robotics
- imitation-learning
- act
- isaac-lab
- galaxea
library_name: generic
---

# ACT Policy for Gearbox Assembly

This model is an **Action Chunking with Transformers (ACT)** policy trained for a gearbox assembly task in Isaac Lab.

## Model Details
- **Architecture**: ACT (Action Chunking with Transformers)
- **Task**: Gearbox Assembly
- **Framework**: Isaac Lab / PyTorch
- **Dataset**: `sim_gearbox_assembly_demos_filtered`

## Training Data Usage
The model was trained using **all available episodes** from the provided dataset.

### Important Data Characteristics
- **Process Simplification**: The **pin insertion step was omitted** in all training episodes. The task focuses on the steps following pin insertion (e.g., gear mounting, cover installation).
- **Data Quality Note**: The training set deliberately includes one known **failed episode** where a pin fell over during the process, causing the subsequent cover installation to fail. This episode was **not filtered out** and was used in training.

## Hyperparameters
The model was trained with the following configuration:

| Parameter | Value |
|:---|:---|
| **Policy Class** | ACT |
| **KL Weight** | 10 |
| **Chunk Size** | 100 |
| **Hidden Dim** | 512 |
| **Feedforward Dim** | 3200 |
| **Batch Size** | 32 |
| **Learning Rate** | 1e-5 |
| **Epochs** | 10000 |
| **Seed** | 0 |
| **Save Interval** | 200 |
| **Early Stopping** | 3000 |

## Usage
This model is designed to be loaded with the ACT policy wrapper in the Galaxea Lab framework for the `GalaxeaLab-GearboxAssembly` environment.
"""

    api = HfApi(token=args.token)
    
    print(f"Uploading README.md to https://huggingface.co/{args.repo_id} ...")
    try:
        # Upload content directly as a file
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("✓ README.md upload completed successfully!")
        print(f"View your model card at: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Error uploading README.md: {e}")

if __name__ == "__main__":
    main()
