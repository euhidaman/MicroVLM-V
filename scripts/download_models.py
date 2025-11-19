"""
Download Model Weights
Downloads Qwen2.5-0.5B and DeiT-Tiny pretrained weights
"""

import os
import sys
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_qwen_model(output_dir="./models/qwen2.5-0.5B"):
    """
    Download Qwen2.5-0.5B from HuggingFace
    
    Args:
        output_dir: directory to save model
    """
    print("Downloading Qwen2.5-0.5B from HuggingFace...")
    
    model_id = "Qwen/Qwen2.5-0.5B"
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"Qwen2.5-0.5B downloaded successfully to {output_dir}")
        print(f"Model size: ~988 MB")
        
    except Exception as e:
        print(f"Error downloading Qwen2.5-0.5B: {e}")
        print("Please ensure you have internet connection and huggingface-hub installed:")
        print("  pip install huggingface-hub")


def download_deit_model(output_dir="./models/deit-tiny"):
    """
    Download DeiT-Tiny from HuggingFace
    
    Args:
        output_dir: directory to save model
    """
    print("Downloading DeiT-Tiny from HuggingFace...")
    
    model_id = "facebook/deit-tiny-patch16-224"
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"DeiT-Tiny downloaded successfully to {output_dir}")
        print(f"Model size: ~20 MB")
        
    except Exception as e:
        print(f"Error downloading DeiT-Tiny: {e}")
        print("Please ensure you have internet connection and huggingface-hub installed:")
        print("  pip install huggingface-hub")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument('--model', type=str, choices=['qwen', 'deit', 'both'],
                       default='both', help='Which model to download')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ['qwen', 'both']:
        qwen_dir = output_dir / "qwen2.5-0.5B"
        download_qwen_model(str(qwen_dir))
    
    if args.model in ['deit', 'both']:
        deit_dir = output_dir / "deit-tiny"
        download_deit_model(str(deit_dir))
    
    print("\nDownload complete!")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
