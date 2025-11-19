"""
Download CC12M Dataset
Downloads metadata and images for CC12M dataset
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cc12m_loader import (
    download_cc12m_metadata,
    download_cc12m_images,
    create_small_test_dataset
)


def main():
    parser = argparse.ArgumentParser(description="Download CC12M dataset")
    parser.add_argument('--mode', type=str, 
                       choices=['metadata', 'images', 'test', 'all'],
                       default='all',
                       help='What to download')
    parser.add_argument('--output_dir', type=str, default='./data/cc12m',
                       help='Output directory')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to download (None for all)')
    parser.add_argument('--test_samples', type=int, default=1000,
                       help='Number of samples for test dataset')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of download workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed error messages for failed downloads')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download metadata
    if args.mode in ['metadata', 'all']:
        print("Downloading CC12M metadata...")
        download_cc12m_metadata(str(output_dir))
        print("Metadata download complete")
    
    # Create test subset
    if args.mode in ['test', 'all']:
        print(f"\nCreating test dataset with {args.test_samples} samples...")
        metadata_file = output_dir / "cc12m_metadata_0.tsv"
        test_file = output_dir / f"cc12m_test_{args.test_samples}.tsv"
        
        if metadata_file.exists():
            create_small_test_dataset(
                str(metadata_file),
                str(test_file),
                num_samples=args.test_samples
            )
            print(f"Test metadata saved to {test_file}")
        else:
            print(f"Metadata file not found: {metadata_file}")
            print("Please download metadata first")
    
    # Download images
    if args.mode in ['images', 'all']:
        print("\nDownloading images...")
        
        # Determine which metadata to use - prefer test file if it exists
        test_file = output_dir / f"cc12m_test_{args.test_samples}.tsv"
        full_file = output_dir / "cc12m_metadata_0.tsv"
        
        if test_file.exists() and args.max_images and args.max_images <= args.test_samples:
            metadata_file = test_file
            print(f"Using test metadata: {metadata_file}")
        elif full_file.exists():
            metadata_file = full_file
            print(f"Using full metadata: {metadata_file}")
        else:
            print(f"Metadata file not found. Tried:")
            print(f"  - {test_file}")
            print(f"  - {full_file}")
            print("Please download metadata first using --mode metadata or --mode test")
            return
        
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        download_cc12m_images(
            str(metadata_file),
            str(image_dir),
            max_images=args.max_images,
            num_workers=args.num_workers,
            verbose=args.verbose
        )
        
        print(f"Images saved to {image_dir}")
    
    print("\nDataset preparation complete!")
    print(f"Data saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. For small-scale testing (1000 images):")
    print(f"     python scripts/train.py --config test")
    print("  2. For full training:")
    print(f"     python scripts/train.py --config stage1")


if __name__ == "__main__":
    main()
