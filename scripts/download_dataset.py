"""
Download CC12M Dataset
Downloads metadata and images for CC12M dataset
"""

import os
import sys
from pathlib import Path
import argparse
import requests
from PIL import Image
from io import BytesIO
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd


def download_tsv_from_gdrive(file_id, output_path):
    """Download TSV file from Google Drive"""
    print("Downloading CC12M TSV file from Google Drive...")
    
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0&confirm=t"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc="Downloading TSV",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"TSV file downloaded to: {output_path}")
    return output_path


def verify_tsv(tsv_path):
    """Verify CC12M TSV file"""
    if not os.path.exists(tsv_path):
        print(f"ERROR: TSV file not found: {tsv_path}")
        return False
    
    file_size = os.path.getsize(tsv_path)
    print(f"\nTSV file size: {file_size / (1024**2):.2f} MB")
    
    if file_size < 1024 * 1024:
        print("WARNING: TSV file seems too small!")
        return False
    
    print("Reading first 5 lines...")
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  Line {i+1}: {line.strip()[:80]}...")
            if i == 4:
                break
    
    print("Counting total lines...")
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total samples in TSV: {total_lines:,}")
    
    if total_lines < 100:
        print("WARNING: Very few samples!")
        return False
    
    print("✓ TSV file verification passed\n")
    return True


def download_image(args):
    """
    Download single image

    Args:
        args: tuple of (item, save_dir, timeout)
            item: (idx, url, caption) tuple
            save_dir: directory to save images
            timeout: download timeout in seconds

    Returns:
        success: bool indicating success
        result: (idx, image_path, caption, error) tuple
    """
    item, save_dir, timeout = args
    idx, url, caption = item

    try:
        # Download image
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Open and verify image
        img = Image.open(BytesIO(response.content))
        img.verify()

        # Re-open for saving (verify closes the image)
        img = Image.open(BytesIO(response.content))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save image
        image_filename = f"{idx:08d}.jpg"
        image_path = os.path.join(save_dir, image_filename)
        img.save(image_path, 'JPEG', quality=95)

        return True, (idx, image_path, caption, None)

    except Exception as e:
        return False, (idx, None, caption, str(e))


def process_tsv(tsv_path, output_dir, max_samples=None, num_workers=16):
    """
    Process CC12M TSV file and download images

    Args:
        tsv_path: path to cc12m.tsv file
        output_dir: directory to save processed dataset
        max_samples: maximum number of samples to download (None for all)
        num_workers: number of parallel download workers
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    print(f"Reading TSV file: {tsv_path}")

    # Read TSV - CC12M format is: caption \t url
    df = pd.read_csv(tsv_path, sep='\t', names=['caption', 'url'])

    if max_samples is not None:
        df = df.head(max_samples)

    print(f"Total samples to download: {len(df)}")

    # Prepare items for download
    items = [(idx, row['url'], row['caption']) for idx, row in df.iterrows()]

    # Download images in parallel
    print(f"Downloading images with {num_workers} workers...")

    # Prepare download arguments
    download_args = [(item, images_dir, 10) for item in items]

    successful_items = []
    failed_items = []

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(download_image, download_args),
            total=len(download_args),
            desc="Downloading images"
        ))

    # Collect results
    for success, result in results:
        if success:
            successful_items.append(result)
        else:
            failed_items.append(result)

    print(f"\nDownload complete!")
    print(f"  Successful: {len(successful_items)}")
    print(f"  Failed: {len(failed_items)}")

    # Save metadata
    metadata = {
        'samples': [
            {
                'idx': idx,
                'image_path': img_path,
                'caption': caption
            }
            for idx, img_path, caption, _ in successful_items
        ]
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

    # Save failed items log
    if failed_items:
        failed_path = os.path.join(output_dir, 'failed_downloads.json')
        failed_log = [
            {'idx': idx, 'caption': caption, 'error': error}
            for idx, _, caption, error in failed_items
        ]
        with open(failed_path, 'w') as f:
            json.dump(failed_log, f, indent=2)
        print(f"Failed downloads logged to: {failed_path}")

    return len(successful_items), len(failed_items)


def create_train_val_split(metadata_path, output_dir, val_ratio=0.05):
    """
    Split dataset into train and validation sets

    Args:
        metadata_path: path to metadata.json
        output_dir: directory to save splits
        val_ratio: fraction of data for validation
    """
    print(f"\nCreating train/val split (val_ratio={val_ratio})...")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    samples = metadata['samples']
    total = len(samples)
    val_size = int(total * val_ratio)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(samples)

    # Split
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    # Save splits
    train_metadata = {'samples': train_samples}
    val_metadata = {'samples': val_samples}

    train_path = os.path.join(output_dir, 'train_metadata.json')
    val_path = os.path.join(output_dir, 'val_metadata.json')

    with open(train_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)

    with open(val_path, 'w') as f:
        json.dump(val_metadata, f, indent=2)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Train metadata: {train_path}")
    print(f"Val metadata: {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Download CC12M dataset")
    parser.add_argument('--tsv_path', type=str, default=None,
                       help='Path to cc12m.tsv (auto-downloads if not provided)')
    parser.add_argument('--output_dir', type=str, default='./data/cc12m',
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of images to download (None for all)')
    parser.add_argument('--num_workers', type=int, default=16,
                       help='Number of download workers')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                       help='Validation split ratio')
    parser.add_argument('--gdrive_file_id', type=str,
                       default='1mZ_sHAp7jpMfFVY2TFN9wZioYujoYfCL',
                       help='Google Drive file ID for CC12M TSV')
    parser.add_argument('--skip_download_tsv', action='store_true',
                       help='Skip downloading TSV (use existing file)')
    
    args = parser.parse_args()
    
    # Determine TSV path
    if args.tsv_path is None:
        args.tsv_path = os.path.join(args.output_dir, 'cc12m.tsv')
    
    # Download TSV if needed
    if not args.skip_download_tsv:
        if not os.path.exists(args.tsv_path) or os.path.getsize(args.tsv_path) < 1024 * 1024:
            print("Downloading CC12M metadata TSV...")
            download_tsv_from_gdrive(args.gdrive_file_id, args.tsv_path)
        else:
            print(f"TSV file already exists: {args.tsv_path}")
    
    # Verify TSV
    if not verify_tsv(args.tsv_path):
        print("\nERROR: TSV verification failed!")
        print("The file may be corrupted or incomplete.")
        print("\nTo fix:")
        print(f"  1. Delete: {args.tsv_path}")
        print("  2. Run again to re-download")
        print("\nOr download manually:")
        print(f"  wget --no-check-certificate 'https://drive.usercontent.google.com/download?id={args.gdrive_file_id}&export=download&authuser=0&confirm=t' -O {args.tsv_path}")
        sys.exit(1)
    
    # Process TSV and download images
    successful, failed = process_tsv(
        args.tsv_path,
        args.output_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers
    )
    
    # Create train/val split
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        create_train_val_split(
            metadata_path,
            args.output_dir,
            val_ratio=args.val_ratio
        )
    
    print("\n" + "="*80)
    print("Dataset preparation complete!")
    print(f"Data saved to: {args.output_dir}")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print("\nNext steps:")
    print("  1. For small-scale testing (1000 images):")
    print(f"     python scripts/train.py --config test")
    print("  2. For full training:")
    print(f"     python scripts/train.py --config stage1")
    print("="*80)


if __name__ == "__main__":
    main()
