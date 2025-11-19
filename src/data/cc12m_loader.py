"""
CC12M Dataset Loader
Downloads and processes Conceptual Captions 12M dataset
"""

import os
import json
import requests
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import io


class CC12MDataset(Dataset):
    """
    Conceptual Captions 12M dataset
    """
    
    def __init__(self, metadata_file, image_dir=None, transform=None, max_samples=None,
                 tokenizer=None, max_length=77):
        """
        Args:
            metadata_file: path to metadata JSON file (train_metadata.json or val_metadata.json)
            image_dir: directory containing downloaded images (optional, can be in metadata)
            transform: image transformations
            max_samples: limit number of samples (for testing)
            tokenizer: text tokenizer
            max_length: maximum text length
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load metadata
        print(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.samples = metadata['samples']
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Default transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get image-caption pair"""
        sample = self.samples[idx]
        caption = sample['caption']
        image_path = sample['image_path']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback to placeholder
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'image': image,
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'caption': caption
            }
        else:
            return {
                'image': image,
                'caption': caption
            }


def download_cc12m_metadata(output_dir):
    """
    Download CC12M metadata
    
    Args:
        output_dir: directory to save metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cc12m.tsv exists (user downloaded manually)
    manual_file = output_dir / "cc12m.tsv"
    target_file = output_dir / "cc12m_metadata_0.tsv"
    
    if manual_file.exists() and not target_file.exists():
        print(f"Found manually downloaded file: {manual_file}")
        print(f"Renaming to: {target_file}")
        manual_file.rename(target_file)
        print("Metadata file ready")
        return
    
    if target_file.exists():
        print(f"Metadata file {target_file} already exists, skipping")
        return
    
    # CC12M metadata URLs (split into shards)
    metadata_urls = [
        "https://storage.googleapis.com/conceptual_12m/cc12m.tsv"
    ]
    
    print("Downloading CC12M metadata...")
    
    for i, url in enumerate(metadata_urls):
        output_file = output_dir / f"cc12m_metadata_{i}.tsv"
        
        if output_file.exists():
            print(f"Metadata file {output_file} already exists, skipping")
            continue
        
        print(f"Downloading from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Saved to {output_file}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            print("\nAlternative: Download manually from:")
            print("https://drive.usercontent.google.com/download?id=1mZ_sHAp7jpMfFVY2TFN9wZioYujoYfCL&export=download&authuser=0")
            print(f"Then save as: {output_file}")
    
    print("Metadata download complete")


def download_cc12m_images(metadata_file, output_dir, max_images=None, num_workers=8, verbose=False):
    """
    Download CC12M images from URLs in metadata
    
    Args:
        metadata_file: path to metadata TSV
        output_dir: directory to save images
        max_images: maximum number of images to download
        num_workers: number of parallel download workers
        verbose: print detailed error messages
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='\t', names=['caption', 'url'])
    
    if max_images is not None:
        metadata = metadata.head(max_images)
    
    print(f"Downloading {len(metadata)} images...")
    
    # Track failed URLs for debugging
    failed_urls = []
    
    def download_image(idx_url):
        idx, url = idx_url
        image_name = f"{idx:08d}.jpg"
        image_path = output_dir / image_name
        
        if image_path.exists():
            return True
        
        # Retry logic
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                # Add user agent to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, timeout=15, headers=headers, stream=True)
                response.raise_for_status()
                
                # Verify it's an image
                img = Image.open(io.BytesIO(response.content))
                img = img.convert('RGB')
                
                # Save
                img.save(image_path, 'JPEG', quality=95)
                return True
                
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout: {e}"
                if attempt == max_retries - 1:
                    break
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                break
            except Exception as e:
                last_error = f"Image processing error: {e}"
                break
        
        if verbose and last_error:
            failed_urls.append((idx, url, last_error))
        return False
    
    # Download with progress bar
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_image, (idx, row['url'])): idx 
            for idx, row in metadata.iterrows()
        }
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({'success': successful, 'failed': failed})
    
    print(f"\nDownload complete: {successful} successful, {failed} failed")
    
    if verbose and failed_urls:
        print(f"\nFirst 10 failed URLs:")
        for idx, url, error in failed_urls[:10]:
            print(f"  {idx}: {url}")
            print(f"     Error: {error}")


def create_small_test_dataset(metadata_file, output_metadata, num_samples=1000):
    """
    Create small test dataset
    
    Args:
        metadata_file: full metadata file
        output_metadata: output path for subset
        num_samples: number of samples to include
    """
    metadata = pd.read_csv(metadata_file, sep='\t', names=['caption', 'url'])
    
    # Sample randomly
    subset = metadata.sample(n=min(num_samples, len(metadata)), random_state=42)
    
    # Save
    subset.to_csv(output_metadata, sep='\t', header=False, index=False)
    
    print(f"Created test dataset with {len(subset)} samples at {output_metadata}")


def create_dataloaders(train_metadata_file, val_metadata_file, tokenizer, 
                       batch_size=32, num_workers=4, max_samples=None):
    """
    Create train and validation dataloaders
    
    Args:
        train_metadata_file: path to train_metadata.json
        val_metadata_file: path to val_metadata.json
        tokenizer: text tokenizer
        batch_size: batch size
        num_workers: number of workers
        max_samples: limit samples for testing
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = CC12MDataset(
        metadata_file=train_metadata_file,
        tokenizer=tokenizer,
        max_samples=max_samples
    )
    
    val_dataset = CC12MDataset(
        metadata_file=val_metadata_file,
        tokenizer=tokenizer,
        max_samples=None  # Use all validation data
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_metadata', action='store_true')
    parser.add_argument('--download_images', action='store_true')
    parser.add_argument('--create_test_set', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./data/cc12m')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--test_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.download_metadata:
        download_cc12m_metadata(args.output_dir)
    
    if args.download_images:
        metadata_file = Path(args.output_dir) / "cc12m_metadata_0.tsv"
        image_dir = Path(args.output_dir) / "images"
        download_cc12m_images(metadata_file, image_dir, args.max_images)
    
    if args.create_test_set:
        metadata_file = Path(args.output_dir) / "cc12m_metadata_0.tsv"
        output_file = Path(args.output_dir) / "cc12m_test_1000.tsv"
        create_small_test_dataset(metadata_file, output_file, args.test_samples)
