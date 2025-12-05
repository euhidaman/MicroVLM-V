#!/usr/bin/env python3
"""
CC12M Resume Download Manager

Downloads additional CC12M images with intelligent resume capability.
Avoids redundant work by tracking:
- Already downloaded images
- Previously failed URLs
- Download progress checkpoints

Usage:
    python scripts/download_resume.py --target 7000000 --workers 64
    python scripts/download_resume.py --target all --workers 32
    python scripts/download_resume.py --status  # Check current progress
"""

import os
import sys
import json
import hashlib
import argparse
import requests
import shutil
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import pandas as pd


class DownloadState:
    """Manages download state and progress tracking"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.state_file = self.data_dir / "download_state.json"
        self.failed_file = self.data_dir / "failed_downloads.json"
        self.url_mapping_file = self.data_dir / "url_to_idx_mapping.json"
        self.checkpoint_file = self.data_dir / "download_checkpoint.json"
        
        # Thread-safe counters
        self._lock = Lock()
        self._success_count = 0
        self._fail_count = 0
        self._skip_count = 0
        
        # Load or initialize state
        self.state = self._load_state()
        self.failed_urls = self._load_failed_urls()
        self.url_mapping = self._load_url_mapping()
    
    def _load_state(self) -> dict:
        """Load download state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "total_attempted": 0,
            "total_successful": 0,
            "total_failed": 0,
            "last_processed_idx": -1,
            "last_update": None
        }
    
    def _load_failed_urls(self) -> dict:
        """Load failed URLs with their error info"""
        if self.failed_file.exists():
            with open(self.failed_file, 'r') as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    # Convert list format to dict format
                    return {item.get('url', str(item.get('idx', ''))): item for item in data}
                return data
        return {}
    
    def _load_url_mapping(self) -> dict:
        """Load URL to index mapping"""
        if self.url_mapping_file.exists():
            with open(self.url_mapping_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_state(self):
        """Save current state to file"""
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def save_failed_urls(self):
        """Save failed URLs to file"""
        with open(self.failed_file, 'w') as f:
            json.dump(self.failed_urls, f, indent=2)
    
    def save_url_mapping(self):
        """Save URL mapping to file"""
        with open(self.url_mapping_file, 'w') as f:
            json.dump(self.url_mapping, f, indent=2)
    
    def save_checkpoint(self, last_idx: int, stats: dict):
        """Save checkpoint for resume"""
        checkpoint = {
            "last_processed_idx": last_idx,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> dict:
        """Load checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def count_existing_images(self) -> int:
        """Count successfully downloaded images"""
        if not self.images_dir.exists():
            return 0
        return len(list(self.images_dir.glob("*.jpg")))
    
    def get_existing_indices(self) -> set:
        """Get set of already downloaded image indices"""
        if not self.images_dir.exists():
            return set()
        indices = set()
        for f in self.images_dir.glob("*.jpg"):
            try:
                idx = int(f.stem)
                indices.add(idx)
            except ValueError:
                continue
        return indices
    
    def get_failed_indices(self) -> set:
        """Get set of indices that previously failed"""
        indices = set()
        for key, value in self.failed_urls.items():
            if isinstance(value, dict) and 'idx' in value:
                indices.add(value['idx'])
            elif key.isdigit():
                indices.add(int(key))
        return indices
    
    def add_success(self, idx: int, url: str):
        """Record successful download"""
        with self._lock:
            self._success_count += 1
            self.url_mapping[url] = idx
    
    def add_failure(self, idx: int, url: str, error: str, retry_count: int = 0):
        """Record failed download"""
        with self._lock:
            self._fail_count += 1
            self.failed_urls[url] = {
                "idx": idx,
                "url": url,
                "error": error,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat()
            }
    
    def add_skip(self):
        """Record skipped URL"""
        with self._lock:
            self._skip_count += 1
    
    def get_session_stats(self) -> dict:
        """Get statistics for current session"""
        with self._lock:
            return {
                "success": self._success_count,
                "failed": self._fail_count,
                "skipped": self._skip_count
            }


def verify_image(image_path: Path) -> bool:
    """Verify image is valid and not corrupt"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to check it can be loaded
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False


def download_single_image(args) -> tuple:
    """
    Download a single image with retry logic.
    
    Args:
        args: (idx, url, save_dir, timeout, max_retries)
    
    Returns:
        (success: bool, idx: int, url: str, error: str or None)
    """
    idx, url, save_dir, timeout, max_retries = args
    image_path = save_dir / f"{idx:08d}.jpg"
    
    # Skip if already exists and valid
    if image_path.exists():
        if verify_image(image_path):
            return (True, idx, url, None, "exists")
        else:
            # Remove corrupt image
            image_path.unlink()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                last_error = f"Invalid content-type: {content_type}"
                break
            
            # Load and verify image
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check minimum size
            if img.size[0] < 10 or img.size[1] < 10:
                last_error = f"Image too small: {img.size}"
                break
            
            # Save
            img.save(image_path, 'JPEG', quality=95)
            
            # Verify saved image
            if verify_image(image_path):
                return (True, idx, url, None, "downloaded")
            else:
                image_path.unlink()
                last_error = "Saved image failed verification"
                continue
                
        except requests.exceptions.Timeout:
            last_error = "Connection timeout"
            time.sleep(0.5 * (attempt + 1))
            continue
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code in [403, 404, 410]:
                # Permanent failures - don't retry
                last_error = f"HTTP {status_code}"
                break
            last_error = f"HTTP error: {e}"
            time.sleep(0.5 * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {str(e)[:100]}"
            time.sleep(0.5 * (attempt + 1))
            continue
        except Exception as e:
            last_error = f"Error: {str(e)[:100]}"
            break
    
    return (False, idx, url, last_error, "failed")


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load CC12M metadata TSV"""
    # Try different possible filenames
    possible_files = [
        data_dir / "cc12m.tsv",
        data_dir / "cc12m_metadata_0.tsv",
        data_dir / "metadata.tsv"
    ]
    
    tsv_file = None
    for f in possible_files:
        if f.exists():
            tsv_file = f
            break
    
    if tsv_file is None:
        raise FileNotFoundError(
            f"No metadata TSV found. Tried: {[str(f) for f in possible_files]}\n"
            "Please download cc12m.tsv first."
        )
    
    print(f"Loading metadata from: {tsv_file}")
    
    # Read first line to detect format
    with open(tsv_file, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        parts = first_line.split('\t')
    
    # Detect format
    if len(parts) >= 2 and parts[0].startswith('http'):
        df = pd.read_csv(tsv_file, sep='\t', names=['url', 'caption'], 
                        encoding='utf-8', on_bad_lines='skip')
    else:
        df = pd.read_csv(tsv_file, sep='\t', names=['caption', 'url'],
                        encoding='utf-8', on_bad_lines='skip')
    
    # Remove any rows with missing URLs
    df = df.dropna(subset=['url'])
    df = df[df['url'].str.startswith('http', na=False)]
    
    print(f"Loaded {len(df):,} URLs from metadata")
    return df


def print_status(data_dir: Path):
    """Print current download status"""
    state = DownloadState(data_dir)
    
    existing = state.count_existing_images()
    failed = len(state.failed_urls)
    
    print("\n" + "=" * 60)
    print("CC12M DOWNLOAD STATUS")
    print("=" * 60)
    print(f"📁 Data directory: {data_dir}")
    print(f"✅ Successfully downloaded: {existing:,}")
    print(f"❌ Failed URLs: {failed:,}")
    print(f"📊 Total processed: {existing + failed:,}")
    
    # Try to get total from metadata
    try:
        df = load_metadata(data_dir)
        total = len(df)
        remaining = total - existing - failed
        print(f"📋 Total in dataset: {total:,}")
        print(f"⏳ Remaining to process: {remaining:,}")
        print(f"📈 Progress: {100 * (existing + failed) / total:.2f}%")
    except FileNotFoundError:
        print("⚠️  Metadata file not found - cannot calculate remaining")
    
    # Show last update
    if state.state.get("last_update"):
        print(f"🕐 Last update: {state.state['last_update']}")
    
    # Disk space
    if state.images_dir.exists():
        total_size = sum(f.stat().st_size for f in state.images_dir.glob("*.jpg"))
        print(f"💾 Disk usage: {total_size / (1024**3):.2f} GB")
    
    print("=" * 60 + "\n")


def download_manager(
    data_dir: Path,
    target: int | str,
    num_workers: int = 32,
    timeout: int = 15,
    max_retries: int = 3,
    checkpoint_interval: int = 10000,
    skip_validation: bool = False
):
    """
    Main download manager with resume capability.
    
    Args:
        data_dir: Directory containing metadata and images
        target: Target number of images ("all" or integer)
        num_workers: Number of concurrent download workers
        timeout: Request timeout in seconds
        max_retries: Maximum retries per URL
        checkpoint_interval: Save checkpoint every N images
        skip_validation: Skip re-validation of existing images
    """
    data_dir = Path(data_dir)
    state = DownloadState(data_dir)
    
    # Ensure directories exist
    state.images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = load_metadata(data_dir)
    total_in_dataset = len(df)
    
    # Determine target
    if target == "all":
        target_count = total_in_dataset
    else:
        target_count = min(int(target), total_in_dataset)
    
    # Get current state
    existing_indices = state.get_existing_indices()
    failed_indices = state.get_failed_indices()
    
    existing_count = len(existing_indices)
    failed_count = len(failed_indices)
    
    print("\n" + "=" * 60)
    print("CC12M RESUME DOWNLOAD MANAGER")
    print("=" * 60)
    print(f"📋 Total in dataset: {total_in_dataset:,}")
    print(f"🎯 Target: {target_count:,}")
    print(f"✅ Already downloaded: {existing_count:,}")
    print(f"❌ Previously failed: {failed_count:,}")
    
    # Calculate what we need
    needed = target_count - existing_count
    if needed <= 0:
        print(f"\n✨ Target already reached! ({existing_count:,} >= {target_count:,})")
        return
    
    print(f"⏳ Need to download: {needed:,} more images")
    print(f"👷 Workers: {num_workers}")
    print("=" * 60)
    
    # Build list of URLs to process
    # Skip: already downloaded OR already failed
    skip_indices = existing_indices | failed_indices
    
    urls_to_process = []
    for idx, row in df.iterrows():
        if idx in skip_indices:
            continue
        urls_to_process.append((idx, row['url']))
        if len(urls_to_process) >= needed:
            break
    
    if not urls_to_process:
        print("\n⚠️  No new URLs to process!")
        print("   All remaining URLs have either succeeded or failed previously.")
        print("   To retry failed URLs, delete failed_downloads.json and run again.")
        return
    
    print(f"\n🚀 Starting download of {len(urls_to_process):,} URLs...")
    
    # Prepare download arguments
    download_args = [
        (idx, url, state.images_dir, timeout, max_retries)
        for idx, url in urls_to_process
    ]
    
    # Progress tracking
    start_time = time.time()
    processed = 0
    session_success = 0
    session_failed = 0
    
    # Download with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_single_image, args): args[0]
            for args in download_args
        }
        
        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                success, idx, url, error, status = future.result()
                
                if success:
                    state.add_success(idx, url)
                    session_success += 1
                else:
                    state.add_failure(idx, url, error or "Unknown error")
                    session_failed += 1
                
                processed += 1
                
                # Update progress bar
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'success': session_success,
                    'failed': session_failed,
                    'rate': f'{rate:.1f}/s'
                })
                pbar.update(1)
                
                # Checkpoint
                if processed % checkpoint_interval == 0:
                    state.save_state()
                    state.save_failed_urls()
                    state.save_url_mapping()
                    state.save_checkpoint(idx, {
                        "session_success": session_success,
                        "session_failed": session_failed,
                        "processed": processed
                    })
    
    # Final save
    state.state["total_successful"] = existing_count + session_success
    state.state["total_failed"] = len(state.failed_urls)
    state.save_state()
    state.save_failed_urls()
    state.save_url_mapping()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"⏱️  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"📊 This session:")
    print(f"   ✅ Downloaded: {session_success:,}")
    print(f"   ❌ Failed: {session_failed:,}")
    print(f"   📈 Rate: {processed/elapsed:.1f} images/sec")
    print(f"\n📊 Total:")
    print(f"   ✅ Total downloaded: {existing_count + session_success:,}")
    print(f"   ❌ Total failed: {len(state.failed_urls):,}")
    print("=" * 60)
    
    # Update metadata.json if needed
    update_metadata(data_dir, state)


def update_metadata(data_dir: Path, state: DownloadState):
    """Update metadata.json with current images"""
    metadata_file = data_dir / "metadata.json"
    images_dir = data_dir / "images"
    
    print("\n📝 Updating metadata.json...")
    
    # Load TSV for captions
    df = load_metadata(data_dir)
    
    # Build samples list
    samples = []
    for img_file in sorted(images_dir.glob("*.jpg")):
        try:
            idx = int(img_file.stem)
            if idx < len(df):
                caption = df.iloc[idx]['caption']
                samples.append({
                    'idx': idx,
                    'image_path': str(img_file),
                    'caption': caption
                })
        except (ValueError, IndexError):
            continue
    
    metadata = {
        'samples': samples,
        'total_count': len(samples),
        'last_updated': datetime.now().isoformat()
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    print(f"✅ Updated metadata.json with {len(samples):,} samples")


def validate_images(data_dir: Path, fix: bool = False):
    """Validate all downloaded images and optionally remove corrupt ones"""
    images_dir = Path(data_dir) / "images"
    
    if not images_dir.exists():
        print("No images directory found")
        return
    
    print("Validating images...")
    
    image_files = list(images_dir.glob("*.jpg"))
    corrupt = []
    
    with tqdm(total=len(image_files), desc="Validating") as pbar:
        for img_path in image_files:
            if not verify_image(img_path):
                corrupt.append(img_path)
            pbar.update(1)
    
    print(f"\nFound {len(corrupt)} corrupt images")
    
    if corrupt and fix:
        print("Removing corrupt images...")
        for img_path in corrupt:
            img_path.unlink()
        print(f"Removed {len(corrupt)} corrupt images")
    elif corrupt:
        print("Run with --fix to remove corrupt images")
        for p in corrupt[:10]:
            print(f"  {p}")
        if len(corrupt) > 10:
            print(f"  ... and {len(corrupt) - 10} more")


def retry_failed(data_dir: Path, num_workers: int = 32, max_urls: int = None):
    """Retry previously failed URLs"""
    state = DownloadState(data_dir)
    
    if not state.failed_urls:
        print("No failed URLs to retry")
        return
    
    print(f"Found {len(state.failed_urls):,} failed URLs")
    
    # Get URLs to retry
    urls_to_retry = []
    for url, info in state.failed_urls.items():
        if isinstance(info, dict):
            idx = info.get('idx', 0)
            retry_count = info.get('retry_count', 0)
            # Skip URLs that have been retried too many times
            if retry_count >= 3:
                continue
            urls_to_retry.append((idx, url, retry_count))
        else:
            urls_to_retry.append((0, url, 0))
    
    if max_urls:
        urls_to_retry = urls_to_retry[:max_urls]
    
    if not urls_to_retry:
        print("No URLs eligible for retry")
        return
    
    print(f"Retrying {len(urls_to_retry)} URLs...")
    
    # Clear these from failed_urls before retry
    for idx, url, _ in urls_to_retry:
        if url in state.failed_urls:
            del state.failed_urls[url]
    
    # Retry downloads
    download_args = [
        (idx, url, state.images_dir, 20, 2)  # Longer timeout for retries
        for idx, url, _ in urls_to_retry
    ]
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_image, args): args for args in download_args}
        
        with tqdm(total=len(futures), desc="Retrying") as pbar:
            for future in as_completed(futures):
                success, idx, url, error, status = future.result()
                
                if success:
                    state.add_success(idx, url)
                    success_count += 1
                else:
                    # Increment retry count
                    orig_info = next((info for i, u, info in urls_to_retry if u == url), 0)
                    retry_count = orig_info if isinstance(orig_info, int) else 0
                    state.add_failure(idx, url, error or "Unknown", retry_count + 1)
                    fail_count += 1
                
                pbar.update(1)
    
    state.save_failed_urls()
    state.save_url_mapping()
    
    print(f"\nRetry complete: {success_count} recovered, {fail_count} still failing")


def main():
    parser = argparse.ArgumentParser(
        description="CC12M Resume Download Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current status
  python scripts/download_resume.py --status
  
  # Download up to 7 million images
  python scripts/download_resume.py --target 7000000 --workers 64
  
  # Download all images
  python scripts/download_resume.py --target all --workers 32
  
  # Validate existing images
  python scripts/download_resume.py --validate
  
  # Retry failed downloads
  python scripts/download_resume.py --retry --max-retry 10000
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='./data/cc12m',
                       help='Data directory (default: ./data/cc12m)')
    parser.add_argument('--target', type=str, default=None,
                       help='Target number of images (integer or "all")')
    parser.add_argument('--workers', type=int, default=32,
                       help='Number of concurrent download workers (default: 32)')
    parser.add_argument('--timeout', type=int, default=15,
                       help='Request timeout in seconds (default: 15)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retries per URL (default: 3)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                       help='Save checkpoint every N images (default: 10000)')
    
    # Actions
    parser.add_argument('--status', action='store_true',
                       help='Show download status and exit')
    parser.add_argument('--validate', action='store_true',
                       help='Validate all downloaded images')
    parser.add_argument('--fix', action='store_true',
                       help='Remove corrupt images when validating')
    parser.add_argument('--retry', action='store_true',
                       help='Retry previously failed URLs')
    parser.add_argument('--max-retry', type=int, default=None,
                       help='Maximum URLs to retry')
    parser.add_argument('--update-metadata', action='store_true',
                       help='Update metadata.json with current images')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    # Handle different actions
    if args.status:
        print_status(data_dir)
        return
    
    if args.validate:
        validate_images(data_dir, fix=args.fix)
        return
    
    if args.retry:
        retry_failed(data_dir, args.workers, args.max_retry)
        return
    
    if args.update_metadata:
        state = DownloadState(data_dir)
        update_metadata(data_dir, state)
        return
    
    # Main download
    if args.target is None:
        parser.print_help()
        print("\n❌ Error: --target is required for downloading")
        print("   Use --target 7000000 or --target all")
        return
    
    try:
        target = args.target if args.target == "all" else int(args.target)
    except ValueError:
        print(f"❌ Invalid target: {args.target}")
        print("   Use an integer or 'all'")
        return
    
    download_manager(
        data_dir=data_dir,
        target=target,
        num_workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()
