#!/usr/bin/env python3
"""Download public datasets for ViralFlip training.

This script provides best-effort download for public datasets.
Many datasets require registration or have gated access - check each
URL for access requirements.

Usage:
    python scripts/download_public_datasets.py --output data/public
    python scripts/download_public_datasets.py --dataset coughvid --output data/public
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile

import yaml

from viralflip.utils.io import ensure_dir
from viralflip.utils.logging import setup_logging, get_logger


logger = get_logger(__name__)


def load_registry() -> dict:
    """Load dataset registry."""
    registry_path = Path(__file__).parent.parent / "data" / "dataset_registry.yaml"
    
    with open(registry_path) as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a file from URL.
    
    Args:
        url: URL to download.
        output_path: Path to save file.
        timeout: Download timeout in seconds.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info(f"Downloading from {url}...")
        
        # Create request with user agent
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ViralFlip Dataset Downloader"}
        )
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        
        logger.info(f"Saved to {output_path}")
        return True
        
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP Error {e.code}: {e.reason}")
        if e.code == 403:
            logger.warning("Access forbidden - dataset may require registration")
        return False
    except urllib.error.URLError as e:
        logger.warning(f"URL Error: {e.reason}")
        return False
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip file.
    
    Args:
        zip_path: Path to zip file.
        output_dir: Directory to extract to.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        logger.info(f"Extracted to {output_dir}")
        return True
    except Exception as e:
        logger.warning(f"Extraction failed: {e}")
        return False


def download_zenodo_record(record_id: str, output_dir: Path) -> bool:
    """Download a Zenodo record.
    
    Args:
        record_id: Zenodo record ID.
        output_dir: Output directory.
        
    Returns:
        True if successful, False otherwise.
    """
    # Zenodo API to get record metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        import json
        
        request = urllib.request.Request(
            api_url,
            headers={"User-Agent": "ViralFlip Dataset Downloader"}
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.loads(response.read().decode())
        
        # Download each file
        files = metadata.get("files", [])
        
        if not files:
            logger.warning("No files found in Zenodo record")
            return False
        
        for file_info in files:
            file_url = file_info.get("links", {}).get("self")
            filename = file_info.get("key", "unknown")
            
            if file_url:
                output_path = output_dir / filename
                if not download_file(file_url, output_path):
                    logger.warning(f"Failed to download {filename}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Zenodo download failed: {e}")
        return False


def download_dataset(
    dataset_key: str,
    category: str,
    dataset_info: dict,
    output_dir: Path,
) -> bool:
    """Download a single dataset.
    
    Args:
        dataset_key: Dataset key in registry.
        category: Dataset category.
        dataset_info: Dataset metadata dict.
        output_dir: Base output directory.
        
    Returns:
        True if successful, False otherwise.
    """
    dataset_dir = ensure_dir(output_dir / category / dataset_key)
    
    url = dataset_info.get("url", "")
    name = dataset_info.get("name", dataset_key)
    access = dataset_info.get("access", "unknown")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {name}")
    logger.info(f"Access: {access}")
    logger.info(f"URL: {url}")
    logger.info(f"{'='*60}")
    
    if access in ["registration_required", "request_required"]:
        logger.warning(f"This dataset requires {access}. Please visit {url} manually.")
        
        # Save info file
        info_path = dataset_dir / "ACCESS_REQUIRED.txt"
        with open(info_path, "w") as f:
            f.write(f"Dataset: {name}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Access: {access}\n")
            f.write(f"\nPlease visit the URL above to request access.\n")
        
        return False
    
    # Try Zenodo downloads
    if "zenodo.org/records/" in url:
        record_id = url.split("/records/")[-1].split("/")[0]
        return download_zenodo_record(record_id, dataset_dir)
    
    # Try direct download for .zip files
    if url.endswith(".zip"):
        zip_path = dataset_dir / "download.zip"
        if download_file(url, zip_path):
            return extract_zip(zip_path, dataset_dir)
        return False
    
    # For GitHub repos, clone would be better but we just save info
    if "github.com" in url:
        info_path = dataset_dir / "GITHUB_REPO.txt"
        with open(info_path, "w") as f:
            f.write(f"Dataset: {name}\n")
            f.write(f"GitHub URL: {url}\n")
            f.write(f"\nTo download, run:\n")
            f.write(f"  git clone {url} {dataset_dir}/repo\n")
        logger.info(f"GitHub repo info saved to {info_path}")
        return True
    
    # For other URLs, save info file
    info_path = dataset_dir / "DOWNLOAD_INFO.txt"
    with open(info_path, "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Access: {access}\n")
        f.write(f"\nPlease visit the URL to download manually.\n")
    
    logger.info(f"Download info saved to {info_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download public datasets")
    parser.add_argument("--output", "-o", type=str, default="data/public",
                       help="Output directory")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                       help="Specific dataset to download (e.g., 'coughvid')")
    parser.add_argument("--category", "-c", type=str, default=None,
                       help="Dataset category (e.g., 'cough_voice', 'rppg')")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load registry
    registry = load_registry()
    
    # List mode
    if args.list:
        print("\nAvailable datasets:\n")
        for category, datasets in registry.items():
            print(f"{category}:")
            for key, info in datasets.items():
                name = info.get("name", key)
                access = info.get("access", "unknown")
                print(f"  - {key}: {name} (access: {access})")
            print()
        return
    
    output_dir = ensure_dir(Path(args.output))
    
    # Download specific dataset
    if args.dataset:
        found = False
        for category, datasets in registry.items():
            if args.dataset in datasets:
                download_dataset(
                    args.dataset, category, datasets[args.dataset], output_dir
                )
                found = True
                break
        
        if not found:
            logger.error(f"Dataset '{args.dataset}' not found in registry")
            sys.exit(1)
        return
    
    # Download by category
    if args.category:
        if args.category not in registry:
            logger.error(f"Category '{args.category}' not found")
            sys.exit(1)
        
        for key, info in registry[args.category].items():
            download_dataset(key, args.category, info, output_dir)
        return
    
    # Download all datasets
    logger.info("Downloading all datasets...")
    
    for category, datasets in registry.items():
        for key, info in datasets.items():
            download_dataset(key, category, info, output_dir)
    
    logger.info("\nDownload process complete!")
    logger.info("Note: Some datasets require manual download due to access restrictions.")


if __name__ == "__main__":
    main()

