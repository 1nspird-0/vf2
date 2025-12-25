#!/usr/bin/env python3
"""Gather real-world public datasets for ViralFlip training.

This script downloads essential publicly available datasets for training.
Focuses on datasets that are:
1. Actually downloadable (open access, direct links)
2. Useful for ViralFlip components

Usage:
    python scripts/gather_real_data.py --output data/real --all
    python scripts/gather_real_data.py --output data/real --datasets esc50,uci_har
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
import ssl

# Disable SSL verification for problematic sites
ssl._create_default_https_context = ssl._create_unverified_context

from viralflip.utils.io import ensure_dir
from viralflip.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# ============================================================================
# Essential Datasets with Direct Download Links
# ============================================================================

DATASETS = {
    # ESC-50 - Environmental Sound Classification (includes coughing)
    "esc50": {
        "name": "ESC-50",
        "category": "audio",
        "url": "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        "description": "Environmental sounds including coughing class for audio classification",
        "size_mb": 600,
        "use_for": ["cough_detection", "audio_quality"],
    },
    
    # UCI HAR - Human Activity Recognition
    "uci_har": {
        "name": "UCI HAR",
        "category": "imu",
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "description": "Smartphone accelerometer/gyroscope from 30 subjects",
        "size_mb": 60,
        "use_for": ["imu_features", "activity_detection"],
    },
    
    # WISDM - Activity and Biometrics
    "wisdm": {
        "name": "WISDM",
        "category": "imu",
        "url": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
        "description": "Smartphone/smartwatch IMU from 51 subjects",
        "size_mb": 400,
        "use_for": ["imu_features", "gait_analysis"],
    },
    
    # Beiwe Sample - Phone Sensing
    "beiwe": {
        "name": "Beiwe Sample",
        "category": "phone_sensing",
        "url": "https://zenodo.org/records/1188879/files/Sample%20Beiwe%20Data.zip?download=1",
        "description": "GPS/WiFi/Bluetooth/accelerometer sample data",
        "size_mb": 50,
        "use_for": ["mobility_features", "behavior_patterns"],
    },
}


def download_with_progress(url: str, output_path: Path, timeout: int = 120) -> bool:
    """Download file with progress bar."""
    try:
        # Create request with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        print(f"  Connecting to {url[:60]}...")
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_len - filled)
                        print(f"\r  [{bar}] {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end='', flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded: {mb:.1f} MB", end='', flush=True)
            
            print()  # Newline
        
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n  HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  Download error: {e}")
        return False


def download_with_curl(url: str, output_path: Path) -> bool:
    """Use curl as fallback for problematic downloads."""
    try:
        print(f"  Using curl to download...")
        result = subprocess.run(
            ['curl', '-L', '-o', str(output_path), '-#', url],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"  Curl failed: {e}")
        return False


def download_with_powershell(url: str, output_path: Path) -> bool:
    """Use PowerShell as fallback on Windows."""
    try:
        print(f"  Using PowerShell to download...")
        cmd = f'Invoke-WebRequest -Uri "{url}" -OutFile "{output_path}" -UseBasicParsing'
        result = subprocess.run(
            ['powershell', '-Command', cmd],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"  PowerShell failed: {e}")
        return False


def download_file(url: str, output_path: Path) -> bool:
    """Download file with multiple fallback methods."""
    # Try Python urllib first
    if download_with_progress(url, output_path):
        return True
    
    # Try curl
    if download_with_curl(url, output_path):
        return True
    
    # Try PowerShell on Windows
    if sys.platform == 'win32':
        if download_with_powershell(url, output_path):
            return True
    
    return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract ZIP file."""
    try:
        print(f"  Extracting to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print(f"  Extraction complete!")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def process_esc50(dataset_dir: Path) -> dict:
    """Process ESC-50 dataset to extract cough samples."""
    print("  Processing ESC-50...")
    
    # Find the extracted directory
    esc_dirs = list(dataset_dir.glob("ESC-50*"))
    if not esc_dirs:
        return {"status": "error", "reason": "ESC-50 directory not found"}
    
    esc_dir = esc_dirs[0]
    
    # Read metadata
    meta_files = list(esc_dir.glob("**/esc50.csv"))
    if not meta_files:
        meta_files = list(esc_dir.glob("**/meta/esc50.csv"))
    
    if not meta_files:
        return {"status": "error", "reason": "Metadata not found"}
    
    # Count cough samples
    cough_count = 0
    total_count = 0
    
    try:
        with open(meta_files[0], 'r') as f:
            header = f.readline()
            for line in f:
                total_count += 1
                if 'coughing' in line.lower():
                    cough_count += 1
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    
    return {
        "status": "success",
        "total_samples": total_count,
        "cough_samples": cough_count,
        "use_for": ["cough_detection", "audio_classification"],
    }


def process_uci_har(dataset_dir: Path) -> dict:
    """Process UCI HAR dataset."""
    print("  Processing UCI HAR...")
    
    # Find data directory
    har_dirs = list(dataset_dir.glob("**/UCI HAR Dataset"))
    if not har_dirs:
        har_dirs = list(dataset_dir.glob("**/train"))
        if har_dirs:
            har_dirs = [har_dirs[0].parent]
    
    if not har_dirs:
        return {"status": "error", "reason": "UCI HAR directory not found"}
    
    har_dir = har_dirs[0]
    
    # Check for key files
    train_exists = (har_dir / "train").exists() or list(dataset_dir.glob("**/train")).count
    test_exists = (har_dir / "test").exists() or list(dataset_dir.glob("**/test")).count
    
    # Count activity labels
    activity_file = har_dir / "activity_labels.txt"
    activities = {}
    if activity_file.exists():
        with open(activity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    activities[parts[0]] = parts[1]
    
    return {
        "status": "success",
        "has_train": bool(train_exists),
        "has_test": bool(test_exists),
        "activities": activities,
        "sampling_rate": 50,
        "use_for": ["imu_features", "activity_recognition", "step_detection"],
    }


def process_wisdm(dataset_dir: Path) -> dict:
    """Process WISDM dataset."""
    print("  Processing WISDM...")
    
    # Count data files
    accel_files = list(dataset_dir.glob("**/*accel*.txt")) + list(dataset_dir.glob("**/*accel*.csv"))
    gyro_files = list(dataset_dir.glob("**/*gyro*.txt")) + list(dataset_dir.glob("**/*gyro*.csv"))
    
    return {
        "status": "success",
        "accel_files": len(accel_files),
        "gyro_files": len(gyro_files),
        "use_for": ["imu_features", "gait_analysis", "activity_recognition"],
    }


def process_beiwe(dataset_dir: Path) -> dict:
    """Process Beiwe sample dataset."""
    print("  Processing Beiwe...")
    
    # Find data streams
    streams = {}
    for stream in ['accelerometer', 'gps', 'gyro', 'wifi', 'bluetooth']:
        files = list(dataset_dir.glob(f"**/{stream}/*.csv"))
        if files:
            streams[stream] = len(files)
    
    return {
        "status": "success",
        "streams": streams,
        "use_for": ["mobility_features", "phone_sensing", "behavior_confounds"],
    }


def gather_dataset(key: str, info: dict, output_dir: Path) -> dict:
    """Download and process a single dataset."""
    dataset_dir = ensure_dir(output_dir / key)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {info['name']}")
    print(f"Size: ~{info.get('size_mb', 'unknown')} MB")
    print(f"Use for: {', '.join(info.get('use_for', []))}")
    print(f"{'='*60}")
    
    result = {"dataset": key, "name": info["name"]}
    
    # Check if already downloaded
    zip_path = dataset_dir / f"{key}.zip"
    existing_files = list(dataset_dir.glob("*"))
    
    if len(existing_files) > 1 or (existing_files and existing_files[0].is_dir()):
        print(f"  Already downloaded, skipping download...")
        result["download"] = "cached"
    else:
        # Download
        print(f"  Downloading {info['name']}...")
        
        if download_file(info["url"], zip_path):
            result["download"] = "success"
            
            # Extract
            if zip_path.exists():
                if extract_zip(zip_path, dataset_dir):
                    # Optionally remove zip to save space
                    # zip_path.unlink()
                    pass
        else:
            result["download"] = "failed"
            return result
    
    # Process based on dataset type
    process_funcs = {
        "esc50": process_esc50,
        "uci_har": process_uci_har,
        "wisdm": process_wisdm,
        "beiwe": process_beiwe,
    }
    
    if key in process_funcs:
        result["processed"] = process_funcs[key](dataset_dir)
    
    return result


def create_training_manifest(output_dir: Path, results: list) -> None:
    """Create manifest of downloaded data."""
    successful = [r for r in results if r.get("download") in ["success", "cached"]]
    
    manifest = {
        "viralflip_data": True,
        "datasets": results,
        "summary": {
            "total_attempted": len(results),
            "successful": len(successful),
        },
        "ready_for": {
            "cough_detection": any("esc50" in r.get("dataset", "") for r in successful),
            "imu_features": any(r.get("dataset") in ["uci_har", "wisdm"] for r in successful),
            "phone_sensing": any("beiwe" in r.get("dataset", "") for r in successful),
        },
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to {output_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Download real datasets for ViralFlip")
    parser.add_argument("--output", "-o", type=str, default="data/real",
                       help="Output directory")
    parser.add_argument("--datasets", "-d", type=str, default=None,
                       help="Comma-separated dataset keys (e.g., esc50,uci_har)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # List mode
    if args.list:
        print("\nAvailable Datasets:")
        print("-" * 60)
        for key, info in DATASETS.items():
            size = f"~{info.get('size_mb', '?')} MB"
            print(f"  {key:15s} {info['name']:25s} {size}")
        print("-" * 60)
        print(f"\nUsage: python {sys.argv[0]} --output data/real --all")
        return
    
    # Determine datasets to download
    if args.datasets:
        keys = [k.strip() for k in args.datasets.split(",")]
    elif args.all:
        keys = list(DATASETS.keys())
    else:
        # Default: download essential datasets
        keys = ["esc50", "uci_har"]
        print("No datasets specified. Downloading essential datasets: esc50, uci_har")
        print("Use --all for all datasets or --datasets for specific ones.")
    
    # Validate keys
    valid_keys = [k for k in keys if k in DATASETS]
    if not valid_keys:
        print(f"No valid datasets found. Available: {list(DATASETS.keys())}")
        return
    
    output_dir = ensure_dir(Path(args.output))
    
    print(f"\nWill download {len(valid_keys)} datasets to {output_dir}")
    print(f"Datasets: {valid_keys}")
    
    # Download each dataset
    results = []
    for key in valid_keys:
        result = gather_dataset(key, DATASETS[key], output_dir)
        results.append(result)
    
    # Create manifest
    create_training_manifest(output_dir, results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    for r in results:
        status = r.get("download", "unknown")
        icon = "[OK]" if status in ["success", "cached"] else "[FAIL]"
        print(f"  {icon} {r['name']}: {status}")
        if "processed" in r:
            proc = r["processed"]
            if proc.get("status") == "success":
                for k, v in proc.items():
                    if k not in ["status", "use_for"]:
                        print(f"       - {k}: {v}")
    
    print(f"\nData saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. python scripts/train.py --config configs/default.yaml --data synthetic")
    print("  2. Use downloaded data to validate feature extractors")


if __name__ == "__main__":
    main()
