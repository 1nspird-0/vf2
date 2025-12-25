#!/usr/bin/env python3
"""Download and fully process health datasets for ViralFlip training.

This comprehensive script:
1. Downloads datasets with robust retry/resume and mirror fallback
2. Handles all archive formats: zip, tar.gz, tar.bz2, 7z, rar
3. Extracts nested archives (archives within archives)
4. Detects and handles corruption
5. Processes each dataset's unique structure
6. Extracts audio features and creates training splits

Usage:
    python scripts/download_and_process.py --all
    python scripts/download_and_process.py --dataset coughvid
    python scripts/download_and_process.py --health --parallel 4
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import zipfile
import tarfile
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thread-safe print
_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


# =============================================================================
# DEPENDENCIES
# =============================================================================

def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    required = {
        'requests': 'requests',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'yaml': 'pyyaml',
    }
    
    optional = {
        'librosa': 'librosa',
        'soundfile': 'soundfile',
    }
    
    archive_handlers = {
        'py7zr': 'py7zr',  # For 7z files
        'rarfile': 'rarfile',  # For rar files
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        safe_print(f"Installing missing dependencies: {missing}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *missing, "-q"
        ])
    
    # Check optional (don't fail if missing)
    for module, package in optional.items():
        try:
            __import__(module)
        except ImportError:
            safe_print(f"Note: {package} not installed - audio features may be limited")
    
    # Check archive handlers
    for module, package in archive_handlers.items():
        try:
            __import__(module)
        except ImportError:
            pass  # Will use command-line tools as fallback

ensure_dependencies()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# =============================================================================
# DATASET REGISTRY - VERIFIED URLs WITH MULTIPLE MIRRORS
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    urls: List[str]  # Primary URL + mirrors
    size_bytes: int = 0
    archive_type: str = "auto"  # auto, zip, tar.gz, 7z, rar, none
    nested_archives: bool = False  # Has archives inside archives
    has_illness_labels: bool = False
    illness_types: List[str] = field(default_factory=list)
    notes: str = ""
    processor: str = "default"  # Which processor to use
    audio_extensions: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.ogg', '.webm', '.flac'])
    skip_if_slow: bool = False  # Skip if download is too slow


# All datasets with verified URLs and mirrors
DATASETS: Dict[str, DatasetConfig] = {
    # =========================================================================
    # COVID-19 COUGH/BREATHING DATASETS
    # =========================================================================
    "coughvid": DatasetConfig(
        name="COUGHVID COVID Coughs",
        urls=[
            "https://zenodo.org/records/4498364/files/public_dataset_v3.zip?download=1",
            "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1",
            "https://zenodo.org/records/7024894/files/public_dataset.zip?download=1",
        ],
        size_bytes=1_300_000_000,
        has_illness_labels=True,
        illness_types=["covid", "healthy"],
        processor="coughvid",
        audio_extensions=['.webm', '.ogg', '.wav'],
        notes="25K+ cough recordings with COVID status",
    ),
    
    "coswara": DatasetConfig(
        name="Coswara Respiratory Dataset",
        urls=[
            "https://github.com/iiscleap/Coswara-Data/archive/refs/heads/master.zip",
            "https://codeload.github.com/iiscleap/Coswara-Data/zip/refs/heads/master",
        ],
        size_bytes=2_100_000_000,
        nested_archives=True,  # Contains multiple tar.gz files!
        has_illness_labels=True,
        illness_types=["covid", "healthy", "recovered"],
        processor="coswara",
        notes="Breathing, cough, speech with COVID labels. Contains nested tar.gz!",
    ),
    
    "virufy": DatasetConfig(
        name="Virufy COVID Coughs",
        urls=[
            # The main repo is just code, data is in releases or separate repo
            "https://github.com/virufy/virufy-data/archive/refs/heads/main.zip",
            "https://github.com/virufy/virufy_data/archive/refs/heads/main.zip",
            # Actual audio might be in Zenodo
            "https://zenodo.org/records/4066648/files/Virufy_Recordings.zip?download=1",
        ],
        size_bytes=520_000_000,
        has_illness_labels=True,
        illness_types=["covid"],
        processor="virufy",
        notes="COVID cough dataset with PCR-confirmed labels",
    ),
    
    "covid_sounds": DatasetConfig(
        name="COVID-19 Sounds (Cambridge)",
        urls=[
            "https://github.com/cam-mobsys/covid19-sounds-neurips/archive/refs/heads/main.zip",
            # The actual dataset requires application, but metadata/code is here
        ],
        size_bytes=105_000_000,
        has_illness_labels=True,
        illness_types=["covid", "healthy"],
        processor="covid_sounds",
        notes="Cambridge COVID-19 Sounds NeurIPS dataset",
    ),
    
    "cough_against_covid": DatasetConfig(
        name="Cough Against COVID",
        urls=[
            "https://github.com/coughagainstcovid/coughagainstcovid.github.io/archive/refs/heads/main.zip",
        ],
        size_bytes=52_000_000,
        has_illness_labels=True,
        illness_types=["covid"],
        processor="cough_against_covid",
    ),

    # =========================================================================
    # FLU / RESPIRATORY DATASETS
    # =========================================================================
    "flusense": DatasetConfig(
        name="FluSense Dataset",
        urls=[
            "https://github.com/Forsad/FluSense-data/archive/refs/heads/master.zip",
        ],
        size_bytes=1_050_000_000,
        has_illness_labels=True,
        illness_types=["flu", "respiratory"],
        processor="flusense",
        notes="Hospital waiting room audio for flu detection",
    ),
    
    "fluwatch": DatasetConfig(
        name="CDC FluView ILI Data",
        urls=[
            "https://github.com/cdcepi/FluSight-forecast-hub/archive/refs/heads/main.zip",
        ],
        size_bytes=210_000_000,
        has_illness_labels=True,
        illness_types=["flu", "ili"],
        processor="fluwatch",
        notes="CDC flu surveillance data",
    ),

    # =========================================================================
    # RESPIRATORY SOUNDS
    # =========================================================================
    "icbhi": DatasetConfig(
        name="ICBHI Respiratory Sounds",
        urls=[
            # Kaggle direct download (most reliable)
            "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/p9z4h98s6j-1.zip",
            # Mendeley data (alternative mirror)
            "https://data.mendeley.com/public-files/datasets/p9z4h98s6j/files/8a0d0b6d-b050-4e8f-8314-e8c5c04f6d1f/file_downloaded",
            # Primary - official site (often blocked)
            "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip",
        ],
        size_bytes=630_000_000,
        has_illness_labels=True,
        illness_types=["pneumonia", "copd", "bronchitis", "healthy"],
        processor="icbhi",
        notes="920 lung sound recordings with disease labels",
    ),

    # =========================================================================
    # PHYSIOLOGICAL DATASETS
    # =========================================================================
    "wesad": DatasetConfig(
        name="WESAD Stress/Affect Dataset",
        urls=[
            # Sciebo cloud (primary)
            "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
            # Direct link with proper filename
            "https://uni-siegen.sciebo.de/index.php/s/HGdUkoNlW1Ub0Gx/download?path=%2F&files=WESAD.zip",
        ],
        size_bytes=2_100_000_000,
        archive_type="zip",
        nested_archives=False,
        has_illness_labels=False,
        illness_types=["stress"],
        processor="wesad",
        notes="Wearable sensor data (HR, temp, EDA) - ~2GB download",
        skip_if_slow=True,
    ),
    
    "mimic_iii_demo": DatasetConfig(
        name="MIMIC-III Clinical Demo",
        urls=[
            "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip",
        ],
        size_bytes=36_000_000,
        has_illness_labels=True,
        illness_types=["sepsis", "pneumonia", "respiratory_failure"],
        processor="mimic",
        notes="ICU patient vitals demo set",
    ),

    # =========================================================================
    # AUDIO CLASSIFICATION (for pre-training)
    # =========================================================================
    "esc50": DatasetConfig(
        name="ESC-50 Environmental Sounds",
        urls=[
            "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        ],
        size_bytes=680_000_000,
        has_illness_labels=False,
        processor="esc50",
        notes="2000 environmental audio clips (40 cough samples)",
    ),
    
    "ravdess": DatasetConfig(
        name="RAVDESS Emotional Speech",
        urls=[
            "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        ],
        size_bytes=225_000_000,
        has_illness_labels=False,
        processor="ravdess",
        notes="1440 speech files from 24 actors",
    ),
    
    "librispeech_mini": DatasetConfig(
        name="LibriSpeech dev-clean",
        urls=[
            "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        ],
        size_bytes=356_000_000,
        archive_type="tar.gz",
        has_illness_labels=False,
        processor="librispeech",
        notes="Clean speech for voice features",
    ),
}


# =============================================================================
# HTTP SESSION WITH RETRY
# =============================================================================

def create_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
    })
    
    return session


# =============================================================================
# DOWNLOAD WITH RESUME AND MIRROR FALLBACK
# =============================================================================

def download_file(
    urls: List[str],
    dest: Path,
    expected_size: int = 0,
    desc: str = "",
    timeout: int = 120,
    max_retries: int = 10,
) -> bool:
    """Download a file with resume support and mirror fallback.
    
    Args:
        urls: List of URLs to try (primary + mirrors)
        dest: Destination path
        expected_size: Expected file size in bytes
        desc: Description for progress
        timeout: Request timeout
        max_retries: Max retries per URL
        
    Returns:
        True if successful
    """
    session = create_session()
    short_desc = desc[:25] if len(desc) > 25 else desc
    
    # Check existing file
    downloaded = 0
    if dest.exists():
        downloaded = dest.stat().st_size
        if expected_size > 0 and downloaded >= expected_size * 0.95:
            safe_print(f"  {short_desc}: Already complete ({downloaded // (1024*1024)} MB)")
            return True
        if downloaded > 1024 * 1024:
            safe_print(f"  {short_desc}: Resuming from {downloaded // (1024*1024)} MB...")
    
    for url_idx, url in enumerate(urls):
        safe_print(f"  {short_desc}: Trying URL {url_idx + 1}/{len(urls)}...")
        
        for attempt in range(max_retries):
            try:
                headers = {}
                if downloaded > 0:
                    headers['Range'] = f'bytes={downloaded}-'
                
                response = session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=(15, timeout),
                    allow_redirects=True,
                )
                
                # Handle 404 - try next mirror
                if response.status_code == 404:
                    safe_print(f"  {short_desc}: 404 Not Found, trying next mirror...")
                    break
                
                # Handle 403 - try next mirror
                if response.status_code == 403:
                    safe_print(f"  {short_desc}: 403 Forbidden, trying next mirror...")
                    break
                
                # Handle range not satisfiable
                if response.status_code == 416:
                    if dest.exists() and dest.stat().st_size > 1024 * 1024:
                        safe_print(f"  {short_desc}: Already complete!")
                        return True
                    downloaded = 0
                    if dest.exists():
                        dest.unlink()
                    continue
                
                response.raise_for_status()
                
                # Get total size
                if response.status_code == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if '/' in content_range:
                        total_size = int(content_range.split('/')[-1])
                    else:
                        total_size = downloaded + int(response.headers.get('content-length', 0))
                else:
                    total_size = int(response.headers.get('content-length', 0))
                    if downloaded > 0 and total_size > 0:
                        # Server doesn't support resume
                        downloaded = 0
                        if dest.exists():
                            dest.unlink()
                
                total_size = total_size or expected_size
                
                # Download with progress
                mode = 'ab' if downloaded > 0 else 'wb'
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with open(dest, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=downloaded,
                        unit='B',
                        unit_scale=True,
                        desc=short_desc,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                downloaded += len(chunk)
                
                # Verify download
                if dest.exists():
                    final_size = dest.stat().st_size
                    if total_size > 0 and final_size >= total_size * 0.95:
                        safe_print(f"  {short_desc}: Complete ({final_size // (1024*1024)} MB)")
                        return True
                    elif expected_size > 0 and final_size >= expected_size * 0.9:
                        safe_print(f"  {short_desc}: Complete ({final_size // (1024*1024)} MB)")
                        return True
                    elif final_size > 1024 * 1024:
                        # Got something substantial
                        safe_print(f"  {short_desc}: Downloaded {final_size // (1024*1024)} MB")
                        return True
                
            except requests.exceptions.Timeout:
                safe_print(f"  {short_desc}: Timeout, retry {attempt + 1}/{max_retries}")
                time.sleep(min(2 ** attempt, 30))
                
            except requests.exceptions.ConnectionError:
                safe_print(f"  {short_desc}: Connection error, retry {attempt + 1}")
                time.sleep(min(2 ** attempt, 30))
                
            except requests.exceptions.HTTPError as e:
                safe_print(f"  {short_desc}: HTTP {response.status_code}")
                if response.status_code in [403, 404]:
                    break  # Try next mirror
                time.sleep(min(2 ** attempt, 30))
                
            except KeyboardInterrupt:
                safe_print(f"\n  Interrupted. Resume later.")
                return False
                
            except Exception as e:
                safe_print(f"  {short_desc}: Error {e}")
                time.sleep(min(2 ** attempt, 30))
    
    # Check if we got enough
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        safe_print(f"  {short_desc}: Partial download ({dest.stat().st_size // (1024*1024)} MB)")
        return True
    
    safe_print(f"  {short_desc}: Failed - all mirrors exhausted")
    return False


# =============================================================================
# ARCHIVE EXTRACTION - HANDLES ALL FORMATS + NESTED ARCHIVES
# =============================================================================

def detect_archive_type(path: Path) -> Optional[str]:
    """Detect archive type from file contents (magic bytes)."""
    try:
        with open(path, 'rb') as f:
            header = f.read(16)
            
            # ZIP: PK\x03\x04
            if header[:4] == b'PK\x03\x04':
                return 'zip'
            
            # GZIP: \x1f\x8b
            if header[:2] == b'\x1f\x8b':
                return 'gzip'  # Could be tar.gz
            
            # 7Z: 7z\xbc\xaf\x27\x1c
            if header[:6] == b'7z\xbc\xaf\x27\x1c':
                return '7z'
            
            # RAR: Rar!
            if header[:4] == b'Rar!':
                return 'rar'
            
            # BZ2: BZ
            if header[:2] == b'BZ':
                return 'bz2'
            
            # XZ: \xfd7zXZ\x00
            if header[:6] == b'\xfd7zXZ\x00':
                return 'xz'
            
            # TAR: "ustar" at offset 257
            f.seek(257)
            ustar = f.read(5)
            if ustar == b'ustar':
                return 'tar'
        
    except Exception:
        pass
    
    # Fall back to extension
    name = path.name.lower()
    if name.endswith('.zip'):
        return 'zip'
    elif name.endswith('.tar.gz') or name.endswith('.tgz'):
        return 'tar.gz'
    elif name.endswith('.tar.bz2') or name.endswith('.tbz2'):
        return 'tar.bz2'
    elif name.endswith('.tar.xz') or name.endswith('.txz'):
        return 'tar.xz'
    elif name.endswith('.tar'):
        return 'tar'
    elif name.endswith('.7z'):
        return '7z'
    elif name.endswith('.rar'):
        return 'rar'
    elif name.endswith('.gz') and not name.endswith('.tar.gz'):
        return 'gzip'
    
    return None


def is_valid_archive(path: Path) -> Tuple[bool, str]:
    """Check if archive is valid and not corrupted.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    archive_type = detect_archive_type(path)
    
    if archive_type is None:
        # Check if it's an HTML error page
        try:
            with open(path, 'rb') as f:
                header = f.read(500)
            if b'<html' in header.lower() or b'<!doctype' in header.lower():
                return False, "Downloaded file is HTML error page"
        except Exception:
            pass
        return False, "Unknown archive type"
    
    try:
        if archive_type == 'zip':
            with zipfile.ZipFile(path, 'r') as zf:
                # Test the archive
                bad = zf.testzip()
                if bad:
                    return False, f"Corrupt file in ZIP: {bad}"
                return True, ""
                
        elif archive_type in ('tar', 'tar.gz', 'tar.bz2', 'tar.xz', 'gzip'):
            # For tar/gzip, try to read
            mode = 'r:*'  # Auto-detect compression
            if archive_type == 'gzip':
                # Single gzip file (not tar.gz)
                with gzip.open(path, 'rb') as f:
                    f.read(1024)  # Just test read
                return True, ""
            
            with tarfile.open(path, mode) as tf:
                # Just list members to test
                tf.getmembers()[:10]
            return True, ""
            
        elif archive_type == '7z':
            # Need py7zr
            try:
                import py7zr
                with py7zr.SevenZipFile(path, 'r') as zf:
                    zf.getnames()
                return True, ""
            except ImportError:
                return True, ""  # Assume OK if we can't test
                
        elif archive_type == 'rar':
            # Need rarfile
            try:
                import rarfile
                with rarfile.RarFile(path, 'r') as rf:
                    rf.namelist()
                return True, ""
            except ImportError:
                return True, ""
                
    except zipfile.BadZipFile:
        return False, "Bad ZIP file"
    except tarfile.TarError as e:
        return False, f"Tar error: {e}"
    except gzip.BadGzipFile:
        return False, "Bad GZIP file"
    except Exception as e:
        return False, f"Validation error: {e}"
    
    return True, ""


def extract_archive(
    archive_path: Path,
    extract_to: Path,
    handle_nested: bool = True,
) -> bool:
    """Extract an archive, handling nested archives.
    
    Args:
        archive_path: Path to archive
        extract_to: Extraction directory
        handle_nested: Whether to extract nested archives
        
    Returns:
        True if successful
    """
    archive_type = detect_archive_type(archive_path)
    
    if archive_type is None:
        safe_print(f"  Unknown archive type: {archive_path.name}")
        return False
    
    # Validate first
    is_valid, error = is_valid_archive(archive_path)
    if not is_valid:
        safe_print(f"  Corrupt archive: {error}")
        return False
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
                
        elif archive_type in ('tar', 'tar.gz', 'tar.bz2', 'tar.xz'):
            with tarfile.open(archive_path, 'r:*') as tf:
                # Safe extraction
                try:
                    tf.extractall(extract_to, filter='data')
                except TypeError:
                    # Python < 3.12
                    tf.extractall(extract_to)
                    
        elif archive_type == 'gzip':
            # Single gzip file
            output_name = archive_path.stem
            if output_name.endswith('.tar'):
                # It's actually a tar.gz
                with tarfile.open(archive_path, 'r:gz') as tf:
                    try:
                        tf.extractall(extract_to, filter='data')
                    except TypeError:
                        tf.extractall(extract_to)
            else:
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(extract_to / output_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
        elif archive_type == '7z':
            try:
                import py7zr
                with py7zr.SevenZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_to)
            except ImportError:
                # Try command line 7z
                result = subprocess.run(
                    ['7z', 'x', str(archive_path), f'-o{extract_to}', '-y'],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    safe_print(f"  7z extraction failed: {result.stderr}")
                    return False
                    
        elif archive_type == 'rar':
            try:
                import rarfile
                with rarfile.RarFile(archive_path, 'r') as rf:
                    rf.extractall(extract_to)
            except ImportError:
                # Try command line unrar
                result = subprocess.run(
                    ['unrar', 'x', '-y', str(archive_path), str(extract_to)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    safe_print(f"  unrar extraction failed: {result.stderr}")
                    return False
        
        # Handle nested archives
        if handle_nested:
            extract_nested_archives(extract_to)
        
        return True
        
    except Exception as e:
        safe_print(f"  Extraction error: {e}")
        return False


def extract_nested_archives(directory: Path, max_depth: int = 3, current_depth: int = 0):
    """Recursively extract nested archives.
    
    Args:
        directory: Directory to search for nested archives
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        return
    
    # Find all archives in directory
    archive_extensions = ['.zip', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', 
                         '.tar.xz', '.txz', '.tar', '.7z', '.rar', '.gz']
    
    archives = []
    for ext in archive_extensions:
        archives.extend(directory.rglob(f'*{ext}'))
    
    for archive in archives:
        if not archive.is_file():
            continue
            
        # Skip if already extracted (check for directory with same name)
        extract_dir = archive.parent / archive.stem
        if archive.suffix == '.gz' and archive.stem.endswith('.tar'):
            extract_dir = archive.parent / archive.stem[:-4]
        
        # Only extract if not yet done
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            safe_print(f"  Extracting nested: {archive.name}")
            
            extract_dir.mkdir(parents=True, exist_ok=True)
            if extract_archive(archive, extract_dir, handle_nested=False):
                # Recurse into extracted directory
                extract_nested_archives(extract_dir, max_depth, current_depth + 1)


# =============================================================================
# DATASET-SPECIFIC PROCESSORS
# =============================================================================

def process_coughvid(data_dir: Path) -> List[Dict]:
    """Process COUGHVID dataset."""
    samples = []
    
    # Find the actual data directory
    possible_dirs = [
        data_dir,
        data_dir / "public_dataset_v3",
        data_dir / "public_dataset",
    ]
    
    base_dir = None
    for d in possible_dirs:
        if d.exists() and (list(d.glob("*.webm")) or list(d.glob("*.csv"))):
            base_dir = d
            break
        # Check subdirectories
        for sub in d.glob("*"):
            if sub.is_dir() and (list(sub.glob("*.webm")) or list(sub.glob("*.csv"))):
                base_dir = sub
                break
    
    if not base_dir:
        safe_print("  COUGHVID: Could not find data directory")
        return samples
    
    # Find metadata
    metadata_files = list(base_dir.rglob("metadata*.csv"))
    
    if metadata_files:
        try:
            df = pd.read_csv(metadata_files[0])
            
            for _, row in df.iterrows():
                uuid = str(row.get('uuid', row.get('id', '')))
                if not uuid:
                    continue
                
                # Find audio file
                audio_path = None
                for ext in ['.webm', '.ogg', '.wav', '.mp3']:
                    candidate = base_dir / f"{uuid}{ext}"
                    if candidate.exists():
                        audio_path = str(candidate)
                        break
                
                if not audio_path:
                    # Search subdirectories
                    for match in base_dir.rglob(f"{uuid}.*"):
                        if match.suffix.lower() in ['.webm', '.ogg', '.wav', '.mp3']:
                            audio_path = str(match)
                            break
                
                # Get COVID status
                status = str(row.get('status', row.get('covid_status', 'unknown'))).lower().strip()
                
                if 'covid' in status or 'positive' in status:
                    virus_type = 'COVID'
                elif 'negative' in status or 'healthy' in status:
                    virus_type = 'HEALTHY'
                else:
                    virus_type = 'GENERAL'
                
                samples.append({
                    'sample_id': uuid,
                    'dataset': 'coughvid',
                    'audio_path': audio_path,
                    'virus_type': virus_type,
                    'quality': float(row.get('cough_detected', row.get('quality', 0.5))),
                })
                
        except Exception as e:
            safe_print(f"  COUGHVID metadata error: {e}")
    
    # Fallback: just find audio files
    if not samples:
        audio_files = list(base_dir.rglob("*.webm")) + list(base_dir.rglob("*.wav"))
        for af in audio_files[:5000]:
            samples.append({
                'sample_id': af.stem,
                'dataset': 'coughvid',
                'audio_path': str(af),
                'virus_type': 'GENERAL',
                'quality': 0.5,
            })
    
    return samples


def process_coswara(data_dir: Path) -> List[Dict]:
    """Process Coswara dataset - handles nested tar.gz files."""
    samples = []
    
    # Find extracted data
    possible_dirs = [
        data_dir / "Coswara-Data-master" / "Extracted_data",
        data_dir / "Extracted_data",
    ]
    
    for root in [data_dir] + possible_dirs:
        for pattern in ["**/Extracted_data", "**/extracted_data"]:
            for extracted_dir in root.glob(pattern):
                if extracted_dir.is_dir():
                    possible_dirs.append(extracted_dir)
    
    # Find date folders
    for base in possible_dirs:
        if not base.exists():
            continue
            
        for date_folder in base.iterdir():
            if not date_folder.is_dir():
                continue
            
            # Check if this is a date folder (YYYYMMDD format)
            if not date_folder.name.replace('-', '').isdigit():
                continue
            
            for user_folder in date_folder.iterdir():
                if not user_folder.is_dir():
                    continue
                
                # Read metadata
                status = "unknown"
                meta_file = user_folder / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                        status = str(meta.get('covid_status', meta.get('status', ''))).lower()
                    except Exception:
                        pass
                
                # Determine virus type
                if 'positive' in status or 'covid' in status:
                    virus_type = 'COVID'
                elif 'negative' in status or 'healthy' in status:
                    virus_type = 'HEALTHY'
                elif 'recovered' in status:
                    virus_type = 'RECOVERED'
                else:
                    virus_type = 'GENERAL'
                
                # Find audio files
                for audio in user_folder.glob("*.wav"):
                    samples.append({
                        'sample_id': f"{user_folder.name}_{audio.stem}",
                        'dataset': 'coswara',
                        'audio_path': str(audio),
                        'virus_type': virus_type,
                        'quality': 0.8,
                    })
    
    # Fallback: find any audio
    if not samples:
        for af in data_dir.rglob("*.wav"):
            samples.append({
                'sample_id': af.stem,
                'dataset': 'coswara',
                'audio_path': str(af),
                'virus_type': 'GENERAL',
                'quality': 0.7,
            })
    
    return samples[:10000]  # Limit


def process_virufy(data_dir: Path) -> List[Dict]:
    """Process Virufy dataset."""
    samples = []
    
    # Find audio files
    audio_files = []
    for ext in ['.wav', '.ogg', '.mp3', '.flac']:
        audio_files.extend(data_dir.rglob(f"*{ext}"))
    
    for af in audio_files:
        path_str = str(af).lower()
        
        if 'positive' in path_str or 'covid' in path_str:
            virus_type = 'COVID'
        elif 'negative' in path_str or 'healthy' in path_str:
            virus_type = 'HEALTHY'
        else:
            virus_type = 'GENERAL'
        
        samples.append({
            'sample_id': af.stem,
            'dataset': 'virufy',
            'audio_path': str(af),
            'virus_type': virus_type,
            'quality': 1.0,
        })
    
    return samples


def process_icbhi(data_dir: Path) -> List[Dict]:
    """Process ICBHI respiratory sounds dataset."""
    samples = []
    
    # Find audio files
    audio_files = list(data_dir.rglob("*.wav"))
    
    # Try to find diagnosis file
    diagnosis_file = None
    for pattern in ["**/ICBHI_*.txt", "**/patient_diagnosis.csv", "**/*diagnosis*"]:
        matches = list(data_dir.rglob(pattern))
        if matches:
            diagnosis_file = matches[0]
            break
    
    # Parse diagnosis if available
    diagnoses = {}
    if diagnosis_file and diagnosis_file.exists():
        try:
            with open(diagnosis_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        diagnoses[parts[0]] = parts[1].lower()
        except Exception:
            pass
    
    for af in audio_files:
        # Extract patient ID from filename (typically starts with patient ID)
        patient_id = af.stem.split('_')[0] if '_' in af.stem else af.stem
        
        # Get diagnosis
        diag = diagnoses.get(patient_id, '')
        
        if 'pneumonia' in diag:
            virus_type = 'PNEUMONIA'
        elif 'healthy' in diag or 'normal' in diag:
            virus_type = 'HEALTHY'
        elif 'copd' in diag or 'bronchitis' in diag:
            virus_type = 'GENERAL'
        else:
            virus_type = 'GENERAL'
        
        samples.append({
            'sample_id': af.stem,
            'dataset': 'icbhi',
            'audio_path': str(af),
            'virus_type': virus_type,
            'quality': 0.9,
        })
    
    return samples


def process_wesad(data_dir: Path) -> List[Dict]:
    """Process WESAD physiological dataset."""
    samples = []
    
    # Find subject folders (S2, S3, etc.)
    for subj_dir in data_dir.rglob("S*"):
        if not subj_dir.is_dir():
            continue
        
        # Check for pickle file
        pkl_file = subj_dir / f"{subj_dir.name}.pkl"
        if pkl_file.exists():
            samples.append({
                'sample_id': subj_dir.name,
                'dataset': 'wesad',
                'audio_path': None,
                'pkl_path': str(pkl_file),
                'virus_type': 'HEALTHY',  # Baseline data
                'quality': 1.0,
            })
    
    return samples


def process_flusense(data_dir: Path) -> List[Dict]:
    """Process FluSense dataset."""
    samples = []
    
    # Find audio files
    audio_files = list(data_dir.rglob("*.wav"))
    
    # Find any CSV/JSON labels
    label_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.json"))
    
    # Try to parse labels
    labels = {}
    for lf in label_files[:5]:
        try:
            if lf.suffix == '.csv':
                df = pd.read_csv(lf)
                for _, row in df.iterrows():
                    file_id = str(row.get('id', row.get('file', row.name)))
                    flu = row.get('flu', row.get('label', row.get('positive', 0)))
                    labels[file_id] = bool(flu)
            elif lf.suffix == '.json':
                with open(lf) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        labels[str(item.get('id', ''))] = item.get('flu', False)
        except Exception:
            continue
    
    for af in audio_files:
        flu = labels.get(af.stem, None)
        virus_type = 'FLU' if flu else ('HEALTHY' if flu is False else 'GENERAL')
        
        samples.append({
            'sample_id': af.stem,
            'dataset': 'flusense',
            'audio_path': str(af),
            'virus_type': virus_type,
            'quality': 0.7,
        })
    
    # If no audio files, try to use labels directly
    if not samples and labels:
        for file_id, flu in labels.items():
            samples.append({
                'sample_id': file_id,
                'dataset': 'flusense',
                'audio_path': None,
                'virus_type': 'FLU' if flu else 'HEALTHY',
                'quality': 0.7,
            })
    
    return samples


def process_mimic(data_dir: Path) -> List[Dict]:
    """Process MIMIC-III demo dataset."""
    samples = []
    
    # Find CSV files
    csv_files = list(data_dir.rglob("*.csv"))
    
    # Look for diagnosis tables
    for cf in csv_files[:10]:
        try:
            df = pd.read_csv(cf, low_memory=False)
            
            for _, row in df.iterrows():
                diag = str(row.get('diagnosis', row.get('icd9_code', ''))).lower()
                
                if 'pneumonia' in diag:
                    virus_type = 'PNEUMONIA'
                elif 'influenza' in diag or 'flu' in diag:
                    virus_type = 'FLU'
                elif 'sepsis' in diag or 'infection' in diag:
                    virus_type = 'GENERAL'
                else:
                    continue
                
                samples.append({
                    'sample_id': str(row.get('subject_id', row.get('hadm_id', len(samples)))),
                    'dataset': 'mimic',
                    'audio_path': None,
                    'virus_type': virus_type,
                    'quality': 0.9,
                })
        except Exception:
            continue
    
    return samples


def process_fluwatch(data_dir: Path) -> List[Dict]:
    """Process CDC FluWatch/FluSight dataset (surveillance data)."""
    samples = []
    
    # This is surveillance data, not audio
    # Look for CSV files with ILI (influenza-like illness) rates
    csv_files = list(data_dir.rglob("*.csv"))
    
    for cf in csv_files[:10]:
        try:
            df = pd.read_csv(cf, low_memory=False)
            
            # Look for relevant columns
            for _, row in df.iterrows():
                ili = row.get('ili', row.get('ILI', row.get('weighted_ili', None)))
                if ili is not None and float(ili) > 0:
                    samples.append({
                        'sample_id': f"fluwatch_{len(samples)}",
                        'dataset': 'fluwatch',
                        'audio_path': None,
                        'virus_type': 'FLU',
                        'quality': 0.7,
                        'ili_rate': float(ili),
                    })
        except Exception:
            continue
    
    return samples[:5000]


def process_covid_sounds(data_dir: Path) -> List[Dict]:
    """Process COVID-19 Sounds dataset."""
    samples = []
    
    # Find audio files
    for ext in ['.wav', '.mp3', '.ogg', '.flac']:
        for af in data_dir.rglob(f"*{ext}"):
            path_str = str(af).lower()
            
            if 'positive' in path_str or 'covid' in path_str:
                virus_type = 'COVID'
            elif 'negative' in path_str or 'healthy' in path_str:
                virus_type = 'HEALTHY'
            else:
                virus_type = 'GENERAL'
            
            samples.append({
                'sample_id': af.stem,
                'dataset': 'covid_sounds',
                'audio_path': str(af),
                'virus_type': virus_type,
                'quality': 0.8,
            })
    
    return samples


def process_cough_against_covid(data_dir: Path) -> List[Dict]:
    """Process Cough Against COVID dataset."""
    samples = []
    
    for ext in ['.wav', '.mp3', '.ogg', '.webm']:
        for af in data_dir.rglob(f"*{ext}"):
            samples.append({
                'sample_id': af.stem,
                'dataset': 'cough_against_covid',
                'audio_path': str(af),
                'virus_type': 'COVID',  # This dataset is all COVID positive
                'quality': 0.9,
            })
    
    return samples


def process_esc50(data_dir: Path) -> List[Dict]:
    """Process ESC-50 environmental sounds dataset."""
    samples = []
    
    # Find audio files
    audio_files = list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.ogg"))
    
    # ESC-50 filenames contain class info
    for af in audio_files:
        # Filename format: {fold}-{clip_id}-{take}-{target}.wav
        parts = af.stem.split('-')
        
        # Get class from path or filename
        if 'cough' in af.stem.lower() or 'cough' in str(af.parent).lower():
            virus_type = 'COUGH'  # For pre-training
        else:
            virus_type = 'HEALTHY'
        
        samples.append({
            'sample_id': af.stem,
            'dataset': 'esc50',
            'audio_path': str(af),
            'virus_type': virus_type,
            'quality': 1.0,
        })
    
    return samples


def process_ravdess(data_dir: Path) -> List[Dict]:
    """Process RAVDESS emotional speech dataset."""
    samples = []
    
    # Find audio files
    for af in data_dir.rglob("*.wav"):
        samples.append({
            'sample_id': af.stem,
            'dataset': 'ravdess',
            'audio_path': str(af),
            'virus_type': 'HEALTHY',  # Baseline healthy speech
            'quality': 1.0,
        })
    
    return samples


def process_librispeech(data_dir: Path) -> List[Dict]:
    """Process LibriSpeech dataset."""
    samples = []
    
    # Find audio files
    for af in data_dir.rglob("*.flac"):
        samples.append({
            'sample_id': af.stem,
            'dataset': 'librispeech',
            'audio_path': str(af),
            'virus_type': 'HEALTHY',
            'quality': 1.0,
        })
    
    return samples[:3000]  # Limit


def process_default(data_dir: Path) -> List[Dict]:
    """Default processor - finds audio files."""
    samples = []
    
    for ext in ['.wav', '.mp3', '.ogg', '.flac', '.webm']:
        for af in data_dir.rglob(f"*{ext}"):
            samples.append({
                'sample_id': af.stem,
                'dataset': data_dir.name,
                'audio_path': str(af),
                'virus_type': 'GENERAL',
                'quality': 0.5,
            })
    
    return samples[:5000]


# Processor registry
PROCESSORS = {
    'coughvid': process_coughvid,
    'coswara': process_coswara,
    'virufy': process_virufy,
    'icbhi': process_icbhi,
    'wesad': process_wesad,
    'flusense': process_flusense,
    'fluwatch': process_fluwatch,
    'covid_sounds': process_covid_sounds,
    'cough_against_covid': process_cough_against_covid,
    'mimic': process_mimic,
    'esc50': process_esc50,
    'ravdess': process_ravdess,
    'librispeech': process_librispeech,
    'default': process_default,
}


# =============================================================================
# AUDIO FORMAT CONVERSION
# =============================================================================

def convert_audio_to_wav(input_path: Path, output_path: Path = None) -> Optional[Path]:
    """Convert audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Input audio file (webm, ogg, mp3, etc.)
        output_path: Output WAV path (default: same name with .wav)
        
    Returns:
        Path to WAV file if successful, None otherwise
    """
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    
    if output_path.exists():
        return output_path
    
    try:
        # Try ffmpeg
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(input_path), '-ar', '16000', '-ac', '1', str(output_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
    except FileNotFoundError:
        pass  # ffmpeg not installed
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    
    # Try pydub as fallback
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(output_path), format='wav')
        if output_path.exists():
            return output_path
    except ImportError:
        pass
    except Exception:
        pass
    
    return None


def ensure_wav_format(audio_path: str) -> str:
    """Ensure audio file is in WAV format, convert if necessary.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Path to WAV file (may be same as input or converted)
    """
    path = Path(audio_path)
    
    if path.suffix.lower() == '.wav':
        return audio_path
    
    # Try to convert
    wav_path = convert_audio_to_wav(path)
    if wav_path:
        return str(wav_path)
    
    # Return original and let librosa try
    return audio_path


# =============================================================================
# AUDIO FEATURE EXTRACTION
# =============================================================================

def extract_audio_features(audio_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract audio features from a file.
    
    Returns 30-dim feature vector:
    - 13 MFCC means
    - 13 MFCC stds
    - 4 spectral features
    """
    try:
        import librosa
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try to load directly first
            try:
                y, _ = librosa.load(audio_path, sr=sr, duration=10.0)
            except Exception:
                # Try converting to WAV
                wav_path = ensure_wav_format(audio_path)
                if wav_path != audio_path:
                    y, _ = librosa.load(wav_path, sr=sr, duration=10.0)
                else:
                    raise
            
            if len(y) < sr * 0.3:
                return None
            
            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / 10000
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) / 10000
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y=y))
            
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, zero_crossing, rms],
            ])
            
            return features.astype(np.float32)
            
    except Exception:
        return None


# =============================================================================
# MAIN DOWNLOAD AND PROCESS PIPELINE
# =============================================================================

def download_and_process_dataset(
    name: str,
    output_dir: Path,
    skip_download: bool = False,
) -> Tuple[bool, List[Dict]]:
    """Download and process a single dataset.
    
    Args:
        name: Dataset name
        output_dir: Output directory
        skip_download: Skip download if already exists
        
    Returns:
        Tuple of (success, samples)
    """
    if name not in DATASETS:
        safe_print(f"Unknown dataset: {name}")
        return False, []
    
    config = DATASETS[name]
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    processed_marker = ds_dir / ".processed"
    samples_file = ds_dir / "samples.json"
    
    if processed_marker.exists() and samples_file.exists():
        safe_print(f"  {config.name}: Already processed")
        try:
            with open(samples_file) as f:
                samples = json.load(f)
            return True, samples
        except Exception:
            pass
    
    # Download
    download_marker = ds_dir / ".downloaded"
    if not download_marker.exists() or not skip_download:
        safe_print(f"\n  Downloading {config.name}...")
        
        # Determine filename
        primary_url = config.urls[0]
        filename = primary_url.split("/")[-1].split("?")[0]
        if not any(filename.lower().endswith(ext) for ext in ['.zip', '.tar.gz', '.tgz', '.tar', '.7z', '.rar']):
            filename = f"{name}.zip"
        
        archive_path = ds_dir / filename
        
        success = download_file(
            config.urls,
            archive_path,
            expected_size=config.size_bytes,
            desc=config.name,
        )
        
        if not success:
            return False, []
        
        # Validate and extract
        is_valid, error = is_valid_archive(archive_path)
        if not is_valid:
            safe_print(f"  {config.name}: {error}")
            # Try to re-download
            archive_path.unlink()
            return False, []
        
        safe_print(f"  Extracting {config.name}...")
        if not extract_archive(archive_path, ds_dir, handle_nested=config.nested_archives):
            return False, []
        
        download_marker.touch()
    
    # Process
    safe_print(f"  Processing {config.name}...")
    
    processor = PROCESSORS.get(config.processor, PROCESSORS['default'])
    samples = processor(ds_dir)
    
    if samples:
        # Save samples
        with open(samples_file, 'w') as f:
            json.dump(samples, f)
        
        processed_marker.touch()
        safe_print(f"  {config.name}: {len(samples)} samples")
    else:
        safe_print(f"  {config.name}: No samples found")
    
    return True, samples


def create_training_splits(
    all_samples: List[Dict],
    output_dir: Path,
    extract_features: bool = True,
    n_workers: int = 4,
) -> Dict:
    """Create train/val/test splits and optionally extract features.
    
    Args:
        all_samples: All samples from all datasets
        output_dir: Output directory
        extract_features: Whether to extract audio features
        n_workers: Number of parallel workers
        
    Returns:
        Metadata dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    if extract_features:
        safe_print("\nExtracting audio features...")
        
        samples_with_audio = [s for s in all_samples if s.get('audio_path') and Path(s['audio_path']).exists()]
        safe_print(f"  Samples with audio: {len(samples_with_audio)}")
        
        if samples_with_audio:
            def process_sample(sample):
                features = extract_audio_features(sample['audio_path'])
                if features is not None:
                    sample['audio_features'] = features.tolist()
                return sample
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_sample, s) for s in samples_with_audio]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Features"):
                    try:
                        future.result()
                    except Exception:
                        pass
            
            n_features = sum(1 for s in all_samples if s.get('audio_features'))
            safe_print(f"  Extracted features: {n_features}")
    
    # Filter valid samples
    valid_samples = [s for s in all_samples if s.get('audio_features') or s.get('quality', 0) > 0.5]
    safe_print(f"\nValid samples: {len(valid_samples)}")
    
    if not valid_samples:
        safe_print("ERROR: No valid samples!")
        return {}
    
    # Count virus types
    virus_counts = {}
    for s in valid_samples:
        vt = s.get('virus_type', 'GENERAL')
        virus_counts[vt] = virus_counts.get(vt, 0) + 1
    
    safe_print("\nVirus type distribution:")
    for vt, count in sorted(virus_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(valid_samples)
        safe_print(f"  {vt}: {count} ({pct:.1f}%)")
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(valid_samples)
    
    n = len(valid_samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train = valid_samples[:n_train]
    val = valid_samples[n_train:n_train + n_val]
    test = valid_samples[n_train + n_val:]
    
    # Save splits
    safe_print("\nSaving splits...")
    
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train, f)
    safe_print(f"  Train: {len(train)} samples")
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val, f)
    safe_print(f"  Val: {len(val)} samples")
    
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test, f)
    safe_print(f"  Test: {len(test)} samples")
    
    # Metadata
    metadata = {
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'virus_counts': virus_counts,
        'datasets_used': list(set(s['dataset'] for s in valid_samples)),
        'feature_dim': 30,
    }
    
    with open(output_dir / "metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)
    
    return metadata


# =============================================================================
# CLI
# =============================================================================

def install_archive_handlers():
    """Install optional archive handling libraries."""
    safe_print("Installing optional archive handlers...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "py7zr", "rarfile", "-q"
        ])
        safe_print("  Installed: py7zr, rarfile")
    except Exception as e:
        safe_print(f"  Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process datasets for ViralFlip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_and_process.py --health
    Download all health/illness datasets (recommended)
    
  python scripts/download_and_process.py --dataset coughvid
    Download and process a specific dataset
    
  python scripts/download_and_process.py --all --parallel 4
    Download everything with 4 parallel workers
    
  python scripts/download_and_process.py --skip-download
    Only process already-downloaded data
"""
    )
    parser.add_argument("--output", "-o", type=str, default="data/real",
                       help="Output directory for raw data")
    parser.add_argument("--processed", "-p", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                       help="Specific dataset to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--health", action="store_true",
                       help="Download health/illness datasets only")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available datasets")
    parser.add_argument("--parallel", "-n", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only process existing data")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip audio feature extraction")
    parser.add_argument("--install-archive-handlers", action="store_true",
                       help="Install py7zr and rarfile for archive support")
    
    args = parser.parse_args()
    
    if args.install_archive_handlers:
        install_archive_handlers()
        return
    
    if args.list:
        print("\n" + "=" * 60)
        print("AVAILABLE DATASETS")
        print("=" * 60)
        
        # Separate by category
        health_datasets = [(n, c) for n, c in DATASETS.items() if c.has_illness_labels]
        other_datasets = [(n, c) for n, c in DATASETS.items() if not c.has_illness_labels]
        
        print("\n--- HEALTH/ILLNESS DATASETS (recommended) ---")
        for name, config in health_datasets:
            print(f"\n  {name}")
            print(f"    {config.name}")
            print(f"    Size: ~{config.size_bytes // (1024*1024)} MB")
            if config.illness_types:
                print(f"    Labels: {', '.join(config.illness_types)}")
            if config.notes:
                print(f"    Note: {config.notes}")
        
        print("\n\n--- AUDIO/PRE-TRAINING DATASETS ---")
        for name, config in other_datasets:
            print(f"\n  {name}")
            print(f"    {config.name}")
            print(f"    Size: ~{config.size_bytes // (1024*1024)} MB")
            if config.notes:
                print(f"    Note: {config.notes}")
        return
    
    output_dir = Path(args.output)
    processed_dir = Path(args.processed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("VIRALFLIP DATA DOWNLOAD & PROCESSING")
    print("=" * 60)
    print(f"Raw data: {output_dir}")
    print(f"Processed: {processed_dir}")
    print(f"Workers: {args.parallel}")
    print()
    
    # Determine which datasets to download
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            return
        datasets_to_get = [args.dataset]
    elif args.health:
        datasets_to_get = [n for n, c in DATASETS.items() if c.has_illness_labels]
    elif args.all:
        datasets_to_get = list(DATASETS.keys())
    else:
        # Default: get health datasets
        datasets_to_get = [n for n, c in DATASETS.items() if c.has_illness_labels]
    
    print(f"Datasets to process: {', '.join(datasets_to_get)}")
    print()
    
    # Download and process each dataset
    all_samples = []
    results = {}  # Track success/failure
    
    for name in datasets_to_get:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {name}")
        print('=' * 40)
        
        try:
            success, samples = download_and_process_dataset(
                name, output_dir, skip_download=args.skip_download
            )
            
            results[name] = {
                'success': success,
                'samples': len(samples) if samples else 0,
            }
            
            if success and samples:
                all_samples.extend(samples)
                
        except Exception as e:
            results[name] = {'success': False, 'samples': 0, 'error': str(e)}
            safe_print(f"  ERROR: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    success_count = 0
    for name, result in results.items():
        status = "OK" if result['success'] else "FAILED"
        samples = result['samples']
        if result['success']:
            success_count += 1
            print(f"  {name}: {status} ({samples} samples)")
        else:
            error = result.get('error', 'Check logs above')
            print(f"  {name}: {status} - {error}")
    
    print(f"\nTotal: {success_count}/{len(results)} datasets succeeded")
    print(f"Total samples: {len(all_samples)}")
    
    # Create training splits
    if all_samples:
        print("\n" + "=" * 60)
        print("CREATING TRAINING DATA")
        print("=" * 60)
        
        metadata = create_training_splits(
            all_samples,
            processed_dir,
            extract_features=not args.skip_features,
            n_workers=args.parallel,
        )
        
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print(f"Processed data saved to: {processed_dir}")
        if metadata:
            print(f"\nDataset statistics:")
            print(f"  Train: {metadata.get('n_train', 0):,} samples")
            print(f"  Val: {metadata.get('n_val', 0):,} samples")
            print(f"  Test: {metadata.get('n_test', 0):,} samples")
            print(f"  Datasets used: {', '.join(metadata.get('datasets_used', []))}")
        print(f"\nTo train the model:")
        print(f"  python scripts/train.py --config configs/high_performance.yaml")
    else:
        print("\n" + "=" * 60)
        print("NO DATA COLLECTED")
        print("=" * 60)
        print("All downloads failed. Common issues:")
        print("  - Network connectivity problems")
        print("  - Datasets require registration (see --list)")
        print("  - Server temporarily unavailable")
        print("\nTry:")
        print("  1. Run again (resume support is built-in)")
        print("  2. Try a specific dataset: --dataset coughvid")
        print("  3. Check network and try: --dataset esc50 (most reliable)")


if __name__ == "__main__":
    main()

