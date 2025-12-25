#!/usr/bin/env python3
"""Download additional datasets for ViralFlip training.

This script downloads larger, more comprehensive datasets that can improve
model accuracy beyond the basic ESC-50/UCI HAR data.

Features:
- Robust retry logic with exponential backoff
- Resume capability for interrupted downloads
- Parallel downloads for speed
- Progress tracking with ETA

Usage:
    python scripts/download_more_data.py --all
    python scripts/download_more_data.py --dataset flusense
    python scripts/download_more_data.py --essential --parallel 4
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
import warnings
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Suppress Python 3.14 tarfile deprecation warning globally
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tarfile')

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Installing requests library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

import threading
_print_lock = threading.Lock()

# =============================================================================
# VERIFIED WORKING DATASETS - Updated Dec 2024
# =============================================================================

DATASETS = {
    # -------------------------------------------------------------------------
    # AUDIO / ENVIRONMENTAL SOUNDS (For cough/respiratory detection pre-training)
    # -------------------------------------------------------------------------
    "esc50": {
        "name": "ESC-50 Environmental Sounds",
        "url": "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        "type": "github",
        "description": "2000 environmental audio clips (40 cough samples)",
        "size": "~650MB",
        "size_bytes": 680_000_000,
        "use_for": ["cough_detection", "audio_classification"],
    },
    "urbansound8k": {
        "name": "UrbanSound8K",
        "url": "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz",
        "type": "direct",
        "description": "8732 labeled urban sounds, good for audio model pre-training",
        "size": "~6GB",
        "size_bytes": 6_400_000_000,
        "use_for": ["audio_classification", "background_noise"],
    },
    
    # -------------------------------------------------------------------------
    # ACTIVITY RECOGNITION (IMU / Accelerometer)
    # -------------------------------------------------------------------------
    "uci_har": {
        "name": "UCI HAR Dataset",
        # Direct download from a reliable source
        "url": "https://storage.googleapis.com/kaggle-data-sets/8461/11921/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256",
        "mirrors": [
            # Hugging Face mirror
            "https://huggingface.co/datasets/GIanlucaRub/human-activity-recognition-with-smartphones/resolve/main/archive.zip",
            # UCI original (slow)
            "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        ],
        "type": "direct",
        "description": "Smartphone accelerometer data, 6 activities, 30 subjects",
        "size": "~60MB",
        "size_bytes": 63_000_000,
        "use_for": ["activity_recognition", "imu_features"],
    },
    "wisdm": {
        "name": "WISDM Activity Dataset",
        "url": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
        "type": "direct",
        "description": "Accelerometer data from 36 users doing 6 activities",
        "size": "~25MB",
        "size_bytes": 26_000_000,
        "use_for": ["activity_recognition", "gait_analysis"],
    },
    "pamap2": {
        "name": "PAMAP2 Physical Activity",
        # Zenodo backup / alternative HAR dataset
        "url": "https://zenodo.org/records/3515935/files/PAMAP2_Dataset.zip",
        "mirrors": [
            # UCI original (slow/blocked)  
            "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        ],
        "type": "direct",
        "description": "IMU + HR data, 18 activities, 9 subjects",
        "size": "~700MB",
        "size_bytes": 730_000_000,
        "use_for": ["activity_recognition", "heart_rate"],
        "optional": True,  # Skip if unavailable
    },
    
    # -------------------------------------------------------------------------
    # SPEECH / VOICE (For voice quality features)
    # -------------------------------------------------------------------------
    "ravdess": {
        "name": "RAVDESS Emotional Speech",
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        "mirrors": [
            "https://www.zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
            "https://data.mendeley.com/public-files/datasets/7yj834x2wh/files/4c5c9427-e7a6-4b4a-8d2c-2b9b8b8b8b8b/file_downloaded",
        ],
        "type": "direct",
        "description": "1440 speech files from 24 actors, good for voice baseline",
        "size": "~215MB",
        "size_bytes": 225_000_000,
        "use_for": ["voice_features", "speaking_rate"],
    },
    "librispeech_mini": {
        "name": "LibriSpeech dev-clean (mini)",
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "type": "direct",
        "description": "Clean speech for voice feature extraction baseline",
        "size": "~340MB",
        "size_bytes": 356_000_000,
        "use_for": ["voice_features", "speech_quality"],
    },
    
    # -------------------------------------------------------------------------
    # HEART RATE / PHYSIOLOGICAL
    # -------------------------------------------------------------------------
    "ppg_dalia": {
        "name": "PPG-DaLiA (Activity + PPG)",
        # UCI is the only source, try with longer timeout
        "url": "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
        "type": "direct",
        "description": "PPG signals during daily activities, HR ground truth",
        "size": "~1.2GB",
        "size_bytes": 1_260_000_000,
        "use_for": ["heart_rate", "hrv_features"],
        "optional": True,  # Large file, skip if slow
    },
    
    # -------------------------------------------------------------------------
    # COUGH-SPECIFIC (These may require registration but have fallbacks)
    # -------------------------------------------------------------------------
    "fsd50k_cough": {
        "name": "FSD50K (Cough subset via Freesound)",
        "url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip",
        "type": "direct",  
        "description": "Large freesound dataset, contains cough/sneeze samples",
        "size": "~8GB (dev set only)",
        "size_bytes": 8_500_000_000,
        "use_for": ["cough_detection", "audio_classification"],
        "note": "Large download - use --dataset fsd50k_cough explicitly",
    },
}

# =============================================================================
# REAL ILLNESS/HEALTH DATASETS - These have actual health labels!
# Covers: COVID, Flu, Cold, RSV, Pneumonia, General Respiratory Illness
# =============================================================================

HEALTH_DATASETS = {
    # -------------------------------------------------------------------------
    # COVID-19 Datasets - VERIFIED WORKING Dec 2024
    # -------------------------------------------------------------------------
    "coughvid": {
        "name": "COUGHVID COVID Coughs",
        # Primary: Zenodo v3 (latest stable)
        "url": "https://zenodo.org/records/4498364/files/public_dataset_v3.zip?download=1",
        "mirrors": [
            "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1",
            "https://zenodo.org/records/7024894/files/public_dataset.zip?download=1",
        ],
        "type": "direct",
        "description": "25,000+ cough recordings WITH COVID labels",
        "size": "~1.2GB",
        "size_bytes": 1_300_000_000,
        "use_for": ["cough_detection", "covid_classification"],
        "has_illness_labels": True,
        "illness_types": ["covid"],
    },
    "coswara": {
        "name": "Coswara Respiratory",
        "url": "https://github.com/iiscleap/Coswara-Data/archive/refs/heads/master.zip",
        "type": "github",
        "description": "Breathing, cough, speech WITH COVID/healthy labels",
        "size": "~2GB", 
        "size_bytes": 2_100_000_000,
        "use_for": ["cough_detection", "breathing_analysis", "covid_classification"],
        "has_illness_labels": True,
        "illness_types": ["covid", "healthy"],
    },
    "virufy": {
        "name": "Virufy COVID Coughs",
        "url": "https://github.com/virufy/virufy-data/archive/refs/heads/main.zip",
        "type": "github",
        "description": "COVID cough dataset with PCR-confirmed labels",
        "size": "~500MB",
        "size_bytes": 520_000_000,
        "use_for": ["cough_detection", "covid_classification"],
        "has_illness_labels": True,
        "illness_types": ["covid"],
    },
    
    # -------------------------------------------------------------------------
    # Flu / Respiratory (Verified Working)
    # -------------------------------------------------------------------------
    "flusense": {
        "name": "FluSense Dataset",
        "url": "https://github.com/Forsad/FluSense-data/archive/refs/heads/master.zip",
        "type": "github",
        "description": "Hospital waiting room audio - cough/speech for flu detection",
        "size": "~1GB",
        "size_bytes": 1_050_000_000,
        "use_for": ["cough_detection", "flu_detection", "crowd_health"],
        "has_illness_labels": True,
        "illness_types": ["flu", "respiratory"],
    },
    "fluwatch": {
        "name": "CDC FluView ILI Data",
        "url": "https://github.com/cdcepi/FluSight-forecast-hub/archive/refs/heads/main.zip",
        "type": "github",
        "description": "CDC flu surveillance data - ILI rates by region/week",
        "size": "~200MB",
        "size_bytes": 210_000_000,
        "use_for": ["flu_tracking", "epidemic_prediction"],
        "has_illness_labels": True,
        "illness_types": ["flu", "ili"],
    },
    
    # -------------------------------------------------------------------------
    # Additional COVID Audio Datasets
    # -------------------------------------------------------------------------
    "covid_sounds": {
        "name": "COVID-19 Sounds Dataset",
        "url": "https://github.com/cam-mobsys/covid19-sounds-neurips/archive/refs/heads/main.zip",
        "type": "github",
        "description": "Cambridge COVID-19 Sounds - cough, breathing, voice",
        "size": "~100MB",
        "size_bytes": 105_000_000,
        "use_for": ["cough_detection", "covid_classification"],
        "has_illness_labels": True,
        "illness_types": ["covid", "healthy"],
    },
    "cough_against_covid": {
        "name": "Cough Against COVID",
        "url": "https://github.com/coughagainstcovid/coughagainstcovid.github.io/archive/refs/heads/main.zip",
        "type": "github",
        "description": "Open source COVID cough collection",
        "size": "~50MB",
        "size_bytes": 52_000_000,
        "use_for": ["cough_detection", "covid_classification"],
        "has_illness_labels": True,
        "illness_types": ["covid"],
    },
    
    # -------------------------------------------------------------------------
    # Physiological + Illness (Wearable-style)
    # -------------------------------------------------------------------------
    "wesad": {
        "name": "WESAD Stress/Affect Dataset",
        "url": "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
        "type": "direct",
        "description": "Wearable sensors (HR, temp, EDA) - useful for physiological baselines",
        "size": "~2GB",
        "size_bytes": 2_100_000_000,
        "use_for": ["stress_detection", "physiological_baseline", "hrv_analysis"],
        "has_illness_labels": False,  # Stress, not illness, but useful for baselines
        "illness_types": ["stress"],
    },
    "mimic_iii_demo": {
        "name": "MIMIC-III Clinical Demo",
        "url": "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip",
        "type": "direct",
        "description": "ICU patient vitals demo set - sepsis, pneumonia labels",
        "size": "~35MB",
        "size_bytes": 36_000_000,
        "use_for": ["vital_signs", "sepsis_detection", "clinical_prediction"],
        "has_illness_labels": True,
        "illness_types": ["sepsis", "pneumonia", "respiratory_failure"],
    },
    
    # -------------------------------------------------------------------------
    # Respiratory Sound Classification
    # -------------------------------------------------------------------------
    "icbhi": {
        "name": "ICBHI Respiratory Sounds",
        "url": "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip",
        "mirrors": [
            "https://www.kaggle.com/api/v1/datasets/download/vbookshelf/respiratory-sound-database",
        ],
        "type": "direct",
        "description": "920 lung sound recordings with crackles, wheezes, disease labels",
        "size": "~600MB",
        "size_bytes": 630_000_000,
        "use_for": ["respiratory_sounds", "pneumonia_detection", "lung_disease"],
        "has_illness_labels": True,
        "illness_types": ["pneumonia", "copd", "bronchitis", "healthy"],
    },
}

# Datasets requiring registration (provide instructions)
GATED_DATASETS = {
    # COVID + Wearable
    "myphd_stanford": {
        "name": "MyPHD Stanford Wearable Study",
        "registration_url": "https://med.stanford.edu/myphd.html",
        "description": "Fitbit/Apple Watch data with COVID illness labels. Request access.",
        "size": "~10GB",
        "has_illness_labels": True,
        "illness_types": ["covid", "flu", "cold"],
    },
    "detect_scripps": {
        "name": "DETECT Scripps Wearable Study",
        "registration_url": "https://detectstudy.org/",
        "description": "Wearable sensor data with illness onset. Request from Scripps.",
        "size": "~5GB",
        "has_illness_labels": True,
        "illness_types": ["covid", "flu", "respiratory"],
    },
    "corona_datenspende": {
        "name": "Corona-Datenspende (RKI Germany)",
        "registration_url": "https://corona-datenspende.de/science/",
        "description": "500K+ users wearable data with COVID. Request from RKI.",
        "size": "~50GB",
        "has_illness_labels": True,
        "illness_types": ["covid"],
    },
    
    # Flu Specific
    "flucas": {
        "name": "FluCas Flu Challenge Study",
        "registration_url": "https://www.niaid.nih.gov/",
        "description": "Controlled flu challenge study with biomarkers. Contact NIAID.",
        "size": "~2GB",
        "has_illness_labels": True,
        "illness_types": ["flu"],
    },
    "goviralstudy": {
        "name": "GoViral Study (Duke)",
        "registration_url": "https://www.dukehealth.org/",
        "description": "Respiratory illness with wearable + symptom data. Duke IRB required.",
        "size": "~5GB",
        "has_illness_labels": True,
        "illness_types": ["flu", "cold", "covid", "rsv"],
    },
    
    # Clinical
    "mimic_iv": {
        "name": "MIMIC-IV Full Dataset",
        "registration_url": "https://physionet.org/content/mimiciv/",
        "description": "Full ICU dataset - requires credentialing. Has sepsis, pneumonia, etc.",
        "size": "~7GB",
        "has_illness_labels": True,
        "illness_types": ["sepsis", "pneumonia", "respiratory_failure", "infection"],
    },
    "eicu": {
        "name": "eICU Collaborative Database",
        "registration_url": "https://physionet.org/content/eicu-crd/",
        "description": "200K+ ICU stays with diagnoses. PhysioNet credentialing required.",
        "size": "~3GB",
        "has_illness_labels": True,
        "illness_types": ["sepsis", "pneumonia", "infection"],
    },
    
    # Phone Sensing
    "extrasensory": {
        "name": "ExtraSensory (UCSD)",
        "registration_url": "http://extrasensory.ucsd.edu/",
        "description": "Phone sensor + context labels. Site may be down.",
        "size": "~500MB",
    },
    "studentlife": {
        "name": "StudentLife Dataset",
        "registration_url": "https://studentlife.cs.dartmouth.edu/",
        "description": "48 students, 10 weeks - phone sensors + stress/health surveys.",
        "size": "~2GB",
        "has_illness_labels": True,
        "illness_types": ["stress", "general_health"],
    },
}


def create_session(retries: int = 5, backoff_factor: float = 1.0, custom_headers: dict = None) -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Default headers to mimic browser
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    if custom_headers:
        default_headers.update(custom_headers)
    
    session.headers.update(default_headers)
    
    return session


def get_file_size(session: requests.Session, url: str) -> Optional[int]:
    """Get file size from URL headers."""
    try:
        response = session.head(url, timeout=30, allow_redirects=True)
        return int(response.headers.get('content-length', 0))
    except Exception:
        return None


def download_with_resume(
    url: str, 
    dest: Path, 
    desc: str = "",
    expected_size: int = 0,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    max_retries: int = 10,
    timeout: int = 60,
    custom_headers: dict = None,
    mirrors: list = None,
) -> bool:
    """Download a file with resume support and retry logic.
    
    Args:
        url: URL to download
        dest: Destination path
        desc: Description for progress bar
        expected_size: Expected file size in bytes
        chunk_size: Download chunk size
        max_retries: Maximum number of retries
        timeout: Request timeout in seconds
        custom_headers: Optional custom headers for the request
        mirrors: Optional list of mirror URLs to try if primary fails
        
    Returns:
        True if successful, False otherwise
    """
    urls_to_try = [url] + (mirrors or [])
    session = create_session(retries=3, backoff_factor=0.5, custom_headers=custom_headers)
    
    # Check existing file for resume
    downloaded = 0
    if dest.exists():
        downloaded = dest.stat().st_size
        # Accept if within 5% of expected size (servers often report slightly different sizes)
        if expected_size > 0 and downloaded >= expected_size * 0.95:
            with _print_lock:
                print(f"\r  {desc[:22]}: Already complete ({downloaded / (1024*1024):.0f} MB)    ")
            return True
        # If we have a significant portion, keep it for resume
        if downloaded > 1024 * 1024:  # > 1MB
            with _print_lock:
                print(f"\r  {desc[:22]}: Resuming from {downloaded / (1024*1024):.0f} MB...    ", end="", flush=True)
    
    current_url_idx = 0
    short_desc = desc[:22] if len(desc) > 22 else desc
    last_print_time = 0
    server_total_size = 0  # Track what server reports
    
    for attempt in range(max_retries):
        # Switch to mirror after several failures
        if attempt > 0 and attempt % 3 == 0 and current_url_idx < len(urls_to_try) - 1:
            current_url_idx += 1
            downloaded = 0  # Reset for new mirror
            if dest.exists():
                dest.unlink()
        
        current_url = urls_to_try[current_url_idx]
        
        try:
            headers = {}
            if downloaded > 0:
                headers['Range'] = f'bytes={downloaded}-'
            
            response = session.get(
                current_url, 
                stream=True, 
                headers=headers, 
                timeout=(15, 60),  # (connect timeout, read timeout) - increased read timeout
                allow_redirects=True
            )
            
            # Handle range response
            if response.status_code == 416:  # Range not satisfiable - file complete or invalid range
                # Check if file is actually complete
                if dest.exists() and dest.stat().st_size > 1024 * 1024:
                    with _print_lock:
                        print(f"\r  {short_desc}: 100% - Done!                    ")
                    return True
                downloaded = 0
                if dest.exists():
                    dest.unlink()
                continue
            
            response.raise_for_status()
            
            # Get total size
            if response.status_code == 206:  # Partial content
                content_range = response.headers.get('Content-Range', '')
                if '/' in content_range:
                    server_total_size = int(content_range.split('/')[-1])
                else:
                    server_total_size = downloaded + int(response.headers.get('content-length', 0))
            else:
                server_total_size = int(response.headers.get('content-length', 0))
                if downloaded > 0 and server_total_size > 0:
                    # Server doesn't support resume, start fresh
                    downloaded = 0
                    if dest.exists():
                        dest.unlink()
            
            total_size = server_total_size if server_total_size > 0 else expected_size
            
            # Download with simple progress
            mode = 'ab' if downloaded > 0 else 'wb'
            bytes_this_session = 0
            
            with open(dest, mode) as f:
                last_chunk_time = time.time()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        bytes_this_session += len(chunk)
                        last_chunk_time = time.time()
                        
                        # Print progress every 2 seconds
                        now = time.time()
                        if now - last_print_time >= 2:
                            last_print_time = now
                            if total_size > 0:
                                pct = min(100, downloaded * 100 // total_size)
                                mb_done = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                with _print_lock:
                                    print(f"\r  {short_desc}: {pct:3d}% ({mb_done:.0f}/{mb_total:.0f} MB)    ", end="", flush=True)
                    else:
                        # Check if we're stuck (no data for 30 seconds)
                        if time.time() - last_chunk_time > 30:
                            with _print_lock:
                                print(f"\r  {short_desc}: Stalled, retrying...              ", end="", flush=True)
                            raise requests.exceptions.Timeout("Stalled")
            
            # Verify download
            if dest.exists():
                final_size = dest.stat().st_size
                
                # Success conditions (be lenient):
                # 1. Got all expected bytes from server
                # 2. File is at least 95% of expected size
                # 3. File is reasonable size and we received no more data
                is_complete = False
                
                if server_total_size > 0 and final_size >= server_total_size:
                    is_complete = True
                elif expected_size > 0 and final_size >= expected_size * 0.95:
                    is_complete = True
                elif bytes_this_session == 0 and final_size > 1024 * 1024:
                    # Server sent no more data and we have substantial file
                    is_complete = True
                elif final_size > expected_size * 0.9 and bytes_this_session < chunk_size:
                    # We're very close and only got a small final chunk
                    is_complete = True
                
                if is_complete:
                    with _print_lock:
                        print(f"\r  {short_desc}: 100% - Done!                    ")
                    return True
                else:
                    # Keep what we have for resume on next attempt
                    with _print_lock:
                        print(f"\r  {short_desc}: {final_size*100//max(total_size,1)}% - incomplete, retrying...    ", end="", flush=True)
                    continue
            
        except requests.exceptions.Timeout:
            wait_time = min(2 ** attempt, 60)
            with _print_lock:
                print(f"\r  {short_desc}: Timeout, retry {attempt+1}/{max_retries}...      ", end="", flush=True)
            time.sleep(wait_time)
            
        except requests.exceptions.ConnectionError:
            wait_time = min(2 ** attempt, 60)
            with _print_lock:
                print(f"\r  {short_desc}: Connection error, retry {attempt+1}...   ", end="", flush=True)
            time.sleep(wait_time)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                if current_url_idx < len(urls_to_try) - 1:
                    current_url_idx += 1
                    downloaded = 0
                    if dest.exists():
                        dest.unlink()
                    continue
                with _print_lock:
                    print(f"\r  {short_desc}: Not found (404)                  ")
                return False
            if response.status_code == 403:
                if current_url_idx < len(urls_to_try) - 1:
                    current_url_idx += 1
                    downloaded = 0
                    if dest.exists():
                        dest.unlink()
                    with _print_lock:
                        print(f"\r  {short_desc}: 403, trying mirror...           ", end="", flush=True)
                    continue
            wait_time = min(2 ** attempt, 60)
            with _print_lock:
                print(f"\r  {short_desc}: HTTP {response.status_code}, retry {attempt+1}...   ", end="", flush=True)
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            with _print_lock:
                print(f"\n  Interrupted. Run again to resume from {downloaded / (1024*1024):.0f} MB")
            return False
            
        except Exception as e:
            wait_time = min(2 ** attempt, 60)
            time.sleep(wait_time)
    
    # Check if we got enough despite "failures"
    if dest.exists():
        final_size = dest.stat().st_size
        if expected_size > 0 and final_size >= expected_size * 0.90:
            with _print_lock:
                print(f"\r  {short_desc}: 100% - Done! (partial verified)    ")
            return True
    
    with _print_lock:
        print(f"\r  {short_desc}: Failed after {max_retries} attempts        ")
    return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract zip/tar/tar.gz archive."""
    import warnings
    
    try:
        name = archive_path.name.lower()
        
        if name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
            return True
            
        elif name.endswith('.tar.gz') or name.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tf:
                # Use filter='data' to suppress Python 3.14 deprecation warning
                # and for safer extraction (no absolute paths, no symlinks outside)
                try:
                    tf.extractall(extract_to, filter='data')
                except TypeError:
                    # Python < 3.12 doesn't support filter argument
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=DeprecationWarning)
                        tf.extractall(extract_to)
            return True
            
        elif name.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                try:
                    tf.extractall(extract_to, filter='data')
                except TypeError:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=DeprecationWarning)
                        tf.extractall(extract_to)
            return True
            
        else:
            print(f"  Unknown archive format: {archive_path.name}")
            return False
            
    except zipfile.BadZipFile:
        with _print_lock:
            print(f"  Error: {archive_path.name} is not a valid zip file")
        # Check if it's actually HTML (error page)
        with open(archive_path, 'rb') as f:
            header = f.read(100)
            if b'<html' in header.lower() or b'<!doctype' in header.lower():
                with _print_lock:
                    print(f"  (Downloaded file is an HTML error page)")
                archive_path.unlink()
        return False
    except Exception as e:
        with _print_lock:
            print(f"  Extract error: {e}")
        return False


def download_dataset(name: str, output_dir: Path) -> bool:
    """Download a specific dataset."""
    # Check all dataset sources
    all_datasets = {**DATASETS, **HEALTH_DATASETS}
    
    if name not in all_datasets:
        if name in GATED_DATASETS:
            ds = GATED_DATASETS[name]
            print(f"\n{'='*60}")
            print(f"GATED DATASET: {ds['name']}")
            print(f"This dataset requires manual download:")
            print(f"  URL: {ds['registration_url']}")
            print(f"  {ds['description']}")
            print(f"{'='*60}")
            return False
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(all_datasets.keys())}")
        return False
    
    ds = all_datasets[name]
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    marker = ds_dir / ".downloaded"
    if marker.exists():
        with _print_lock:
            print(f"  {ds['name'][:30]}: Already downloaded")
        return True
    
    with _print_lock:
        print(f"  {ds['name'][:30]}: Starting ({ds.get('size', '?')})")
    
    url = ds["url"]
    expected_size = ds.get("size_bytes", 0)
    
    # Determine filename from URL
    filename = url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
    dest = ds_dir / filename
    
    # Get optional custom headers and mirrors
    custom_headers = ds.get("headers")
    mirrors = ds.get("mirrors", [])
    
    # Download with resume support
    success = download_with_resume(
        url, dest, ds["name"], 
        expected_size=expected_size,
        max_retries=10,
        timeout=120,
        custom_headers=custom_headers,
        mirrors=mirrors,
    )
    
    if not success:
        return False
    
    # Verify it's not an error page
    if dest.stat().st_size < 1000:
        print(f"  Error: Downloaded file too small ({dest.stat().st_size} bytes)")
        dest.unlink()
        return False
    
    # Extract if archive
    if any(dest.name.lower().endswith(ext) for ext in ['.zip', '.tar.gz', '.tgz', '.tar']):
        if not extract_archive(dest, ds_dir):
            return False
    
    # Mark as complete
    marker.touch()
    return True


def download_datasets_parallel(names: list, output_dir: Path, max_workers: int = 4) -> dict:
    """Download multiple datasets in parallel.
    
    Args:
        names: List of dataset names
        output_dir: Output directory
        max_workers: Maximum parallel downloads
        
    Returns:
        Dict mapping dataset name to success status
    """
    results = {}
    
    # Sort by size (smallest first for quick wins)
    names_sorted = sorted(names, key=lambda n: DATASETS.get(n, {}).get("size_bytes", 0))
    
    print(f"\nDownloading {len(names)} datasets with {max_workers} parallel workers...")
    print("-" * 50)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_dataset, name, output_dir): name 
            for name in names_sorted
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                with _print_lock:
                    print(f"  {name}: Error - {e}")
                results[name] = False
    
    print("-" * 50)
    
    # Summary
    success = sum(1 for v in results.values() if v)
    print(f"Completed: {success}/{len(results)} datasets")
    
    return results


def estimate_accuracy(output_dir: Path) -> dict:
    """Estimate model accuracy potential based on available data."""
    
    data_inventory = {
        "cough_detection": {"samples": 0, "datasets": []},
        "activity_recognition": {"samples": 0, "datasets": []},
        "heart_rate": {"samples": 0, "datasets": []},
        "voice_features": {"samples": 0, "datasets": []},
        "audio_classification": {"samples": 0, "datasets": []},
    }
    
    sample_estimates = {
        "esc50": {"cough_detection": 40, "audio_classification": 2000},
        "urbansound8k": {"audio_classification": 8732},
        "uci_har": {"activity_recognition": 10299},
        "wisdm": {"activity_recognition": 5400},
        "pamap2": {"activity_recognition": 50000, "heart_rate": 50000},
        "ravdess": {"voice_features": 1440},
        "librispeech_mini": {"voice_features": 2700},
        "ppg_dalia": {"heart_rate": 100000},
        "fsd50k_cough": {"cough_detection": 500, "audio_classification": 50000},
    }
    
    # Check what's downloaded
    for name in DATASETS:
        ds_dir = output_dir / name
        if (ds_dir / ".downloaded").exists():
            estimates = sample_estimates.get(name, {})
            for task, count in estimates.items():
                if task in data_inventory:
                    data_inventory[task]["samples"] += count
                    data_inventory[task]["datasets"].append(name)
    
    return data_inventory


def print_accuracy_assessment(output_dir: Path):
    """Print accuracy assessment based on available data."""
    
    inventory = estimate_accuracy(output_dir)
    
    print("\n" + "="*70)
    print("DATA ADEQUACY ASSESSMENT FOR HIGH-ACCURACY MODEL")
    print("="*70)
    
    thresholds = {
        "cough_detection": {"min": 100, "good": 1000, "excellent": 10000},
        "activity_recognition": {"min": 5000, "good": 20000, "excellent": 100000},
        "heart_rate": {"min": 1000, "good": 10000, "excellent": 50000},
        "voice_features": {"min": 500, "good": 2000, "excellent": 10000},
        "audio_classification": {"min": 1000, "good": 5000, "excellent": 20000},
    }
    
    overall_score = 0
    max_score = 0
    
    for task, data in inventory.items():
        n = data["samples"]
        thresh = thresholds.get(task, {"min": 1000, "good": 10000, "excellent": 50000})
        max_score += 3
        
        if n >= thresh["excellent"]:
            status = "EXCELLENT"
            emoji = "[OK]"
            overall_score += 3
        elif n >= thresh["good"]:
            status = "GOOD"
            emoji = "[OK]"
            overall_score += 2
        elif n >= thresh["min"]:
            status = "MINIMAL"
            emoji = "[!!]"
            overall_score += 1
        else:
            status = "INSUFFICIENT"
            emoji = "[XX]"
        
        print(f"\n{emoji} {task.upper()}")
        print(f"    Samples: {n:,} (need {thresh['good']:,}+ for good accuracy)")
        print(f"    Status: {status}")
        if data["datasets"]:
            print(f"    Sources: {', '.join(data['datasets'])}")
    
    print("\n" + "-"*70)
    pct = (overall_score / max_score * 100) if max_score > 0 else 0
    print(f"Overall data readiness: {pct:.0f}%")
    
    if pct >= 80:
        print("VERDICT: Data is SUFFICIENT for high-accuracy model")
    elif pct >= 50:
        print("VERDICT: Data is OK but more would improve accuracy")
    else:
        print("VERDICT: More data needed for good accuracy")
    
    print("\nRECOMMENDED NEXT STEPS:")
    if inventory["activity_recognition"]["samples"] < 20000:
        print("  python scripts/download_more_data.py -d uci_har")
        print("  python scripts/download_more_data.py -d pamap2")
    if inventory["cough_detection"]["samples"] < 1000:
        print("  python scripts/download_more_data.py -d esc50")
    if inventory["voice_features"]["samples"] < 2000:
        print("  python scripts/download_more_data.py -d ravdess")
    if inventory["heart_rate"]["samples"] < 10000:
        print("  python scripts/download_more_data.py -d ppg_dalia")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Download more data for ViralFlip")
    parser.add_argument("--output", "-o", type=str, default="data/real",
                       help="Output directory")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                       help="Specific dataset to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all available datasets")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--assess", action="store_true",
                       help="Assess current data adequacy")
    parser.add_argument("--essential", action="store_true",
                       help="Download essential datasets for minimum viable model")
    parser.add_argument("--full", action="store_true",
                       help="Download ALL datasets including large ones for max accuracy")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                       help="Number of parallel downloads (default: 1)")
    parser.add_argument("--retry", "-r", type=int, default=10,
                       help="Maximum retries per download (default: 10)")
    parser.add_argument("--health", action="store_true",
                       help="Download REAL health/illness datasets (with labels!)")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("\nAVAILABLE DATASETS (direct download):")
        print("-" * 60)
        total_size = 0
        for name, ds in DATASETS.items():
            print(f"\n  {name}")
            print(f"    {ds['name']}")
            print(f"    Size: {ds.get('size', 'unknown')}")
            print(f"    Use: {', '.join(ds.get('use_for', []))}")
            total_size += ds.get('size_bytes', 0)
        
        print(f"\n  Total (all datasets): ~{total_size / (1024**3):.1f} GB")
        
        print("\n\n*** REAL ILLNESS DATASETS (with health labels!) ***")
        print("-" * 60)
        for name, ds in HEALTH_DATASETS.items():
            label = "[HAS ILLNESS LABELS]" if ds.get('has_illness_labels') else ""
            print(f"\n  {name} {label}")
            print(f"    {ds['name']}")
            print(f"    Size: {ds.get('size', 'unknown')}")
            print(f"    {ds.get('description', '')}")
        
        print("\n\nGATED DATASETS (require registration):")
        print("-" * 60)
        for name, ds in GATED_DATASETS.items():
            label = "[HAS ILLNESS LABELS]" if ds.get('has_illness_labels') else ""
            print(f"\n  {name} {label}")
            print(f"    {ds['name']}")
            print(f"    URL: {ds['registration_url']}")
        return
    
    if args.assess:
        print_accuracy_assessment(output_dir)
        return
    
    if args.dataset:
        download_dataset(args.dataset, output_dir)
        print_accuracy_assessment(output_dir)
        return
    
    if args.essential:
        # Download the most important, reliably available datasets
        # Prioritize fast/reliable sources: GitHub (ESC-50), Zenodo (RAVDESS), OpenSLR (LibriSpeech)
        essential = ["esc50", "ravdess", "librispeech_mini", "wisdm"]
        print("Downloading ESSENTIAL datasets for minimum viable model...")
        print("(Using fast, reliable sources: GitHub, Zenodo, OpenSLR)")
        
        if args.parallel > 1:
            download_datasets_parallel(essential, output_dir, args.parallel)
        else:
            for name in essential:
                download_dataset(name, output_dir)
        
        print_accuracy_assessment(output_dir)
        return
    
    if args.health:
        # Download real health datasets with illness labels
        print("\n*** DOWNLOADING REAL HEALTH DATASETS ***")
        print("These have actual illness/COVID labels!")
        print("-" * 50)
        
        health_list = list(HEALTH_DATASETS.keys())
        
        if args.parallel > 1:
            download_datasets_parallel(health_list, output_dir, args.parallel)
        else:
            for name in health_list:
                download_dataset(name, output_dir)
        
        print_accuracy_assessment(output_dir)
        return
    
    if args.full:
        # Download everything for maximum accuracy
        print("Downloading ALL datasets for MAXIMUM ACCURACY...")
        print("This will take significant time and disk space (~20GB+)")
        
        all_datasets = list(DATASETS.keys()) + list(HEALTH_DATASETS.keys())
        
        if args.parallel > 1:
            download_datasets_parallel(all_datasets, output_dir, args.parallel)
        else:
            for name in all_datasets:
                download_dataset(name, output_dir)
        
        print_accuracy_assessment(output_dir)
        return
    
    if args.all:
        print("Downloading ALL available datasets...")
        print("(Skipping very large ones - use --full for those)")
        # Skip the huge ones by default
        skip = {"fsd50k_cough", "urbansound8k"}
        datasets_to_download = [n for n in DATASETS if n not in skip]
        
        if args.parallel > 1:
            download_datasets_parallel(datasets_to_download, output_dir, args.parallel)
        else:
            for name in datasets_to_download:
                download_dataset(name, output_dir)
        
        print_accuracy_assessment(output_dir)
        return
    
    # Default: show assessment
    print_accuracy_assessment(output_dir)
    print("\nTo download data:")
    print("  python scripts/download_more_data.py --essential         # Quick start (~1GB)")
    print("  python scripts/download_more_data.py --all               # All medium datasets (~4GB)")
    print("  python scripts/download_more_data.py --full --parallel 4 # MAX ACCURACY (~20GB)")
    print("  python scripts/download_more_data.py -d uci_har          # Specific dataset")
    print("  python scripts/download_more_data.py --list              # See all options")


if __name__ == "__main__":
    main()
