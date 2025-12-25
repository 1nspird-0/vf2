#!/usr/bin/env python3
"""Prepare real health datasets for ViralFlip training.

This script:
1. Discovers downloaded health datasets
2. Extracts audio features (MFCCs, spectral features)
3. Creates train/val/test splits
4. Saves processed data in training-ready format

Usage:
    python scripts/prepare_real_data.py --data-dir data/ --output data/processed/
    python scripts/prepare_real_data.py --parallel 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from viralflip.model.virus_types import VirusType, VIRUS_NAMES, NUM_VIRUS_CLASSES


def check_dependencies():
    """Check required dependencies."""
    missing = []
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    
    if missing:
        print(f"Installing missing dependencies: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["-q"])


# Label mapping to VirusType
LABEL_MAP = {
    # COVID
    "covid": VirusType.COVID,
    "covid-19": VirusType.COVID,
    "covid_positive": VirusType.COVID,
    "positive": VirusType.COVID,
    "sars-cov-2": VirusType.COVID,
    "symptomatic_covid": VirusType.COVID,
    
    # Flu
    "flu": VirusType.FLU,
    "influenza": VirusType.FLU,
    "ili": VirusType.FLU,
    
    # Cold
    "cold": VirusType.COLD,
    "common_cold": VirusType.COLD,
    
    # RSV
    "rsv": VirusType.RSV,
    
    # Pneumonia
    "pneumonia": VirusType.PNEUMONIA,
    
    # General respiratory
    "respiratory": VirusType.GENERAL,
    "symptomatic": VirusType.GENERAL,
    "sick": VirusType.GENERAL,
    "other": VirusType.GENERAL,
    
    # Healthy
    "healthy": VirusType.HEALTHY,
    "negative": VirusType.HEALTHY,
    "covid_negative": VirusType.HEALTHY,
    "control": VirusType.HEALTHY,
    "asymptomatic": VirusType.HEALTHY,
    "no_resp_illness_exposed": VirusType.HEALTHY,
}


def extract_audio_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> Optional[np.ndarray]:
    """Extract audio features from a file.
    
    Returns a 30-dimensional feature vector:
    - 13 MFCC means
    - 13 MFCC stds  
    - 4 spectral features (centroid, rolloff, zcr, rms)
    """
    try:
        import librosa
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr, duration=10.0)
            
            if len(y) < sr * 0.3:  # Less than 0.3 seconds
                return None
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Normalize
            spectral_centroid = spectral_centroid / 10000  # Scale to ~[0,1]
            spectral_rolloff = spectral_rolloff / 10000
            
            # Combine
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, zero_crossing, rms],
            ])
            
            return features.astype(np.float32)
            
    except Exception as e:
        return None


def find_dataset_dir(data_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Find a dataset directory matching any of the patterns."""
    for pattern in patterns:
        # Check direct subdirectory
        matches = list(data_dir.glob(pattern))
        if matches:
            # If it's a directory, return it
            for m in matches:
                if m.is_dir():
                    return m
            # If it's a file, return parent
            return matches[0].parent if matches else None
        
        # Check nested (after extraction)
        matches = list(data_dir.glob(f"**/{pattern}"))
        if matches:
            for m in matches:
                if m.is_dir():
                    return m
    return None


def load_coughvid(data_dir: Path) -> List[Dict]:
    """Load COUGHVID dataset."""
    samples = []
    
    # Find COUGHVID directory - check multiple patterns
    patterns = ["coughvid*", "*oughvid*", "*ublic_dataset*", "COUGHVID*"]
    coughvid_path = find_dataset_dir(data_dir, patterns)
    
    if not coughvid_path:
        print("  COUGHVID: Not found")
        return samples
    
    print(f"  Loading COUGHVID from {coughvid_path}")
    
    # Find metadata - search deeply
    metadata_files = list(coughvid_path.glob("**/metadata*.csv"))
    if not metadata_files:
        # Try without metadata - just find audio files with covid in path
        audio_files = list(coughvid_path.glob("**/*.wav")) + list(coughvid_path.glob("**/*.webm"))
        for af in audio_files[:1000]:  # Limit for speed
            # Infer label from path
            path_str = str(af).lower()
            if "positive" in path_str or "covid" in path_str:
                virus_type = VirusType.COVID
            elif "negative" in path_str or "healthy" in path_str:
                virus_type = VirusType.HEALTHY
            else:
                virus_type = VirusType.GENERAL
            
            samples.append({
                "sample_id": af.stem,
                "dataset": "coughvid",
                "virus_type": virus_type.value,
                "audio_path": str(af),
                "quality": 0.8,
            })
        return samples
    
    try:
        import pandas as pd
        df = pd.read_csv(metadata_files[0])
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="COUGHVID"):
            # Get status
            status = str(row.get("status", row.get("covid_status", "unknown"))).lower().strip()
            virus_type = LABEL_MAP.get(status, VirusType.GENERAL)
            
            # Find audio file
            uuid = str(row.get("uuid", row.get("id", "")))
            if not uuid:
                continue
            
            audio_path = None
            for ext in [".webm", ".ogg", ".wav", ".mp3"]:
                candidate = coughvid_path / f"{uuid}{ext}"
                if candidate.exists():
                    audio_path = str(candidate)
                    break
            
            if not audio_path:
                # Search in subdirs
                matches = list(coughvid_path.glob(f"**/{uuid}.*"))
                if matches:
                    audio_path = str(matches[0])
            
            quality = float(row.get("cough_detected", row.get("quality", 0.5)))
            
            samples.append({
                "sample_id": uuid,
                "dataset": "coughvid",
                "virus_type": virus_type.value,
                "audio_path": audio_path,
                "quality": quality,
            })
    except Exception as e:
        print(f"  Error loading COUGHVID: {e}")
    
    return samples


def load_coswara(data_dir: Path) -> List[Dict]:
    """Load Coswara dataset."""
    samples = []
    
    # Find Coswara - check inside the coswara download folder
    coswara_path = find_dataset_dir(data_dir, ["coswara*", "*oswara*", "Coswara*"])
    if not coswara_path:
        print("  Coswara: Not found")
        return samples
    
    print(f"  Loading Coswara from {coswara_path}")
    
    # Coswara GitHub structure: Coswara-Data-master/Extracted_data/YYYYMMDD/userid/
    extracted = list(coswara_path.glob("**/Extracted_data"))
    if extracted:
        base = extracted[0]
        for date_folder in base.iterdir():
            if not date_folder.is_dir():
                continue
            for user_folder in date_folder.iterdir():
                if not user_folder.is_dir():
                    continue
                
                # Read metadata.json for COVID status
                status = "unknown"
                meta = user_folder / "metadata.json"
                if meta.exists():
                    try:
                        with open(meta) as f:
                            m = json.load(f)
                        status = str(m.get("covid_status", m.get("status", ""))).lower()
                    except:
                        pass
                
                virus_type = LABEL_MAP.get(status, VirusType.GENERAL)
                
                # Find audio files
                for audio in user_folder.glob("*.wav"):
                    samples.append({
                        "sample_id": f"{user_folder.name}_{audio.stem}",
                        "dataset": "coswara",
                        "virus_type": virus_type.value,
                        "audio_path": str(audio),
                        "quality": 0.8,
                    })
    
    # Fallback: find any audio files
    if not samples:
        all_wav = list(coswara_path.glob("**/*.wav"))[:3000]
        for af in all_wav:
            samples.append({
                "sample_id": af.stem,
                "dataset": "coswara",
                "virus_type": VirusType.GENERAL.value,
                "audio_path": str(af),
                "quality": 0.7,
            })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_virufy(data_dir: Path) -> List[Dict]:
    """Load Virufy dataset."""
    samples = []
    
    virufy_path = find_dataset_dir(data_dir, ["virufy*", "*irufy*", "Virufy*"])
    if not virufy_path:
        print("  Virufy: Not found")
        return samples
    
    print(f"  Loading Virufy from {virufy_path}")
    
    # Find all audio files
    audio_files = list(virufy_path.glob("**/*.wav")) + list(virufy_path.glob("**/*.ogg"))
    
    for audio_file in audio_files:
        # Determine label from path
        path_str = str(audio_file).lower()
        
        if "positive" in path_str or "covid" in path_str:
            virus_type = VirusType.COVID
        elif "negative" in path_str or "healthy" in path_str:
            virus_type = VirusType.HEALTHY
        else:
            virus_type = VirusType.GENERAL
        
        samples.append({
            "sample_id": audio_file.stem,
            "dataset": "virufy",
            "virus_type": virus_type.value,
            "audio_path": str(audio_file),
            "quality": 1.0,
        })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_dicova(data_dir: Path) -> List[Dict]:
    """Load DiCOVA dataset."""
    samples = []
    
    dicova_path = find_dataset_dir(data_dir, ["dicova*", "*icova*", "DiCOVA*"])
    if not dicova_path:
        print("  DiCOVA: Not found")
        return samples
    
    print(f"  Loading DiCOVA from {dicova_path}")
    
    # Look for label/metadata files
    csv_files = list(dicova_path.glob("**/*.csv"))
    
    if csv_files:
        try:
            import pandas as pd
            for cf in csv_files[:3]:
                df = pd.read_csv(cf)
                for _, row in df.iterrows():
                    label = str(row.get("label", row.get("covid_status", row.get("status", "unknown")))).lower().strip()
                    virus_type = LABEL_MAP.get(label, VirusType.GENERAL)
                    
                    samples.append({
                        "sample_id": str(row.get("file_name", row.get("id", row.name))),
                        "dataset": "dicova",
                        "virus_type": virus_type.value,
                        "audio_path": None,
                        "quality": 0.9,
                    })
        except Exception as e:
            print(f"  Error: {e}")
    
    # Also find any audio files
    if not samples:
        audio_files = list(dicova_path.glob("**/*.wav"))[:500]
        for af in audio_files:
            path_str = str(af).lower()
            if "positive" in path_str or "p_" in path_str:
                virus_type = VirusType.COVID
            elif "negative" in path_str or "n_" in path_str:
                virus_type = VirusType.HEALTHY
            else:
                virus_type = VirusType.GENERAL
            
            samples.append({
                "sample_id": af.stem,
                "dataset": "dicova",
                "virus_type": virus_type.value,
                "audio_path": str(af),
                "quality": 0.9,
            })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_flusense(data_dir: Path) -> List[Dict]:
    """Load FluSense dataset."""
    samples = []
    
    flusense_path = find_dataset_dir(data_dir, ["flusense*", "*lusense*", "FluSense*"])
    if not flusense_path:
        print("  FluSense: Not found")
        return samples
    
    print(f"  Loading FluSense from {flusense_path}")
    
    # FluSense has audio segments with labels
    # Look for any annotation/label files
    csv_files = list(flusense_path.glob("**/*.csv"))
    json_files = list(flusense_path.glob("**/*.json"))
    
    # Try CSV first
    if csv_files:
        try:
            import pandas as pd
            for cf in csv_files[:3]:
                df = pd.read_csv(cf)
                for _, row in df.iterrows():
                    flu = row.get("flu", row.get("label", row.get("positive", 0)))
                    virus_type = VirusType.FLU if flu else VirusType.HEALTHY
                    samples.append({
                        "sample_id": str(row.get("id", row.name)),
                        "dataset": "flusense",
                        "virus_type": virus_type.value,
                        "audio_path": None,
                        "quality": 0.7,
                    })
        except:
            pass
    
    # Try JSON
    if not samples and json_files:
        for jf in json_files[:5]:
            try:
                with open(jf) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        flu = item.get("flu_positive", item.get("flu", item.get("label", False)))
                        virus_type = VirusType.FLU if flu else VirusType.HEALTHY
                        samples.append({
                            "sample_id": str(item.get("id", len(samples))),
                            "dataset": "flusense",
                            "virus_type": virus_type.value,
                            "audio_path": None,
                            "quality": 0.7,
                        })
            except:
                continue
    
    # Fallback: find audio files
    if not samples:
        audio_files = list(flusense_path.glob("**/*.wav"))[:1000]
        for af in audio_files:
            samples.append({
                "sample_id": af.stem,
                "dataset": "flusense",
                "virus_type": VirusType.GENERAL.value,
                "audio_path": str(af),
                "quality": 0.6,
            })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_wesad(data_dir: Path) -> List[Dict]:
    """Load WESAD physiological dataset (for baselines)."""
    samples = []
    
    wesad_path = find_dataset_dir(data_dir, ["wesad*", "WESAD*"])
    if not wesad_path:
        print("  WESAD: Not found")
        return samples
    
    print(f"  Loading WESAD from {wesad_path}")
    
    # WESAD has subject folders S2, S3, etc with pickle files
    for subj_dir in wesad_path.glob("**/S*"):
        if not subj_dir.is_dir():
            continue
        
        pkl_file = subj_dir / f"{subj_dir.name}.pkl"
        if pkl_file.exists():
            # Use as healthy baseline data
            samples.append({
                "sample_id": subj_dir.name,
                "dataset": "wesad",
                "virus_type": VirusType.HEALTHY.value,
                "audio_path": None,  # No audio, but has physio
                "quality": 1.0,
            })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_icbhi(data_dir: Path) -> List[Dict]:
    """Load ICBHI respiratory sounds dataset."""
    samples = []
    
    icbhi_path = find_dataset_dir(data_dir, ["icbhi*", "ICBHI*", "*espiratory*"])
    if not icbhi_path:
        print("  ICBHI: Not found")
        return samples
    
    print(f"  Loading ICBHI from {icbhi_path}")
    
    # ICBHI has audio files with diagnosis info in filename or txt files
    audio_files = list(icbhi_path.glob("**/*.wav"))
    
    for af in audio_files:
        # Check for accompanying txt file with diagnosis
        txt_file = af.with_suffix(".txt")
        
        virus_type = VirusType.GENERAL  # Default to respiratory
        
        # Try to get diagnosis from filename or txt
        path_str = str(af).lower()
        if "healthy" in path_str or "normal" in path_str:
            virus_type = VirusType.HEALTHY
        elif "pneumonia" in path_str:
            virus_type = VirusType.PNEUMONIA
        elif "copd" in path_str or "asthma" in path_str:
            virus_type = VirusType.GENERAL
        
        if txt_file.exists():
            try:
                with open(txt_file) as f:
                    content = f.read().lower()
                if "pneumonia" in content:
                    virus_type = VirusType.PNEUMONIA
                elif "healthy" in content or "normal" in content:
                    virus_type = VirusType.HEALTHY
            except:
                pass
        
        samples.append({
            "sample_id": af.stem,
            "dataset": "icbhi",
            "virus_type": virus_type.value,
            "audio_path": str(af),
            "quality": 0.9,
        })
    
    print(f"    Found {len(samples)} samples")
    return samples


def load_mimic_demo(data_dir: Path) -> List[Dict]:
    """Load MIMIC-III demo dataset."""
    samples = []
    
    mimic_path = find_dataset_dir(data_dir, ["mimic*", "MIMIC*"])
    if not mimic_path:
        print("  MIMIC: Not found")
        return samples
    
    print(f"  Loading MIMIC from {mimic_path}")
    
    # MIMIC has CSV files with diagnoses
    csv_files = list(mimic_path.glob("**/*.csv"))
    
    for cf in csv_files[:5]:  # Check first few
        try:
            import pandas as pd
            df = pd.read_csv(cf)
            
            # Look for diagnosis columns
            for _, row in df.iterrows():
                diag = str(row.get("diagnosis", row.get("icd9_code", ""))).lower()
                
                if "pneumonia" in diag:
                    virus_type = VirusType.PNEUMONIA
                elif "influenza" in diag or "flu" in diag:
                    virus_type = VirusType.FLU
                elif "sepsis" in diag or "infection" in diag:
                    virus_type = VirusType.GENERAL
                else:
                    continue  # Skip non-respiratory
                
                samples.append({
                    "sample_id": str(row.get("subject_id", row.get("hadm_id", len(samples)))),
                    "dataset": "mimic",
                    "virus_type": virus_type.value,
                    "audio_path": None,
                    "quality": 0.9,
                })
        except Exception as e:
            continue
    
    print(f"    Found {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare real health data for training")
    parser.add_argument("--data-dir", "-d", type=str, default="data/",
                       help="Directory containing downloaded datasets")
    parser.add_argument("--output", "-o", type=str, default="data/processed/",
                       help="Output directory for processed data")
    parser.add_argument("--parallel", "-p", type=int, default=4,
                       help="Number of parallel workers for feature extraction")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip audio feature extraction")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (for testing)")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ViralFlip Real Data Preparation")
    print(f"{'='*60}\n")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.parallel}")
    print()
    
    # Show what folders exist and find the actual data location
    print("Checking data directory contents...")
    
    # Check multiple possible locations
    possible_dirs = [
        data_dir,
        data_dir / "real",
        data_dir / "raw",
        data_dir / "downloads",
    ]
    
    actual_data_dir = None
    for d in possible_dirs:
        if d.exists():
            subdirs = [sd.name for sd in d.iterdir() if sd.is_dir()]
            print(f"  {d}: {subdirs[:10]}")
            # Check if this looks like our data
            if any(name in str(subdirs).lower() for name in ["cough", "flu", "coswara", "virufy", "wesad"]):
                actual_data_dir = d
                print(f"  -> Using {d} as data source")
    
    if actual_data_dir:
        data_dir = actual_data_dir
    elif not data_dir.exists():
        print(f"  WARNING: {data_dir} does not exist!")
    print()
    
    # Load all datasets
    all_samples = []
    
    print("Loading datasets...")
    loaders = [
        ("COUGHVID", load_coughvid),
        ("Coswara", load_coswara),
        ("Virufy", load_virufy),
        ("DiCOVA", load_dicova),
        ("FluSense", load_flusense),
        ("WESAD", load_wesad),
        ("ICBHI", load_icbhi),
        ("MIMIC", load_mimic_demo),
    ]
    
    for name, loader in loaders:
        try:
            samples = loader(data_dir)
            if args.max_samples:
                samples = samples[:args.max_samples]
            all_samples.extend(samples)
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    if not all_samples:
        print("\nNo samples found!")
        print("Please download health datasets first:")
        print("  python scripts/download_more_data.py --health --parallel 4")
        return
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Extract audio features
    if not args.skip_features:
        print("\nExtracting audio features...")
        
        samples_with_audio = [s for s in all_samples if s.get("audio_path")]
        print(f"  Samples with audio: {len(samples_with_audio)}")
        
        if samples_with_audio:
            def process_sample(sample):
                audio_path = sample.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    features = extract_audio_features(audio_path)
                    if features is not None:
                        sample["audio_features"] = features.tolist()
                return sample
            
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(process_sample, s) for s in samples_with_audio]
                
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Features")):
                    try:
                        result = future.result()
                    except:
                        pass
            
            # Count successful extractions
            n_with_features = sum(1 for s in all_samples if s.get("audio_features"))
            print(f"  Successfully extracted: {n_with_features}")
    
    # Filter samples with features or valid labels
    valid_samples = [s for s in all_samples if s.get("audio_features") or s.get("quality", 0) > 0.5]
    print(f"\nValid samples: {len(valid_samples)}")
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(valid_samples)
    
    n = len(valid_samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]
    
    # Count virus types
    virus_counts = {v.name: 0 for v in VirusType}
    for s in valid_samples:
        vt = VirusType(s.get("virus_type", 0))
        virus_counts[vt.name] += 1
    
    print(f"\nVirus type distribution:")
    for name, count in sorted(virus_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / len(valid_samples)
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Save splits
    print(f"\nSaving processed data...")
    
    # Validate we have data
    if len(train_samples) == 0:
        print("  ERROR: No training samples! Check dataset loading above.")
        return
    
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=None)  # Compact but valid
        f.write("\n")  # Add newline for wc -l
    file_size = (output_dir / "train.json").stat().st_size
    print(f"  Train: {len(train_samples)} samples ({file_size / 1024:.1f} KB)")
    
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f, indent=None)
        f.write("\n")
    file_size = (output_dir / "val.json").stat().st_size
    print(f"  Val: {len(val_samples)} samples ({file_size / 1024:.1f} KB)")
    
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f, indent=None)
        f.write("\n")
    file_size = (output_dir / "test.json").stat().st_size
    print(f"  Test: {len(test_samples)} samples ({file_size / 1024:.1f} KB)")
    
    # Save metadata
    metadata = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "virus_types": [v.name for v in VirusType],
        "virus_counts": virus_counts,
        "datasets_used": list(set(s["dataset"] for s in valid_samples)),
        "feature_dim": 30,
    }
    
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    
    print(f"\n{'='*60}")
    print("Done! Data saved to:", output_dir)
    print(f"{'='*60}")
    print("\nNext step - train the model:")
    print("  python scripts/train.py --config configs/high_performance.yaml")


if __name__ == "__main__":
    main()

