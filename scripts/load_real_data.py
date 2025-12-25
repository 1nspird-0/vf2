#!/usr/bin/env python3
"""Load real data for ViralFlip training.

This script provides functions to load the downloaded real datasets
and convert them into formats usable by ViralFlip.

Usage:
    # As a script - extract and prepare all data:
    python scripts/load_real_data.py --data-dir data/real
    
    # As a module:
    from scripts.load_real_data import RealDataLoader
    loader = RealDataLoader("data/real")
    cough_data = loader.load_cough_audio()
    imu_data = loader.load_imu_data()
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class RealDataLoader:
    """Loader for real datasets downloaded by gather_real_data.py."""
    
    def __init__(self, data_dir: str = "data/real"):
        self.data_dir = Path(data_dir)
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> dict:
        """Load the data manifest."""
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {"datasets": []}
    
    def available_datasets(self) -> List[str]:
        """List available datasets."""
        return [d["dataset"] for d in self.manifest.get("datasets", [])]
    
    def summary(self) -> None:
        """Print summary of available data."""
        print("\n" + "="*60)
        print("AVAILABLE REAL DATA")
        print("="*60)
        
        for ds in self.manifest.get("datasets", []):
            name = ds.get("name", ds.get("dataset", "Unknown"))
            status = ds.get("download", "unknown")
            processed = ds.get("processed", {})
            
            print(f"\n{name}")
            print(f"  Download: {status}")
            if processed:
                print(f"  Processed: {processed.get('status', 'no')}")
                if "use_for" in processed:
                    print(f"  Use for: {', '.join(processed['use_for'])}")
        
        print("\n" + "="*60)
    
    # =========================================================================
    # ESC-50 / Cough Audio
    # =========================================================================
    
    def extract_esc50(self) -> Path:
        """Extract ESC-50 dataset if needed."""
        esc50_dir = self.data_dir / "esc50"
        zip_path = esc50_dir / "esc50.zip"
        extract_dir = esc50_dir / "ESC-50-master"
        
        if not extract_dir.exists() and zip_path.exists():
            print("Extracting ESC-50...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(esc50_dir)
        
        return extract_dir
    
    def load_cough_audio(self, as_features: bool = True) -> Dict[str, Any]:
        """Load cough audio data from ESC-50.
        
        Args:
            as_features: If True, extract mel-spectrogram features.
                        If False, return raw audio paths.
        
        Returns:
            Dictionary with 'cough' and 'non_cough' samples
        """
        esc50_dir = self.extract_esc50()
        
        if not esc50_dir.exists():
            print("ESC-50 not found. Run: python scripts/gather_real_data.py --datasets esc50")
            return {}
        
        audio_dir = esc50_dir / "audio"
        meta_path = esc50_dir / "meta" / "esc50.csv"
        
        if not HAS_PANDAS:
            print("pandas required for ESC-50 loading")
            return {}
        
        df = pd.read_csv(meta_path)
        
        # Category 24 is 'coughing' in ESC-50
        cough_files = df[df['target'] == 24]['filename'].tolist()
        non_cough_files = df[df['target'] != 24]['filename'].tolist()
        
        result = {
            "cough_files": [str(audio_dir / f) for f in cough_files],
            "non_cough_files": [str(audio_dir / f) for f in non_cough_files],
            "n_cough": len(cough_files),
            "n_non_cough": len(non_cough_files),
            "categories": df.groupby('target')['category'].first().to_dict(),
        }
        
        if as_features and HAS_LIBROSA:
            print("Extracting audio features...")
            result["cough_features"] = self._extract_audio_features(result["cough_files"][:40])
            # Sample equal number of non-cough for balanced training
            np.random.seed(42)
            sampled_non_cough = np.random.choice(
                result["non_cough_files"], 
                size=min(200, len(result["non_cough_files"])),
                replace=False
            )
            result["non_cough_features"] = self._extract_audio_features(list(sampled_non_cough))
        
        return result
    
    def _extract_audio_features(self, audio_paths: List[str], sr: int = 16000) -> np.ndarray:
        """Extract mel-spectrogram features from audio files."""
        if not HAS_LIBROSA:
            return np.array([])
        
        features = []
        for path in audio_paths:
            try:
                y, _ = librosa.load(path, sr=sr, duration=5.0)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                # Normalize and flatten
                features.append(mel_db.mean(axis=1))  # Average over time
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        return np.array(features)
    
    # =========================================================================
    # UCI HAR / IMU Data
    # =========================================================================
    
    def extract_uci_har(self) -> Path:
        """Extract UCI HAR dataset if needed."""
        uci_dir = self.data_dir / "uci_har"
        zip_path = uci_dir / "UCI HAR Dataset.zip"
        extract_dir = uci_dir / "UCI HAR Dataset"
        
        if not extract_dir.exists() and zip_path.exists():
            print("Extracting UCI HAR...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(uci_dir)
        
        return extract_dir
    
    def load_imu_data(self, split: str = "train") -> Dict[str, Any]:
        """Load IMU data from UCI HAR.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with accelerometer data, labels, etc.
        """
        uci_dir = self.extract_uci_har()
        
        if not uci_dir.exists():
            print("UCI HAR not found. Run: python scripts/gather_real_data.py --datasets uci_har")
            return {}
        
        split_dir = uci_dir / split
        if not split_dir.exists():
            print(f"Split '{split}' not found in UCI HAR")
            return {}
        
        # Load features (already extracted by dataset creators)
        X_path = split_dir / f"X_{split}.txt"
        y_path = split_dir / f"y_{split}.txt"
        subject_path = split_dir / f"subject_{split}.txt"
        
        X = np.loadtxt(X_path)
        y = np.loadtxt(y_path, dtype=int)
        subjects = np.loadtxt(subject_path, dtype=int)
        
        # Activity labels
        activity_labels = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS", 
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING",
        }
        
        # Load raw inertial signals if available
        inertial_dir = split_dir / "Inertial Signals"
        raw_data = {}
        if inertial_dir.exists():
            signal_files = [
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z",
            ]
            for signal in signal_files:
                signal_path = inertial_dir / f"{signal}_{split}.txt"
                if signal_path.exists():
                    raw_data[signal] = np.loadtxt(signal_path)
        
        return {
            "X": X,  # (n_samples, 561) features
            "y": y,  # (n_samples,) activity labels
            "subjects": subjects,  # (n_samples,) subject IDs
            "activity_labels": activity_labels,
            "raw_inertial": raw_data,
            "n_samples": len(y),
            "n_subjects": len(np.unique(subjects)),
            "n_features": X.shape[1],
        }
    
    # =========================================================================
    # WISDM
    # =========================================================================
    
    def extract_wisdm(self) -> Path:
        """Extract WISDM dataset if needed."""
        wisdm_dir = self.data_dir / "wisdm"
        zip_path = wisdm_dir / "wisdm.zip"
        
        # Check for extracted directory
        for subdir in wisdm_dir.iterdir():
            if subdir.is_dir() and "wisdm" in subdir.name.lower():
                return subdir
        
        if zip_path.exists():
            print("Extracting WISDM...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(wisdm_dir)
            
            # Find extracted dir
            for subdir in wisdm_dir.iterdir():
                if subdir.is_dir():
                    return subdir
        
        return wisdm_dir
    
    def load_wisdm_data(self) -> Dict[str, Any]:
        """Load WISDM activity recognition data."""
        wisdm_dir = self.extract_wisdm()
        
        # Look for phone accelerometer data
        phone_accel = wisdm_dir / "phone" / "accel"
        watch_accel = wisdm_dir / "watch" / "accel"
        
        result = {"source": "wisdm", "data": {}}
        
        for source, path in [("phone_accel", phone_accel), ("watch_accel", watch_accel)]:
            if path.exists():
                files = list(path.glob("*.txt"))
                result["data"][source] = {
                    "n_files": len(files),
                    "files": [str(f) for f in files[:5]],  # Sample paths
                }
        
        return result
    
    # =========================================================================
    # Beiwe Phone Sensing
    # =========================================================================
    
    def load_beiwe_data(self) -> Dict[str, Any]:
        """Load Beiwe sample phone sensing data."""
        beiwe_dir = self.data_dir / "beiwe"
        
        if not beiwe_dir.exists():
            return {}
        
        result = {"source": "beiwe", "streams": {}}
        
        for stream_dir in beiwe_dir.iterdir():
            if stream_dir.is_dir():
                files = list(stream_dir.glob("*.csv"))
                if files:
                    result["streams"][stream_dir.name] = {
                        "n_files": len(files),
                        "sample_file": str(files[0]) if files else None,
                    }
        
        return result
    
    # =========================================================================
    # Combined Training Data
    # =========================================================================
    
    def create_training_batch(
        self,
        include_cough: bool = True,
        include_imu: bool = True,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Create a combined training batch from all available data.
        
        This creates a simplified format that can be used for training
        individual components of ViralFlip.
        """
        batch = {
            "cough_detector": None,
            "imu_features": None,
            "phone_sensing": None,
        }
        
        if include_cough:
            cough_data = self.load_cough_audio(as_features=False)
            if cough_data:
                batch["cough_detector"] = {
                    "positive_files": cough_data.get("cough_files", []),
                    "negative_files": cough_data.get("non_cough_files", [])[:200],
                    "ready": True,
                }
        
        if include_imu:
            imu_data = self.load_imu_data("train")
            if imu_data:
                n = min(max_samples, imu_data["n_samples"])
                batch["imu_features"] = {
                    "X": imu_data["X"][:n],
                    "y": imu_data["y"][:n],
                    "subjects": imu_data["subjects"][:n],
                    "activity_labels": imu_data["activity_labels"],
                    "ready": True,
                }
        
        beiwe_data = self.load_beiwe_data()
        if beiwe_data.get("streams"):
            batch["phone_sensing"] = {
                "streams": list(beiwe_data["streams"].keys()),
                "ready": True,
            }
        
        return batch


# =============================================================================
# PyTorch Datasets
# =============================================================================

if HAS_TORCH:
    class CoughAudioDataset(Dataset):
        """PyTorch Dataset for cough audio classification."""
        
        def __init__(
            self,
            cough_files: List[str],
            non_cough_files: List[str],
            transform=None,
            sr: int = 16000,
        ):
            self.files = [(f, 1) for f in cough_files] + [(f, 0) for f in non_cough_files]
            self.transform = transform
            self.sr = sr
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            path, label = self.files[idx]
            
            if HAS_LIBROSA:
                y, _ = librosa.load(path, sr=self.sr, duration=5.0)
                # Pad or truncate to fixed length
                target_len = 5 * self.sr
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]
                
                if self.transform:
                    y = self.transform(y)
                
                return torch.FloatTensor(y), torch.LongTensor([label])
            else:
                # Return placeholder
                return torch.zeros(5 * self.sr), torch.LongTensor([label])
    
    
    class IMUDataset(Dataset):
        """PyTorch Dataset for IMU/HAR data."""
        
        def __init__(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y) - 1  # 0-indexed
            self.subjects = subjects
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load real data for ViralFlip")
    parser.add_argument("--data-dir", type=str, default="data/real",
                       help="Directory containing downloaded data")
    parser.add_argument("--extract", action="store_true",
                       help="Extract all compressed datasets")
    parser.add_argument("--summary", action="store_true",
                       help="Print data summary")
    parser.add_argument("--test-load", action="store_true",
                       help="Test loading all datasets")
    
    args = parser.parse_args()
    
    loader = RealDataLoader(args.data_dir)
    
    if args.summary or not any([args.extract, args.test_load]):
        loader.summary()
    
    if args.extract:
        print("\nExtracting datasets...")
        loader.extract_esc50()
        loader.extract_uci_har()
        loader.extract_wisdm()
        print("Done!")
    
    if args.test_load:
        print("\n" + "="*60)
        print("TESTING DATA LOADING")
        print("="*60)
        
        # Test cough audio
        print("\n[1] Loading cough audio...")
        cough_data = loader.load_cough_audio(as_features=False)
        if cough_data:
            print(f"    Cough samples: {cough_data.get('n_cough', 0)}")
            print(f"    Non-cough samples: {cough_data.get('n_non_cough', 0)}")
        else:
            print("    Not available")
        
        # Test IMU
        print("\n[2] Loading IMU data...")
        imu_data = loader.load_imu_data("train")
        if imu_data:
            print(f"    Samples: {imu_data.get('n_samples', 0)}")
            print(f"    Subjects: {imu_data.get('n_subjects', 0)}")
            print(f"    Features: {imu_data.get('n_features', 0)}")
            print(f"    Activities: {list(imu_data.get('activity_labels', {}).values())}")
        else:
            print("    Not available")
        
        # Test phone sensing
        print("\n[3] Loading phone sensing data...")
        beiwe_data = loader.load_beiwe_data()
        if beiwe_data.get("streams"):
            print(f"    Streams: {list(beiwe_data['streams'].keys())}")
        else:
            print("    Not available")
        
        # Create batch
        print("\n[4] Creating training batch...")
        batch = loader.create_training_batch()
        for key, val in batch.items():
            if val and val.get("ready"):
                print(f"    {key}: READY")
            else:
                print(f"    {key}: Not available")
        
        print("\n" + "="*60)
        print("LOADING COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()

