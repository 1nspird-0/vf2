"""Real data loader for health datasets with illness labels.

Loads and processes downloaded datasets:
- COUGHVID: COVID cough recordings
- Coswara: Breathing/cough/speech with COVID labels  
- Virufy: PCR-confirmed COVID coughs
- DiCOVA: Respiratory illness coughs
- FluSense: Hospital waiting room flu detection
- Sound-Dr: Pneumonia/COVID coughs
- WESAD: Physiological signals (stress/affect)
- And more...

Extracts features and maps to ViralFlip's expected format.
"""

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml

from viralflip.model.virus_types import VirusType, VIRUS_NAMES
from viralflip.utils.logging import get_logger

logger = get_logger(__name__)


# Map dataset illness labels to VirusType
LABEL_TO_VIRUS_TYPE = {
    # COVID variants
    "covid": VirusType.COVID,
    "covid-19": VirusType.COVID,
    "covid_positive": VirusType.COVID,
    "positive": VirusType.COVID,
    "sars-cov-2": VirusType.COVID,
    
    # Flu variants  
    "flu": VirusType.FLU,
    "influenza": VirusType.FLU,
    "influenza_a": VirusType.FLU,
    "influenza_b": VirusType.FLU,
    "ili": VirusType.FLU,  # Influenza-like illness
    
    # Cold variants
    "cold": VirusType.COLD,
    "common_cold": VirusType.COLD,
    "rhinovirus": VirusType.COLD,
    
    # RSV
    "rsv": VirusType.RSV,
    "respiratory_syncytial": VirusType.RSV,
    
    # Pneumonia
    "pneumonia": VirusType.PNEUMONIA,
    "bacterial_pneumonia": VirusType.PNEUMONIA,
    "viral_pneumonia": VirusType.PNEUMONIA,
    
    # General respiratory
    "respiratory": VirusType.GENERAL,
    "respiratory_illness": VirusType.GENERAL,
    "symptomatic": VirusType.GENERAL,
    "sick": VirusType.GENERAL,
    
    # Healthy
    "healthy": VirusType.HEALTHY,
    "negative": VirusType.HEALTHY,
    "covid_negative": VirusType.HEALTHY,
    "control": VirusType.HEALTHY,
    "asymptomatic": VirusType.HEALTHY,
}


@dataclass
class RealSample:
    """Single sample from a real dataset."""
    
    sample_id: str
    dataset: str
    virus_type: VirusType
    
    # Audio features (if available)
    audio_path: Optional[str] = None
    audio_features: Optional[np.ndarray] = None  # MFCCs, spectral, etc.
    
    # Cough-specific features
    cough_features: Optional[np.ndarray] = None
    cough_count: int = 0
    
    # Voice features
    voice_features: Optional[np.ndarray] = None
    
    # Breathing features
    breathing_features: Optional[np.ndarray] = None
    
    # Physiological signals (HR, HRV, etc.)
    hr_mean: Optional[float] = None
    hrv_features: Optional[np.ndarray] = None
    
    # Activity/IMU features
    activity_features: Optional[np.ndarray] = None
    
    # Quality score
    quality: float = 1.0
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


class RealDataLoader:
    """Loader for real health datasets."""
    
    def __init__(
        self,
        data_dir: str = "data/",
        feature_extractors: Optional[Dict] = None,
    ):
        """Initialize real data loader.
        
        Args:
            data_dir: Base directory for downloaded datasets.
            feature_extractors: Optional dict of feature extraction functions.
        """
        self.data_dir = Path(data_dir)
        self.feature_extractors = feature_extractors or {}
        
        # Try to import audio processing libraries
        self._has_librosa = False
        try:
            import librosa
            self._has_librosa = True
        except ImportError:
            logger.warning("librosa not installed - audio feature extraction limited")
    
    def discover_datasets(self) -> Dict[str, Path]:
        """Discover available downloaded datasets.
        
        Returns:
            Dict mapping dataset name to path.
        """
        datasets = {}
        
        # Check for known dataset directories
        dataset_patterns = {
            "coughvid": ["coughvid*", "public_dataset*", "COUGHVID*"],
            "coswara": ["Coswara*", "coswara*"],
            "virufy": ["virufy*", "Virufy*"],
            "dicova": ["dicova*", "DiCOVA*"],
            "flusense": ["FluSense*", "flusense*"],
            "sound_dr": ["sound-dr*", "Sound-Dr*"],
            "wesad": ["WESAD*", "wesad*"],
            "esc50": ["ESC-50*", "esc50*"],
            "uci_har": ["UCI*HAR*", "har*", "Human*Activity*"],
            "wisdm": ["WISDM*", "wisdm*"],
            "librispeech": ["LibriSpeech*", "librispeech*", "dev-clean*"],
        }
        
        for name, patterns in dataset_patterns.items():
            for pattern in patterns:
                matches = list(self.data_dir.glob(pattern))
                if matches:
                    datasets[name] = matches[0]
                    break
        
        return datasets
    
    def load_coughvid(self, path: Path) -> List[RealSample]:
        """Load COUGHVID dataset.
        
        COUGHVID contains 25K+ cough recordings with COVID labels.
        """
        samples = []
        
        # Load metadata CSV
        metadata_file = path / "metadata_compiled.csv"
        if not metadata_file.exists():
            metadata_file = list(path.glob("**/metadata*.csv"))
            if metadata_file:
                metadata_file = metadata_file[0]
            else:
                logger.warning(f"No metadata found in {path}")
                return samples
        
        try:
            import pandas as pd
            df = pd.read_csv(metadata_file)
        except Exception as e:
            logger.warning(f"Could not load COUGHVID metadata: {e}")
            return samples
        
        # Process each row
        for _, row in df.iterrows():
            # Get COVID status
            status = str(row.get("status", row.get("covid_status", "unknown"))).lower()
            virus_type = LABEL_TO_VIRUS_TYPE.get(status, VirusType.GENERAL)
            
            # Get audio file
            uuid = row.get("uuid", row.get("id", ""))
            audio_path = path / f"{uuid}.webm"
            if not audio_path.exists():
                audio_path = path / f"{uuid}.ogg"
            if not audio_path.exists():
                audio_path = list(path.glob(f"**/{uuid}.*"))
                audio_path = audio_path[0] if audio_path else None
            
            sample = RealSample(
                sample_id=str(uuid),
                dataset="coughvid",
                virus_type=virus_type,
                audio_path=str(audio_path) if audio_path else None,
                quality=float(row.get("quality", row.get("cough_detected", 1.0))),
                metadata={
                    "age": row.get("age"),
                    "gender": row.get("gender"),
                    "respiratory_condition": row.get("respiratory_condition"),
                }
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from COUGHVID")
        return samples
    
    def load_coswara(self, path: Path) -> List[RealSample]:
        """Load Coswara dataset.
        
        Coswara contains breathing, cough, and speech with COVID labels.
        """
        samples = []
        
        # Find annotation file
        annotations = list(path.glob("**/combined_data.csv")) + list(path.glob("**/annotations*.csv"))
        
        if not annotations:
            # Try loading from folder structure
            for user_dir in path.glob("*/"):
                if not user_dir.is_dir():
                    continue
                
                # Check for metadata
                meta_file = user_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                        
                        status = str(meta.get("covid_status", "unknown")).lower()
                        virus_type = LABEL_TO_VIRUS_TYPE.get(status, VirusType.GENERAL)
                        
                        # Find audio files
                        audio_files = list(user_dir.glob("*.wav")) + list(user_dir.glob("*.mp3"))
                        
                        sample = RealSample(
                            sample_id=user_dir.name,
                            dataset="coswara",
                            virus_type=virus_type,
                            audio_path=str(audio_files[0]) if audio_files else None,
                            metadata=meta,
                        )
                        samples.append(sample)
                    except Exception as e:
                        continue
        else:
            # Load from CSV
            try:
                import pandas as pd
                df = pd.read_csv(annotations[0])
                
                for _, row in df.iterrows():
                    status = str(row.get("covid_status", row.get("status", "unknown"))).lower()
                    virus_type = LABEL_TO_VIRUS_TYPE.get(status, VirusType.GENERAL)
                    
                    sample = RealSample(
                        sample_id=str(row.get("id", row.name)),
                        dataset="coswara",
                        virus_type=virus_type,
                        metadata=row.to_dict(),
                    )
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Could not load Coswara CSV: {e}")
        
        logger.info(f"Loaded {len(samples)} samples from Coswara")
        return samples
    
    def load_virufy(self, path: Path) -> List[RealSample]:
        """Load Virufy COVID cough dataset."""
        samples = []
        
        # Virufy has PCR-confirmed COVID coughs
        for audio_file in path.glob("**/*.wav"):
            # Determine label from folder structure
            parent = audio_file.parent.name.lower()
            
            if "positive" in parent or "covid" in parent:
                virus_type = VirusType.COVID
            elif "negative" in parent or "healthy" in parent:
                virus_type = VirusType.HEALTHY
            else:
                virus_type = VirusType.GENERAL
            
            sample = RealSample(
                sample_id=audio_file.stem,
                dataset="virufy",
                virus_type=virus_type,
                audio_path=str(audio_file),
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from Virufy")
        return samples
    
    def load_flusense(self, path: Path) -> List[RealSample]:
        """Load FluSense hospital waiting room data."""
        samples = []
        
        # FluSense has cough/speech events with timestamps
        annotations = list(path.glob("**/annotations*.json")) + list(path.glob("**/labels*.csv"))
        
        if annotations:
            try:
                if annotations[0].suffix == ".json":
                    with open(annotations[0]) as f:
                        data = json.load(f)
                    
                    for item in data:
                        virus_type = VirusType.FLU if item.get("flu_positive") else VirusType.HEALTHY
                        
                        sample = RealSample(
                            sample_id=str(item.get("id", item.get("file_id"))),
                            dataset="flusense",
                            virus_type=virus_type,
                            cough_count=item.get("cough_count", 0),
                            metadata=item,
                        )
                        samples.append(sample)
                else:
                    import pandas as pd
                    df = pd.read_csv(annotations[0])
                    
                    for _, row in df.iterrows():
                        label = str(row.get("label", row.get("flu_status", ""))).lower()
                        virus_type = LABEL_TO_VIRUS_TYPE.get(label, VirusType.GENERAL)
                        
                        sample = RealSample(
                            sample_id=str(row.name),
                            dataset="flusense",
                            virus_type=virus_type,
                            metadata=row.to_dict(),
                        )
                        samples.append(sample)
            except Exception as e:
                logger.warning(f"Could not load FluSense annotations: {e}")
        
        logger.info(f"Loaded {len(samples)} samples from FluSense")
        return samples
    
    def load_dicova(self, path: Path) -> List[RealSample]:
        """Load DiCOVA respiratory illness dataset."""
        samples = []
        
        # DiCOVA challenge data structure
        for split in ["train", "val", "test", "Track1", "Track2"]:
            split_dir = path / split
            if not split_dir.exists():
                continue
            
            # Load labels
            label_file = split_dir / "labels.csv"
            if not label_file.exists():
                label_file = list(split_dir.glob("*labels*.csv"))
                label_file = label_file[0] if label_file else None
            
            if label_file:
                try:
                    import pandas as pd
                    df = pd.read_csv(label_file)
                    
                    for _, row in df.iterrows():
                        label = str(row.get("label", row.get("covid_status", "unknown"))).lower()
                        virus_type = LABEL_TO_VIRUS_TYPE.get(label, VirusType.GENERAL)
                        
                        sample = RealSample(
                            sample_id=str(row.get("file_name", row.name)),
                            dataset="dicova",
                            virus_type=virus_type,
                            metadata=row.to_dict(),
                        )
                        samples.append(sample)
                except Exception as e:
                    continue
        
        logger.info(f"Loaded {len(samples)} samples from DiCOVA")
        return samples
    
    def load_wesad(self, path: Path) -> List[RealSample]:
        """Load WESAD physiological dataset.
        
        Note: WESAD has stress labels, not illness labels.
        We use it to learn physiological baselines.
        """
        samples = []
        
        # WESAD has subject folders with pickle files
        for subj_dir in path.glob("S*"):
            if not subj_dir.is_dir():
                continue
            
            pkl_file = subj_dir / f"{subj_dir.name}.pkl"
            if pkl_file.exists():
                # Note: Not loading pickle here, just recording metadata
                sample = RealSample(
                    sample_id=subj_dir.name,
                    dataset="wesad",
                    virus_type=VirusType.HEALTHY,  # Use as healthy baseline
                    audio_path=str(pkl_file),  # Store path for later processing
                    metadata={"type": "physiological"},
                )
                samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from WESAD")
        return samples
    
    def load_all(self, n_workers: int = 4) -> List[RealSample]:
        """Load all available datasets.
        
        Args:
            n_workers: Number of parallel workers.
            
        Returns:
            Combined list of samples from all datasets.
        """
        datasets = self.discover_datasets()
        
        if not datasets:
            logger.warning(f"No datasets found in {self.data_dir}")
            logger.info("Run: python scripts/download_more_data.py --health --parallel 4")
            return []
        
        logger.info(f"Found {len(datasets)} datasets: {list(datasets.keys())}")
        
        # Dataset loaders
        loaders = {
            "coughvid": self.load_coughvid,
            "coswara": self.load_coswara,
            "virufy": self.load_virufy,
            "flusense": self.load_flusense,
            "dicova": self.load_dicova,
            "wesad": self.load_wesad,
        }
        
        all_samples = []
        
        # Load in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            
            for name, path in datasets.items():
                if name in loaders:
                    future = executor.submit(loaders[name], path)
                    futures[future] = name
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                except Exception as e:
                    logger.warning(f"Error loading {name}: {e}")
        
        # Log distribution
        virus_counts = {}
        for sample in all_samples:
            vt = VIRUS_NAMES[sample.virus_type]
            virus_counts[vt] = virus_counts.get(vt, 0) + 1
        
        logger.info(f"Total samples: {len(all_samples)}")
        logger.info(f"Virus distribution: {virus_counts}")
        
        return all_samples
    
    def extract_audio_features(
        self,
        audio_path: str,
        sr: int = 16000,
        n_mfcc: int = 13,
    ) -> Optional[np.ndarray]:
        """Extract audio features from a file.
        
        Args:
            audio_path: Path to audio file.
            sr: Sample rate.
            n_mfcc: Number of MFCC coefficients.
            
        Returns:
            Feature array or None if extraction fails.
        """
        if not self._has_librosa:
            return None
        
        try:
            import librosa
            
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr)
            
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                return None
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract other features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Combine features
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, zero_crossing, rms],
            ])
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Could not extract features from {audio_path}: {e}")
            return None


def load_health_datasets(
    data_dir: str = "data/",
    n_workers: int = 4,
) -> List[RealSample]:
    """Convenience function to load all health datasets.
    
    Args:
        data_dir: Base directory for datasets.
        n_workers: Number of parallel workers.
        
    Returns:
        List of RealSample objects.
    """
    loader = RealDataLoader(data_dir=data_dir)
    return loader.load_all(n_workers=n_workers)


def create_training_data(
    samples: List[RealSample],
    output_dir: str = "data/processed/",
    extract_features: bool = True,
    n_workers: int = 4,
) -> Dict[str, Any]:
    """Process samples into training-ready format.
    
    Args:
        samples: List of RealSample objects.
        output_dir: Output directory for processed data.
        extract_features: Whether to extract audio features.
        n_workers: Number of parallel workers.
        
    Returns:
        Dict with processed data info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = RealDataLoader()
    
    # Extract features if requested
    if extract_features:
        logger.info("Extracting audio features...")
        
        def process_sample(sample: RealSample) -> RealSample:
            if sample.audio_path and os.path.exists(sample.audio_path):
                features = loader.extract_audio_features(sample.audio_path)
                if features is not None:
                    sample.audio_features = features
            return sample
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            samples = list(executor.map(process_sample, samples))
    
    # Split into train/val/test
    np.random.shuffle(samples)
    n = len(samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    # Save processed data
    def save_split(split_samples: List[RealSample], name: str):
        split_data = []
        for s in split_samples:
            split_data.append({
                "sample_id": s.sample_id,
                "dataset": s.dataset,
                "virus_type": s.virus_type.value,
                "virus_name": VIRUS_NAMES[s.virus_type],
                "audio_features": s.audio_features.tolist() if s.audio_features is not None else None,
                "quality": s.quality,
            })
        
        with open(output_path / f"{name}.json", "w") as f:
            json.dump(split_data, f)
        
        return len(split_data)
    
    n_train = save_split(train_samples, "train")
    n_val = save_split(val_samples, "val")
    n_test = save_split(test_samples, "test")
    
    # Save metadata
    metadata = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "virus_types": [VIRUS_NAMES[vt] for vt in VirusType],
        "datasets_used": list(set(s.dataset for s in samples)),
    }
    
    with open(output_path / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    
    logger.info(f"Saved processed data to {output_path}")
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    return metadata

