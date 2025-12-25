"""PyTorch Dataset for real health data.

Loads processed health datasets and creates training batches.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from viralflip.model.virus_types import VirusType, NUM_VIRUS_CLASSES


# Default feature dimensions per modality
DEFAULT_FEATURE_DIMS = {
    "voice": 24,
    "cough": 30,  # MFCCs + spectral features from audio
    "breathing": 20,
    "rppg": 5,
    "activity": 6,
    "audio": 30,  # Generic audio features
}


class ViralFlipDataset(Dataset):
    """Dataset for ViralFlip training from real health data."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        feature_dims: Optional[Dict[str, int]] = None,
        max_seq_len: int = 13,  # max_lag + 1
        augment: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to processed data directory.
            split: One of 'train', 'val', 'test'.
            feature_dims: Feature dimensions per modality.
            max_seq_len: Maximum sequence length.
            augment: Whether to apply data augmentation.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.feature_dims = feature_dims or DEFAULT_FEATURE_DIMS
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        # Load data
        split_file = self.data_path / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {split_file}\n"
                f"Run: python scripts/prepare_real_data.py to process datasets"
            )
        
        with open(split_file) as f:
            self.samples = json.load(f)
        
        # Build modality list from feature dims
        self.modalities = list(self.feature_dims.keys())
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dict with features, labels, and metadata.
        """
        sample = self.samples[idx]
        
        # Build feature tensors
        features = {}
        missing = {}
        quality = {}
        
        # Get audio features if available
        audio_features = sample.get("audio_features")
        if audio_features is not None:
            audio_features = np.array(audio_features, dtype=np.float32)
        
        for modality in self.modalities:
            dim = self.feature_dims[modality]
            
            if modality in ["cough", "voice", "breathing", "audio"] and audio_features is not None:
                # Use audio features, padded/truncated to expected dim
                feat = np.zeros((self.max_seq_len, dim), dtype=np.float32)
                
                # Copy features to last timestep (most recent)
                if len(audio_features) >= dim:
                    feat[-1, :] = audio_features[:dim]
                else:
                    feat[-1, :len(audio_features)] = audio_features
                
                # Simulate temporal context by copying with noise
                for t in range(self.max_seq_len - 1):
                    noise_factor = 0.1 * (self.max_seq_len - 1 - t) / self.max_seq_len
                    feat[t, :] = feat[-1, :] * (1 - noise_factor) + np.random.randn(dim) * noise_factor * 0.1
                
                features[modality] = torch.from_numpy(feat)
                missing[modality] = torch.zeros(self.max_seq_len, dtype=torch.bool)
                quality[modality] = torch.ones(self.max_seq_len) * sample.get("quality", 1.0)
            else:
                # Missing modality
                features[modality] = torch.zeros(self.max_seq_len, dim)
                missing[modality] = torch.ones(self.max_seq_len, dtype=torch.bool)
                quality[modality] = torch.zeros(self.max_seq_len)
        
        # Get virus type label
        virus_type = sample.get("virus_type", 0)
        
        # Create binary risk labels (1 if any illness, 0 if healthy)
        is_ill = 1.0 if virus_type > 0 else 0.0
        
        # Multi-horizon labels (for compatibility with trainer)
        # Since we have point samples, all horizons have same label
        labels = torch.tensor([is_ill, is_ill, is_ill], dtype=torch.float32)
        
        # Data augmentation
        if self.augment and self.split == "train":
            features = self._augment(features)
        
        return {
            "features": features,
            "missing": missing,
            "quality": quality,
            "labels": labels,
            "virus_type": torch.tensor(virus_type, dtype=torch.long),
            "user_id": sample.get("sample_id", str(idx)),
            "bin_idx": 0,
        }
    
    def _augment(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation.
        
        Args:
            features: Dict of feature tensors.
            
        Returns:
            Augmented features.
        """
        augmented = {}
        
        for modality, feat in features.items():
            # Add small Gaussian noise
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(feat) * 0.05
                feat = feat + noise
            
            # Random scaling
            if torch.rand(1) < 0.3:
                scale = 0.9 + torch.rand(1) * 0.2  # 0.9 to 1.1
                feat = feat * scale
            
            augmented[modality] = feat
        
        return augmented


def collate_viralflip_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for ViralFlip batches.
    
    Args:
        batch: List of sample dicts.
        
    Returns:
        Batched dict with stacked tensors.
    """
    modalities = list(batch[0]["features"].keys())
    
    features = {}
    missing = {}
    quality = {}
    
    for modality in modalities:
        features[modality] = torch.stack([b["features"][modality] for b in batch])
        missing[modality] = torch.stack([b["missing"][modality] for b in batch])
        quality[modality] = torch.stack([b["quality"][modality] for b in batch])
    
    labels = torch.stack([b["labels"] for b in batch])
    virus_type = torch.stack([b["virus_type"] for b in batch])
    user_ids = [b["user_id"] for b in batch]
    
    return {
        "features": features,
        "missing": missing,
        "quality": quality,
        "labels": labels,
        "virus_type": virus_type,
        "user_ids": user_ids,
    }


class CombinedHealthDataset(Dataset):
    """Dataset that combines multiple health data sources.
    
    Handles mixing samples from different datasets with different
    feature availability.
    """
    
    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        target_size: Optional[int] = None,
    ):
        """Initialize combined dataset.
        
        Args:
            datasets: List of dataset configs with 'path' and optional 'weight'.
            weights: Sampling weights per dataset.
            target_size: Target dataset size (samples from weighted mix).
        """
        self.all_samples = []
        self.dataset_indices = []
        
        for i, ds_config in enumerate(datasets):
            path = Path(ds_config["path"])
            if path.exists():
                with open(path) as f:
                    samples = json.load(f)
                self.all_samples.extend(samples)
                self.dataset_indices.extend([i] * len(samples))
        
        self.weights = weights
        self.target_size = target_size or len(self.all_samples)
    
    def __len__(self) -> int:
        return min(self.target_size, len(self.all_samples))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.all_samples[idx % len(self.all_samples)]
        
        # Basic processing
        audio_features = sample.get("audio_features")
        if audio_features is not None:
            audio_features = np.array(audio_features, dtype=np.float32)
            features = {"audio": torch.from_numpy(audio_features)}
        else:
            features = {"audio": torch.zeros(30)}
        
        virus_type = sample.get("virus_type", 0)
        is_ill = 1.0 if virus_type > 0 else 0.0
        
        return {
            "features": features,
            "labels": torch.tensor([is_ill, is_ill, is_ill], dtype=torch.float32),
            "virus_type": torch.tensor(virus_type, dtype=torch.long),
            "sample_id": sample.get("sample_id", str(idx)),
            "dataset": sample.get("dataset", "unknown"),
        }

