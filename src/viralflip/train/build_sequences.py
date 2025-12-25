"""Build training sequences from user data with proper splits.

Key responsibilities:
1. User-level split (train users vs test users)
2. Temporal split (within user: train/val/test by time)
3. Sequence construction with lag context
4. Label derivation from onset timestamps

LEAKAGE PREVENTION:
- User-level split ensures no user appears in multiple splits
- Temporal split ensures no future data leaks into training
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class UserBin:
    """Single time bin for a user."""
    
    user_id: str
    bin_idx: int
    timestamp: float  # Unix timestamp or bin index
    
    # Feature blocks: modality -> features
    features: dict[str, np.ndarray]
    
    # Quality: modality -> quality vector
    quality: dict[str, np.ndarray]
    
    # Missing mask: modality -> bool
    missing: dict[str, bool]
    
    # Labels for each horizon
    labels: dict[int, int]  # horizon -> 0/1
    
    # Virus type label (0=HEALTHY, 1=COVID, 2=FLU, etc.)
    virus_type: int = 0
    
    # Whether in washout period (exclude from training)
    in_washout: bool = False


@dataclass
class UserData:
    """Complete data for one user."""
    
    user_id: str
    bins: list[UserBin]
    onset_timestamps: list[float]  # Illness onset times
    
    def __len__(self) -> int:
        return len(self.bins)
    
    def get_features_array(self, modality: str) -> np.ndarray:
        """Get feature array for modality across all bins."""
        return np.array([b.features.get(modality, np.zeros(1)) for b in self.bins])
    
    def get_labels_array(self, horizon: int) -> np.ndarray:
        """Get label array for horizon across all bins."""
        return np.array([b.labels.get(horizon, 0) for b in self.bins])


class SequenceBuilder:
    """Build training sequences from user data."""
    
    def __init__(
        self,
        horizons: list[int] = [24, 48, 72],
        bin_hours: int = 6,
        max_lag: int = 12,
        washout_bins: int = 28,  # 7 days * 4 bins/day
        train_user_frac: float = 0.7,
        val_user_frac: float = 0.15,
        train_time_frac: float = 0.7,
        val_time_frac: float = 0.15,
        seed: int = 42,
    ):
        """Initialize sequence builder.
        
        Args:
            horizons: Prediction horizons in hours.
            bin_hours: Duration of each bin in hours.
            max_lag: Maximum lag for sequence context.
            washout_bins: Bins to exclude after onset.
            train_user_frac: Fraction of users for training.
            val_user_frac: Fraction of users for validation.
            train_time_frac: Fraction of time for training (within user).
            val_time_frac: Fraction of time for validation (within user).
            seed: Random seed for splits.
        """
        self.horizons = horizons
        self.bin_hours = bin_hours
        self.max_lag = max_lag
        self.washout_bins = washout_bins
        self.train_user_frac = train_user_frac
        self.val_user_frac = val_user_frac
        self.train_time_frac = train_time_frac
        self.val_time_frac = val_time_frac
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
    
    def derive_labels(
        self,
        n_bins: int,
        onset_bin_indices: list[int],
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Derive multi-horizon labels from onset indices.
        
        Args:
            n_bins: Total number of bins.
            onset_bin_indices: List of bin indices where illness started.
            
        Returns:
            Tuple of (labels_dict, washout_mask).
            labels_dict: horizon -> array of shape (n_bins,).
            washout_mask: Boolean array, True = in washout.
        """
        labels = {h: np.zeros(n_bins, dtype=np.int32) for h in self.horizons}
        washout = np.zeros(n_bins, dtype=bool)
        
        for onset_idx in onset_bin_indices:
            # Mark washout period
            washout_start = onset_idx
            washout_end = min(onset_idx + self.washout_bins, n_bins)
            washout[washout_start:washout_end] = True
            
            # Compute labels
            for horizon in self.horizons:
                horizon_bins = horizon // self.bin_hours
                
                # Which bins have onset in (t, t+horizon]?
                # For bin t: onset in (t, t + horizon_bins]
                # So: onset_idx - horizon_bins <= t < onset_idx
                label_start = max(0, onset_idx - horizon_bins)
                label_end = onset_idx
                
                labels[horizon][label_start:label_end] = 1
        
        return labels, washout
    
    def split_users(
        self,
        user_ids: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Split users into train/val/test.
        
        Args:
            user_ids: List of all user IDs.
            
        Returns:
            Tuple of (train_users, val_users, test_users).
        """
        n_users = len(user_ids)
        shuffled = self._rng.permutation(user_ids)
        
        n_train = int(n_users * self.train_user_frac)
        n_val = int(n_users * self.val_user_frac)
        
        train_users = list(shuffled[:n_train])
        val_users = list(shuffled[n_train:n_train + n_val])
        test_users = list(shuffled[n_train + n_val:])
        
        return train_users, val_users, test_users
    
    def split_temporal(
        self,
        n_bins: int,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Split bins temporally into train/val/test ranges.
        
        Args:
            n_bins: Total number of bins.
            
        Returns:
            Tuple of ((train_start, train_end), (val_start, val_end), (test_start, test_end)).
        """
        n_train = int(n_bins * self.train_time_frac)
        n_val = int(n_bins * self.val_time_frac)
        
        train_range = (0, n_train)
        val_range = (n_train, n_train + n_val)
        test_range = (n_train + n_val, n_bins)
        
        return train_range, val_range, test_range
    
    def build_sequences(
        self,
        user_data: UserData,
        time_range: tuple[int, int],
        modalities: list[str],
    ) -> list[dict]:
        """Build training sequences for a user within a time range.
        
        Args:
            user_data: User data object.
            time_range: (start_bin, end_bin) for this split.
            modalities: List of modalities to include.
            
        Returns:
            List of sequence dicts, each containing features, labels, etc.
        """
        sequences = []
        start_bin, end_bin = time_range
        
        for t in range(start_bin, end_bin):
            bin_data = user_data.bins[t]
            
            # Skip washout periods
            if bin_data.in_washout:
                continue
            
            # Build sequence with lag context
            seq_start = max(0, t - self.max_lag)
            seq_end = t + 1
            
            # Collect features for each modality
            seq_features = {}
            seq_missing = {}
            seq_quality = {}
            
            for modality in modalities:
                # Get features for sequence
                features = []
                missing = []
                quality = []
                
                for s in range(seq_start, seq_end):
                    s_bin = user_data.bins[s]
                    
                    if modality in s_bin.features:
                        features.append(s_bin.features[modality])
                        missing.append(s_bin.missing.get(modality, False))
                        quality.append(s_bin.quality.get(modality, np.array([1.0])))
                    else:
                        # Missing modality
                        features.append(np.zeros_like(user_data.bins[0].features.get(
                            modality, np.zeros(1)
                        )))
                        missing.append(True)
                        quality.append(np.array([0.0]))
                
                # Pad to max_lag + 1 if needed
                while len(features) < self.max_lag + 1:
                    features.insert(0, np.zeros_like(features[0]))
                    missing.insert(0, True)
                    quality.insert(0, np.array([0.0]))
                
                seq_features[modality] = np.array(features)
                seq_missing[modality] = np.array(missing)
                seq_quality[modality] = np.array([q.mean() for q in quality])
            
            # Get labels
            labels = np.array([bin_data.labels.get(h, 0) for h in self.horizons])
            
            # Get virus type
            virus_type = getattr(bin_data, 'virus_type', 0)
            
            sequences.append({
                "user_id": user_data.user_id,
                "bin_idx": t,
                "features": seq_features,
                "missing": seq_missing,
                "quality": seq_quality,
                "labels": labels,
                "virus_type": virus_type,
            })
        
        return sequences


class UserDataset(Dataset):
    """PyTorch dataset for user sequences."""
    
    def __init__(
        self,
        sequences: list[dict],
        modalities: list[str],
        feature_dims: dict[str, int],
    ):
        """Initialize dataset.
        
        Args:
            sequences: List of sequence dicts from SequenceBuilder.
            modalities: List of modality names.
            feature_dims: Dict mapping modality to feature dimension.
        """
        self.sequences = sequences
        self.modalities = modalities
        self.feature_dims = feature_dims
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sequence.
        
        Returns:
            Dict with tensors for features, missing, quality, labels, virus_type.
        """
        seq = self.sequences[idx]
        
        features = {}
        missing = {}
        quality = {}
        
        for modality in self.modalities:
            if modality in seq["features"]:
                features[modality] = torch.from_numpy(seq["features"][modality]).float()
                missing[modality] = torch.from_numpy(seq["missing"][modality]).bool()
                quality[modality] = torch.from_numpy(seq["quality"][modality]).float()
            else:
                # Create zeros
                seq_len = seq["features"][self.modalities[0]].shape[0]
                dim = self.feature_dims[modality]
                features[modality] = torch.zeros(seq_len, dim)
                missing[modality] = torch.ones(seq_len, dtype=torch.bool)
                quality[modality] = torch.zeros(seq_len)
        
        labels = torch.from_numpy(seq["labels"]).float()
        
        # Virus type (0 = HEALTHY if not present)
        virus_type = torch.tensor(seq.get("virus_type", 0), dtype=torch.long)
        
        return {
            "features": features,
            "missing": missing,
            "quality": quality,
            "labels": labels,
            "virus_type": virus_type,
            "user_id": seq["user_id"],
            "bin_idx": seq["bin_idx"],
        }


def collate_user_batch(batch: list[dict]) -> dict:
    """Collate function for user sequences.
    
    Args:
        batch: List of sequence dicts.
        
    Returns:
        Batched dict with stacked tensors.
    """
    modalities = list(batch[0]["features"].keys())
    
    # Stack features
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

