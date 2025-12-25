"""Quality assessment module for all sensor modalities.

Provides unified quality assessment and aggregation across all feature blocks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModalityQuality:
    """Quality metrics for a single modality."""
    
    name: str
    quality_score: float  # 0 to 1
    is_missing: bool
    quality_vector: np.ndarray
    
    def __repr__(self) -> str:
        status = "MISSING" if self.is_missing else f"{self.quality_score:.2f}"
        return f"ModalityQuality({self.name}: {status})"


@dataclass  
class AggregatedQuality:
    """Aggregated quality metrics across all modalities."""
    
    modalities: dict[str, ModalityQuality]
    overall_score: float
    n_present: int
    n_total: int
    missing_rate: float
    
    def to_confidence_inputs(self) -> dict[str, float]:
        """Get inputs for confidence scoring."""
        return {
            "n_present": self.n_present,
            "n_total": self.n_total,
            "mean_quality": self.overall_score,
            "missing_rate": self.missing_rate,
        }


class QualityAssessor:
    """Unified quality assessment across modalities."""
    
    # Expected modalities and their quality dimensions
    MODALITIES = {
        "voice": {"dims": 3, "names": ["snr_est", "voiced_fraction", "clipping_rate"]},
        "cough": {"dims": 1, "names": ["mic_uptime_fraction"]},
        "tap": {"dims": 1, "names": ["completion_flag"]},
        "gait_active": {"dims": 2, "names": ["steps_detected_ok", "placement_flag"]},
        "imu_passive": {"dims": 1, "names": ["imu_uptime_fraction"]},
        "rppg": {"dims": 2, "names": ["sqi", "motion_artifact_score"]},
        "gps": {"dims": 1, "names": ["gps_uptime_fraction"]},
        "light": {"dims": 1, "names": ["light_uptime_fraction"]},
        "baro": {"dims": 1, "names": ["baro_uptime_fraction"]},
        "screen": {"dims": 1, "names": ["screen_event_uptime_fraction"]},
    }
    
    # Minimum quality thresholds for each modality
    QUALITY_THRESHOLDS = {
        "voice": 0.3,
        "cough": 0.5,
        "tap": 0.5,
        "gait_active": 0.5,
        "imu_passive": 0.3,
        "rppg": 0.3,
        "gps": 0.3,
        "light": 0.3,
        "baro": 0.3,
        "screen": 0.5,
    }
    
    def __init__(
        self,
        quality_thresholds: dict[str, float] | None = None,
    ):
        """Initialize quality assessor.
        
        Args:
            quality_thresholds: Optional custom quality thresholds per modality.
        """
        self.thresholds = quality_thresholds or self.QUALITY_THRESHOLDS.copy()
    
    def assess_modality(
        self,
        modality: str,
        quality_vector: np.ndarray | None,
        features: np.ndarray | None = None,
    ) -> ModalityQuality:
        """Assess quality for a single modality.
        
        Args:
            modality: Modality name.
            quality_vector: Quality metrics array.
            features: Optional feature array to check for validity.
            
        Returns:
            ModalityQuality object.
        """
        if quality_vector is None or len(quality_vector) == 0:
            return ModalityQuality(
                name=modality,
                quality_score=0.0,
                is_missing=True,
                quality_vector=np.array([]),
            )
        
        # Check if features are all zeros (missing data indicator)
        if features is not None and np.all(features == 0):
            return ModalityQuality(
                name=modality,
                quality_score=0.0,
                is_missing=True,
                quality_vector=quality_vector,
            )
        
        # Compute quality score based on modality
        if modality == "voice":
            # SNR, voiced fraction, and low clipping
            snr_score = np.clip(quality_vector[0] / 30.0, 0, 1)  # 30 dB is good
            voiced_score = quality_vector[1]  # 0-1
            clipping_score = 1.0 - quality_vector[2]  # Lower is better
            quality_score = (snr_score + voiced_score + clipping_score) / 3
            
        elif modality == "rppg":
            # SQI is main quality; motion artifact inverted
            sqi = quality_vector[0]
            motion = 1.0 - quality_vector[1]  # Lower motion is better
            quality_score = (sqi + motion) / 2
            
        elif modality == "gait_active":
            # Both flags should be 1
            quality_score = np.mean(quality_vector)
            
        else:
            # For other modalities, quality is typically uptime fraction
            quality_score = np.mean(quality_vector)
        
        quality_score = float(np.clip(quality_score, 0, 1))
        threshold = self.thresholds.get(modality, 0.3)
        is_missing = quality_score < threshold
        
        return ModalityQuality(
            name=modality,
            quality_score=quality_score,
            is_missing=is_missing,
            quality_vector=quality_vector,
        )
    
    def assess_all(
        self,
        quality_dict: dict[str, np.ndarray],
        feature_dict: dict[str, np.ndarray] | None = None,
    ) -> AggregatedQuality:
        """Assess quality across all modalities.
        
        Args:
            quality_dict: Dict mapping modality name to quality vector.
            feature_dict: Optional dict mapping modality name to feature vector.
            
        Returns:
            AggregatedQuality object.
        """
        modalities = {}
        n_present = 0
        quality_sum = 0.0
        
        for modality in self.MODALITIES:
            quality_vec = quality_dict.get(modality)
            features = feature_dict.get(modality) if feature_dict else None
            
            mod_quality = self.assess_modality(modality, quality_vec, features)
            modalities[modality] = mod_quality
            
            if not mod_quality.is_missing:
                n_present += 1
                quality_sum += mod_quality.quality_score
        
        n_total = len(self.MODALITIES)
        overall_score = quality_sum / n_present if n_present > 0 else 0.0
        missing_rate = 1.0 - (n_present / n_total)
        
        return AggregatedQuality(
            modalities=modalities,
            overall_score=overall_score,
            n_present=n_present,
            n_total=n_total,
            missing_rate=missing_rate,
        )
    
    def create_missing_mask(
        self,
        quality_dict: dict[str, np.ndarray],
        feature_dict: dict[str, np.ndarray] | None = None,
    ) -> dict[str, bool]:
        """Create missing indicator mask for all modalities.
        
        Args:
            quality_dict: Dict mapping modality name to quality vector.
            feature_dict: Optional dict mapping modality name to feature vector.
            
        Returns:
            Dict mapping modality name to missing flag.
        """
        aggregated = self.assess_all(quality_dict, feature_dict)
        return {name: mq.is_missing for name, mq in aggregated.modalities.items()}
    
    @staticmethod
    def get_modality_dim(modality: str) -> int:
        """Get quality vector dimension for a modality.
        
        Args:
            modality: Modality name.
            
        Returns:
            Number of quality dimensions.
        """
        return QualityAssessor.MODALITIES.get(modality, {}).get("dims", 1)

