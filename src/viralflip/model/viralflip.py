"""ViralFlip: Unified illness prediction model with best-in-class features.

Combines the best features from all previous models:
- Pretrained encoder for robust representations (optional)
- Drift score compression with interpretable weights
- Lag lattice multi-horizon hazard model
- Sparse pairwise modality interactions
- Per-user personalization
- Multi-class virus type classification
- Confidence scoring based on data quality
- Conformal prediction for calibrated uncertainty
- Environment-invariant learning (IRM compatible)
- Uncertainty decomposition (aleatoric + epistemic)

Architecture:
1. Drift Score (φ): Compress per-modality drifts to scalar scores
   - Optional pretrained encoder for enhanced representations
2. Lag Lattice: Multi-horizon hazard model with temporal structure
3. Interactions: Sparse pairwise modality interactions
4. Personalization: Per-user calibration
5. Virus Classifier: Multi-class classification of virus type
6. Conformal Predictor: Calibrated uncertainty with coverage guarantees
7. Environment Classifier: IRM-compatible environment detection
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from viralflip.model.drift_score import DriftScoreModule
from viralflip.model.lag_lattice import LagLatticeHazardModel
from viralflip.model.interactions import InteractionModule
from viralflip.model.personalization import PersonalizationLayer
from viralflip.model.virus_types import (
    VirusType, NUM_VIRUS_CLASSES, VIRUS_NAMES, VIRUS_SHORT_NAMES
)


@dataclass
class ViralFlipOutput:
    """Output container for ViralFlip predictions."""
    
    # Risk probabilities for each horizon
    risks: dict[int, float]  # horizon -> probability
    
    # Confidence scores
    confidences: dict[int, float]  # horizon -> confidence
    
    # Raw logits (before personalization)
    raw_logits: dict[int, float]
    
    # Quality summary
    quality_summary: dict[str, Any] = field(default_factory=dict)
    
    # For interpretability
    drift_scores: dict[str, float] = field(default_factory=dict)  # modality -> drift score
    
    # Virus type classification
    virus_probs: Optional[dict[str, float]] = None  # virus_name -> probability
    virus_prediction: Optional[str] = None  # Most likely virus type
    virus_confidence: Optional[float] = None  # Confidence in virus prediction
    
    # Conformal prediction
    conformal_lower: Optional[dict[int, float]] = None  # horizon -> lower bound
    conformal_upper: Optional[dict[int, float]] = None  # horizon -> upper bound
    should_alert: bool = False
    alert_confidence: float = 0.0
    
    # Encoder representation (for interpretability)
    encoder_representation: Optional[np.ndarray] = None
    
    # Active sensing recommendation
    recommended_sensors: Optional[list[str]] = None
    
    # Uncertainty decomposition
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    
    # Environment info
    detected_environment: Optional[str] = None
    in_transition: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "risks": self.risks,
            "confidences": self.confidences,
            "raw_logits": self.raw_logits,
            "quality_summary": self.quality_summary,
            "drift_scores": self.drift_scores,
            "should_alert": self.should_alert,
            "alert_confidence": self.alert_confidence,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "epistemic_uncertainty": self.epistemic_uncertainty,
        }
        
        # Add virus classification if available
        if self.virus_probs is not None:
            result["virus_probs"] = self.virus_probs
            result["virus_prediction"] = self.virus_prediction
            result["virus_confidence"] = self.virus_confidence
        
        # Add conformal bounds if available
        if self.conformal_lower is not None:
            result["conformal_lower"] = self.conformal_lower
            result["conformal_upper"] = self.conformal_upper
        
        # Add environment info if available
        if self.detected_environment is not None:
            result["detected_environment"] = self.detected_environment
            result["in_transition"] = self.in_transition
        
        return result
    
    def get_illness_summary(self) -> str:
        """Get human-readable illness summary."""
        # Find max risk across horizons
        max_risk = max(self.risks.values())
        max_horizon = max(self.risks.keys(), key=lambda h: self.risks[h])
        
        if max_risk < 0.2:
            return "Low risk of illness"
        
        summary = f"Risk: {max_risk:.0%} chance of illness in {max_horizon}h"
        
        if self.virus_prediction and self.virus_prediction != "Healthy":
            summary += f" | Likely: {self.virus_prediction} ({self.virus_confidence:.0%} conf)"
        
        if self.should_alert:
            summary += f" | ALERT (conf: {self.alert_confidence:.0%})"
        
        return summary


class VirusClassifier(nn.Module):
    """Multi-class virus type classifier.
    
    Takes drift scores and predicts which type of virus is causing the illness.
    Uses a small MLP to transform drift patterns into virus type probabilities.
    """
    
    def __init__(
        self,
        n_modalities: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        n_classes: int = NUM_VIRUS_CLASSES,
    ):
        """Initialize virus classifier.
        
        Args:
            n_modalities: Number of input modalities (drift scores).
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
            n_classes: Number of virus classes (including HEALTHY).
        """
        super().__init__()
        
        self.n_classes = n_classes
        
        # MLP classifier
        # Input: drift scores concatenated across modalities + risk score
        input_dim = n_modalities + 1  # +1 for aggregated risk
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        drift_scores: torch.Tensor,
        risk_score: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            drift_scores: Drift scores, shape (batch, n_modalities).
            risk_score: Aggregated risk score, shape (batch,) or (batch, 1).
            
        Returns:
            Virus class logits, shape (batch, n_classes).
        """
        # Ensure risk_score has correct shape
        if risk_score.dim() == 1:
            risk_score = risk_score.unsqueeze(-1)
        
        # Concatenate drift scores with risk
        features = torch.cat([drift_scores, risk_score], dim=-1)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(
        self,
        drift_scores: torch.Tensor,
        risk_score: torch.Tensor,
    ) -> torch.Tensor:
        """Get probability distribution over virus types.
        
        Args:
            drift_scores: Drift scores, shape (batch, n_modalities).
            risk_score: Aggregated risk score, shape (batch,).
            
        Returns:
            Virus class probabilities, shape (batch, n_classes).
        """
        logits = self.forward(drift_scores, risk_score)
        return F.softmax(logits, dim=-1)


class ConfidenceScorer(nn.Module):
    """Score prediction confidence based on data quality and missingness."""
    
    def __init__(
        self,
        n_modalities: int,
        gamma0: float = 0.0,
        gamma1: float = 0.3,
        gamma2: float = 0.2,
        gamma3: float = 0.5,
    ):
        """Initialize confidence scorer.
        
        Args:
            n_modalities: Total number of modalities.
            gamma0: Intercept.
            gamma1: Weight for modality presence.
            gamma2: Weight for mean quality.
            gamma3: Weight for missing rate (negative effect).
        """
        super().__init__()
        
        self.n_modalities = n_modalities
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
    
    def forward(
        self,
        missing_mask: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence scores.
        
        Args:
            missing_mask: Binary mask, shape (batch, n_modalities).
                         1 = missing, 0 = present.
            quality_scores: Quality scores, shape (batch, n_modalities).
            
        Returns:
            Confidence scores, shape (batch,).
        """
        # Number of present modalities
        n_present = (1 - missing_mask).sum(dim=-1)  # (batch,)
        
        # Mean quality of present modalities
        present_quality = quality_scores * (1 - missing_mask)
        quality_sum = present_quality.sum(dim=-1)
        mean_quality = quality_sum / (n_present + 1e-6)
        
        # Missing rate
        missing_rate = missing_mask.float().mean(dim=-1)
        
        # Confidence score
        logit = (
            self.gamma0 +
            self.gamma1 * (n_present / self.n_modalities) +
            self.gamma2 * mean_quality -
            self.gamma3 * missing_rate
        )
        
        confidence = torch.sigmoid(logit)
        
        return confidence


class EncoderBackedDriftScore(nn.Module):
    """Drift score module backed by pretrained encoder.
    
    Uses pretrained encoder representations instead of simple
    linear drift scoring, while maintaining interpretability.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        modality_dims: dict[str, int],
        embed_dim: int = 128,
        use_encoder_features: bool = True,
        l1_lambda: float = 0.01,
    ):
        """Initialize encoder-backed drift score.
        
        Args:
            encoder: Pretrained multimodal encoder
            modality_dims: Dict mapping modality to feature dimension
            embed_dim: Encoder embedding dimension
            use_encoder_features: Whether to use encoder or fall back to linear
            l1_lambda: L1 regularization strength
        """
        super().__init__()
        
        self.encoder = encoder
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.use_encoder_features = use_encoder_features
        self.l1_lambda = l1_lambda
        
        # Projection from encoder to per-modality drift scores
        n_modalities = len(self.modalities)
        
        self.score_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_modalities),
            nn.Softplus(),  # Non-negative scores
        )
        
        # Fallback linear drift score
        self.linear_drift = DriftScoreModule(
            modality_dims=modality_dims,
            l1_lambda=l1_lambda,
        )
        
        # Attention for interpretability
        self.modality_attention = nn.Linear(embed_dim, n_modalities)
    
    def forward(
        self,
        drift_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Compute drift scores using encoder.
        
        Args:
            drift_dict: Dict mapping modality to drift tensor
            missing_mask: Optional missing mask
            
        Returns:
            Tuple of (scores_dict, encoder_representation)
        """
        if not self.use_encoder_features:
            # Fall back to linear
            scores = self.linear_drift(drift_dict)
            return scores, None
        
        # Get encoder representation
        pooled, tokens = self.encoder(drift_dict, missing_mask, return_all_tokens=False)
        
        # Project to per-modality scores
        # pooled: (batch, embed_dim)
        raw_scores = self.score_projection(pooled)  # (batch, n_modalities)
        
        # Apply attention for interpretable modality weights
        attention = F.softmax(self.modality_attention(pooled), dim=-1)
        
        # Modulate scores by attention
        modulated_scores = raw_scores * attention
        
        # Build scores dict
        scores_dict = {}
        for i, modality in enumerate(self.modalities):
            scores_dict[modality] = modulated_scores[:, i]
        
        return scores_dict, pooled
    
    def get_attention_weights(
        self,
        drift_dict: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Get attention weights for interpretability."""
        with torch.no_grad():
            pooled, _ = self.encoder(drift_dict, return_all_tokens=False)
            attention = F.softmax(self.modality_attention(pooled), dim=-1)
            
            return {
                mod: attention[0, i].item()
                for i, mod in enumerate(self.modalities)
            }
    
    def l1_penalty(self) -> torch.Tensor:
        """L1 penalty on projection weights."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for param in self.score_projection.parameters():
            penalty = penalty + torch.abs(param).sum()
        
        return self.l1_lambda * penalty
    
    def get_state_summary(self) -> dict:
        """Get summary of learned weights."""
        if self.use_encoder_features:
            return {"type": "encoder_backed", "modalities": self.modalities}
        else:
            return self.linear_drift.get_state_summary()


class ViralFlip(nn.Module):
    """Unified ViralFlip prediction model.
    
    Combines drift scoring, lag lattice hazard model, interactions,
    personalization, virus classification, confidence scoring,
    uncertainty estimation, and environment detection.
    
    Supports both lightweight (linear) and heavy (encoder-backed) modes
    for different deployment scenarios.
    """
    
    # Default modalities
    MODALITIES = [
        "voice", "cough", "tap", "gait_active", 
        "rppg", "light", "baro",  # Physiology
        "gps", "imu_passive", "screen",  # Behavior
    ]
    
    # Physiology modalities (for drift score)
    PHYSIOLOGY_MODALITIES = ["voice", "cough", "tap", "gait_active", "rppg", "light", "baro"]
    BEHAVIOR_MODALITIES = ["gps", "imu_passive", "screen"]
    
    def __init__(
        self,
        feature_dims: dict[str, int],
        horizons: list[int] = [24, 48, 72],
        max_lag: int = 12,
        # Encoder settings
        use_encoder: bool = False,
        encoder: Optional[nn.Module] = None,
        encoder_embed_dim: int = 128,
        # Regularization
        l1_lambda_drift: float = 0.01,
        l1_lambda_lattice: float = 0.01,
        # Interactions
        use_interactions: bool = False,
        interaction_pairs: Optional[list[tuple[str, str]]] = None,
        l1_lambda_interaction: float = 0.1,
        # Personalization
        use_personalization: bool = True,
        # Missing indicators
        use_missing_indicators: bool = True,
        # Confidence scoring
        confidence_gamma: Optional[dict[str, float]] = None,
        # Virus classification
        use_virus_classifier: bool = True,
        virus_classifier_hidden: int = 64,
        virus_classifier_dropout: float = 0.3,
        # Conformal prediction
        use_conformal: bool = True,
        conformal_alpha: float = 0.1,
        # Environment detection
        use_environment_detection: bool = True,
        n_environments: int = 4,
    ):
        """Initialize ViralFlip model.
        
        Args:
            feature_dims: Dict mapping modality to feature dimension.
            horizons: Prediction horizons in hours.
            max_lag: Maximum lag for lattice model (in bins).
            use_encoder: Whether to use pretrained encoder for drift scores.
            encoder: Optional pretrained encoder module.
            encoder_embed_dim: Encoder embedding dimension.
            l1_lambda_drift: L1 regularization for drift score weights.
            l1_lambda_lattice: L1 regularization for lattice weights.
            use_interactions: Whether to use interaction module.
            interaction_pairs: Optional custom interaction pairs.
            l1_lambda_interaction: L1 regularization for interactions.
            use_personalization: Whether to use personalization layer.
            use_missing_indicators: Whether to use missing indicators in lattice.
            confidence_gamma: Optional custom confidence scoring weights.
            use_virus_classifier: Whether to classify virus type.
            virus_classifier_hidden: Hidden dim for virus classifier.
            virus_classifier_dropout: Dropout rate for virus classifier.
            use_conformal: Whether to use conformal prediction.
            conformal_alpha: Conformal miscoverage rate.
            use_environment_detection: Whether to detect environments.
            n_environments: Number of environments for IRM.
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.horizons = horizons
        self.n_horizons = len(horizons)
        self.use_encoder = use_encoder
        
        # Filter to physiology modalities present in feature_dims
        self.physiology_modalities = [
            m for m in self.PHYSIOLOGY_MODALITIES if m in feature_dims
        ]
        self.behavior_modalities = [
            m for m in self.BEHAVIOR_MODALITIES if m in feature_dims
        ]
        
        physiology_dims = {m: feature_dims[m] for m in self.physiology_modalities}
        
        # Drift score module
        if use_encoder and encoder is not None:
            self.drift_score = EncoderBackedDriftScore(
                encoder=encoder,
                modality_dims=physiology_dims,
                embed_dim=encoder_embed_dim,
                use_encoder_features=True,
                l1_lambda=l1_lambda_drift,
            )
            self.encoder = encoder
        else:
            self.drift_score = DriftScoreModule(
                modality_dims=physiology_dims,
                l1_lambda=l1_lambda_drift,
            )
            self.encoder = None
        
        # Lag lattice model
        self.lag_lattice = LagLatticeHazardModel(
            n_modalities=len(self.physiology_modalities),
            horizons=horizons,
            max_lag=max_lag,
            l1_lambda=l1_lambda_lattice,
            use_missing_indicators=use_missing_indicators,
        )
        
        # Interactions (optional)
        self.use_interactions = use_interactions
        if use_interactions:
            self.interactions = InteractionModule(
                modality_names=self.physiology_modalities,
                horizons=horizons,
                interaction_pairs=interaction_pairs,
                l1_lambda=l1_lambda_interaction,
            )
        else:
            self.interactions = None
        
        # Personalization (optional)
        self.use_personalization = use_personalization
        if use_personalization:
            self.personalization = PersonalizationLayer(
                n_horizons=self.n_horizons,
            )
        else:
            self.personalization = None
        
        # Confidence scorer
        gamma = confidence_gamma or {}
        self.confidence_scorer = ConfidenceScorer(
            n_modalities=len(self.physiology_modalities),
            gamma0=gamma.get("gamma0", 0.0),
            gamma1=gamma.get("gamma1", 0.3),
            gamma2=gamma.get("gamma2", 0.2),
            gamma3=gamma.get("gamma3", 0.5),
        )
        
        # Virus classifier (optional)
        self.use_virus_classifier = use_virus_classifier
        if use_virus_classifier:
            self.virus_classifier = VirusClassifier(
                n_modalities=len(self.physiology_modalities),
                hidden_dim=virus_classifier_hidden,
                dropout=virus_classifier_dropout,
                n_classes=NUM_VIRUS_CLASSES,
            )
        else:
            self.virus_classifier = None
        
        # Uncertainty estimation head
        if use_encoder and encoder is not None:
            uncertainty_input_dim = encoder_embed_dim
        else:
            uncertainty_input_dim = len(self.physiology_modalities)
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(uncertainty_input_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.n_horizons),
            nn.Softplus(),  # Non-negative uncertainty
        )
        
        # Environment classifier (for IRM)
        self.use_environment_detection = use_environment_detection
        behavior_dim = sum(feature_dims.get(m, 0) for m in self.behavior_modalities)
        if use_environment_detection and behavior_dim > 0:
            self.env_classifier = nn.Sequential(
                nn.Linear(behavior_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_environments),
            )
            self.n_environments = n_environments
        else:
            self.env_classifier = None
            self.n_environments = 0
        
        # Conformal prediction settings
        self.use_conformal = use_conformal
        self.conformal_alpha = conformal_alpha
        # Conformal calibration data (populated during calibration)
        self._conformal_quantiles: Optional[dict[int, tuple[float, float]]] = None
    
    def forward(
        self,
        drift_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
        quality_scores: Optional[dict[str, torch.Tensor]] = None,
        user_ids: Optional[list[str]] = None,
        behavior_features: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            drift_dict: Dict mapping modality to drift tensor.
                       Shape: (batch, seq_len, n_features).
            missing_mask: Optional dict mapping modality to missing indicator.
                         Shape: (batch, seq_len).
            quality_scores: Optional dict mapping modality to quality score.
                           Shape: (batch, seq_len) or (batch,).
            user_ids: Optional list of user IDs for personalization.
            behavior_features: Optional behavior features for environment detection.
            return_features: Whether to return intermediate features.
            
        Returns:
            Tuple of (risk_probs, confidence_scores, virus_logits, encoder_features).
            risk_probs: (batch, seq_len, n_horizons).
            confidence_scores: (batch, seq_len) or (batch,).
            virus_logits: (batch, n_virus_classes) or None.
            encoder_features: (batch, embed_dim) or None.
        """
        # Compute drift scores
        encoder_features = None
        if isinstance(self.drift_score, EncoderBackedDriftScore):
            scores_dict, encoder_features = self.drift_score(drift_dict, missing_mask)
        else:
            scores_dict = self.drift_score(drift_dict)
        
        # Build score tensor in modality order
        batch_size = next(iter(drift_dict.values())).shape[0]
        seq_len = next(iter(drift_dict.values())).shape[1]
        device = next(iter(drift_dict.values())).device
        
        score_tensor = torch.zeros(
            batch_size, seq_len, len(self.physiology_modalities),
            device=device,
        )
        
        for i, modality in enumerate(self.physiology_modalities):
            if modality in scores_dict:
                score = scores_dict[modality]
                if score.dim() == 1:
                    score = score.unsqueeze(1).expand(-1, seq_len)
                score_tensor[:, :, i] = score
        
        # Build missing indicator tensor
        if missing_mask is not None:
            missing_tensor = torch.zeros_like(score_tensor)
            for i, modality in enumerate(self.physiology_modalities):
                if modality in missing_mask:
                    mask = missing_mask[modality]
                    if mask.dim() == 2:
                        missing_tensor[:, :, i] = mask.float()
                    else:
                        missing_tensor[:, :, i] = mask.unsqueeze(1).expand(-1, seq_len).float()
        else:
            missing_tensor = None
        
        # Lag lattice forward
        risk_probs = self.lag_lattice(score_tensor, missing_tensor)
        
        # Add interactions if enabled
        if self.use_interactions and self.interactions is not None:
            adjusted_slices = []
            for t in range(seq_len):
                t_scores = {
                    m: scores_dict[m][:, t] if scores_dict[m].dim() >= 2 else scores_dict[m]
                    for m in scores_dict
                }
                if t_scores:
                    interaction_contrib = self.interactions(t_scores)
                    adjusted = torch.sigmoid(
                        torch.logit(risk_probs[:, t, :].clamp(1e-7, 1-1e-7)) + interaction_contrib
                    )
                    adjusted_slices.append(adjusted)
                else:
                    adjusted_slices.append(risk_probs[:, t, :])
            risk_probs = torch.stack(adjusted_slices, dim=1)
        
        # Personalization
        if self.use_personalization and self.personalization is not None and user_ids is not None:
            risk_probs = self.personalization(risk_probs, user_ids)
        
        # Confidence scoring
        if quality_scores is not None and missing_mask is not None:
            quality_tensor = torch.zeros(
                batch_size, len(self.physiology_modalities),
                device=device,
            )
            for i, modality in enumerate(self.physiology_modalities):
                if modality in quality_scores:
                    q = quality_scores[modality]
                    if q.dim() == 2:
                        quality_tensor[:, i] = q.mean(dim=1)
                    else:
                        quality_tensor[:, i] = q
            
            missing_for_conf = missing_tensor[:, -1, :] if missing_tensor is not None else torch.zeros(
                batch_size, len(self.physiology_modalities), device=device
            )
            confidence_scores = self.confidence_scorer(missing_for_conf, quality_tensor)
        else:
            confidence_scores = torch.ones(batch_size, device=device)
        
        # Virus classification
        virus_logits = None
        if self.use_virus_classifier and self.virus_classifier is not None:
            drift_scores_for_virus = score_tensor[:, -1, :]  # (batch, n_modalities)
            mean_risk = risk_probs[:, -1, :].mean(dim=-1)  # (batch,)
            virus_logits = self.virus_classifier(drift_scores_for_virus, mean_risk)
        
        if return_features:
            return risk_probs, confidence_scores, virus_logits, encoder_features
        
        return risk_probs, confidence_scores, virus_logits, None
    
    def predict(
        self,
        drift_dict: dict[str, np.ndarray],
        missing_mask: Optional[dict[str, bool]] = None,
        quality_scores: Optional[dict[str, float]] = None,
        user_id: Optional[str] = None,
        alert_threshold: float = 0.3,
    ) -> ViralFlipOutput:
        """Convenience method for single-sample prediction.
        
        Args:
            drift_dict: Dict mapping modality to drift array.
                       Shape: (seq_len, n_features) or (n_features,).
            missing_mask: Optional dict mapping modality to missing flag.
            quality_scores: Optional dict mapping modality to quality score.
            user_id: Optional user ID for personalization.
            alert_threshold: Threshold for alerting.
            
        Returns:
            ViralFlipOutput with predictions and metadata.
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            drift_tensors = {}
            for m, arr in drift_dict.items():
                if isinstance(arr, np.ndarray):
                    t = torch.from_numpy(arr).float()
                else:
                    t = torch.tensor(arr).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # Add seq dim
                t = t.unsqueeze(0)  # Add batch dim
                drift_tensors[m] = t
            
            # Missing mask
            if missing_mask:
                missing_tensors = {
                    m: torch.tensor([[float(v)]]) for m, v in missing_mask.items()
                }
            else:
                missing_tensors = None
            
            # Quality
            if quality_scores:
                quality_tensors = {
                    m: torch.tensor([v]) for m, v in quality_scores.items()
                }
            else:
                quality_tensors = None
            
            # Forward
            risk_probs, confidence, virus_logits, encoder_features = self.forward(
                drift_tensors,
                missing_tensors,
                quality_tensors,
                [user_id] if user_id else None,
                return_features=True,
            )
            
            # Extract final timestep
            risk_probs_np = risk_probs[0, -1, :].cpu().numpy()
            confidence_val = confidence[0].item()
            
            # Compute drift scores for interpretability
            if isinstance(self.drift_score, EncoderBackedDriftScore):
                scores_dict, _ = self.drift_score(drift_tensors)
                drift_scores = {
                    m: scores_dict[m][0].item() if scores_dict[m].dim() > 0 else scores_dict[m].item()
                    for m in scores_dict
                }
            else:
                scores_dict = self.drift_score(drift_tensors)
                drift_scores = {
                    m: scores_dict[m][0, -1].item() if scores_dict[m].dim() > 1 else scores_dict[m][0].item()
                    for m in scores_dict
                }
            
            # Estimate uncertainty
            if encoder_features is not None:
                uncertainty = self.uncertainty_head(encoder_features)[0].cpu().numpy()
            else:
                # Use mean score for uncertainty estimation
                mean_scores = torch.tensor([list(drift_scores.values())]).float()
                uncertainty = self.uncertainty_head(mean_scores)[0].cpu().numpy()
            
            # Virus classification
            virus_probs = None
            virus_prediction = None
            virus_confidence = None
            
            if virus_logits is not None:
                virus_probs_tensor = F.softmax(virus_logits[0], dim=-1)
                virus_probs_np = virus_probs_tensor.cpu().numpy()
                
                virus_probs = {}
                for vt in VirusType:
                    virus_probs[VIRUS_NAMES[vt]] = float(virus_probs_np[vt.value])
                
                pred_idx = int(virus_probs_np.argmax())
                virus_prediction = VIRUS_NAMES[VirusType(pred_idx)]
                virus_confidence = float(virus_probs_np[pred_idx])
        
        # Build output
        risks = {h: float(risk_probs_np[i]) for i, h in enumerate(self.horizons)}
        
        # Confidences from both data quality and uncertainty
        base_confidence = confidence_val
        uncertainty_penalty = float(np.mean(uncertainty))
        confidences = {
            h: max(0.0, min(1.0, base_confidence * (1.0 - min(1.0, uncertainty[i]))))
            for i, h in enumerate(self.horizons)
        }
        
        raw_logits = {
            h: float(np.log(risk_probs_np[i] / (1 - risk_probs_np[i] + 1e-7)))
            for i, h in enumerate(self.horizons)
        }
        
        quality_summary = {
            "missing": missing_mask or {},
            "qualities": quality_scores or {},
        }
        
        # Conformal bounds
        conformal_lower = None
        conformal_upper = None
        if self.use_conformal and self._conformal_quantiles is not None:
            conformal_lower = {}
            conformal_upper = {}
            for h in self.horizons:
                if h in self._conformal_quantiles:
                    lo, hi = self._conformal_quantiles[h]
                    conformal_lower[h] = max(0.0, risks[h] - lo)
                    conformal_upper[h] = min(1.0, risks[h] + hi)
        
        # Alert determination
        should_alert = any(r >= alert_threshold for r in risks.values())
        alert_confidence = max(confidences.values()) if should_alert else 0.0
        
        # Uncertainty decomposition
        aleatoric = float(np.mean(uncertainty))
        epistemic = float(np.std([risks[h] for h in self.horizons]))
        
        return ViralFlipOutput(
            risks=risks,
            confidences=confidences,
            raw_logits=raw_logits,
            quality_summary=quality_summary,
            drift_scores=drift_scores,
            virus_probs=virus_probs,
            virus_prediction=virus_prediction,
            virus_confidence=virus_confidence,
            conformal_lower=conformal_lower,
            conformal_upper=conformal_upper,
            should_alert=should_alert,
            alert_confidence=alert_confidence,
            encoder_representation=encoder_features[0].cpu().numpy() if encoder_features is not None else None,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )
    
    def calibrate_conformal(
        self,
        calibration_predictions: dict[int, np.ndarray],
        calibration_labels: dict[int, np.ndarray],
    ) -> None:
        """Calibrate conformal prediction quantiles.
        
        Args:
            calibration_predictions: Dict mapping horizon to predicted probs
            calibration_labels: Dict mapping horizon to binary labels
        """
        self._conformal_quantiles = {}
        
        for h in self.horizons:
            if h not in calibration_predictions or h not in calibration_labels:
                continue
            
            preds = calibration_predictions[h]
            labels = calibration_labels[h]
            
            # Compute residuals
            residuals = np.abs(preds - labels)
            
            # Quantile for coverage
            q = np.quantile(residuals, 1 - self.conformal_alpha)
            
            self._conformal_quantiles[h] = (q, q)
    
    def get_environment_logits(
        self,
        behavior_features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get environment classification logits.
        
        Args:
            behavior_features: Behavior features tensor
            
        Returns:
            Environment logits (batch, n_environments) or None
        """
        if self.env_classifier is None:
            return None
        
        return self.env_classifier(behavior_features)
    
    def total_penalty(self) -> torch.Tensor:
        """Compute total regularization penalty."""
        penalty = self.drift_score.l1_penalty()
        penalty = penalty + self.lag_lattice.l1_penalty()
        
        if self.use_interactions and self.interactions is not None:
            penalty = penalty + self.interactions.l1_penalty()
        
        return penalty
    
    def load_pretrained_encoder(self, path: Path) -> None:
        """Load pretrained encoder weights.
        
        Args:
            path: Path to encoder checkpoint
        """
        if self.encoder is None:
            raise ValueError("Model not configured for pretrained encoder")
        
        state_dict = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(state_dict)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
    
    def get_modality_importance(self) -> dict[str, float]:
        """Get learned modality importance weights."""
        if isinstance(self.drift_score, EncoderBackedDriftScore):
            dummy_input = {
                m: torch.zeros(1, 1, self.feature_dims[m])
                for m in self.physiology_modalities
            }
            return self.drift_score.get_attention_weights(dummy_input)
        else:
            summary = self.drift_score.get_state_summary()
            return {m: info["total_weight"] for m, info in summary.items()}
    
    def get_state_summary(self) -> dict:
        """Get summary of learned model parameters."""
        summary = {
            "horizons": self.horizons,
            "modalities": self.physiology_modalities,
            "use_encoder": self.use_encoder,
            "use_virus_classifier": self.use_virus_classifier,
            "use_conformal": self.use_conformal,
            "use_environment_detection": self.use_environment_detection,
            "drift_score": self.drift_score.get_state_summary(),
            "lag_lattice": self.lag_lattice.get_state_summary(),
            "modality_importance": self.get_modality_importance(),
        }
        
        if self.use_interactions and self.interactions is not None:
            summary["interactions"] = self.interactions.get_state_summary()
        
        if self.use_personalization and self.personalization is not None:
            summary["personalization"] = self.personalization.get_state_dict()
        
        if self.use_virus_classifier:
            summary["virus_classifier"] = {
                "enabled": True,
                "n_classes": NUM_VIRUS_CLASSES,
                "class_names": [VIRUS_NAMES[vt] for vt in VirusType],
            }
        
        return summary


# Backwards compatibility aliases
ViralFlipModel = ViralFlip
ViralFlipX = ViralFlip
ViralFlipXOutput = ViralFlipOutput

