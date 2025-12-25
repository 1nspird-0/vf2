# ViralFlip

**Predict Viral Illness 24-72 Hours Before Symptoms Using Phone Sensors**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Data-Real%20Health%20Datasets-purple.svg" alt="Data">
</p>

> ⚠️ **SAFETY DISCLAIMER**: This is a research tool, NOT a medical device. Do not use for clinical decision-making.

---

## What ViralFlip Does

ViralFlip detects **pre-symptomatic illness** by analyzing subtle changes in your phone sensor data:

| Output | Description |
|--------|-------------|
| **24h Risk** | Probability of illness onset in next 24 hours |
| **48h Risk** | Probability of illness onset in next 48 hours |
| **72h Risk** | Probability of illness onset in next 72 hours |
| **Virus Type** | Classification: COVID, Flu, Cold, RSV, Pneumonia, or General |
| **Confidence** | How confident the model is in its prediction |
| **Uncertainty** | Aleatoric and epistemic uncertainty estimates |

### Example Prediction

```python
from viralflip import ViralFlip

model = ViralFlip(feature_dims={"voice": 30, "cough": 30, "rppg": 5})
output = model.predict(drift_dict={"voice": voice_features, "cough": cough_features})

print(output.risks)           # {24: 0.42, 48: 0.65, 72: 0.78}
print(output.virus_prediction)  # "Influenza"
print(output.should_alert)      # True
print(output.get_illness_summary())
# "Risk: 78% chance of illness in 72h | Likely: Influenza (73% conf) | ALERT"
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🦠 **Virus Classification** | Identifies COVID, Flu, Cold, RSV, Pneumonia, General Respiratory |
| 📱 **Passive Sensing** | Uses voice, cough, heart rate, gait, activity from phone sensors |
| 🧬 **Personal Baselines** | Learns what's "normal" for YOU, detects deviations |
| 🛡️ **Behavior Debiasing** | Removes false signals from travel, sleep changes, etc. |
| 🎯 **Conformal Prediction** | Calibrated uncertainty with coverage guarantees |
| 🔬 **Pretrained Encoder** | Optional transformer encoder for robust representations |
| ⚡ **GPU Optimized** | Mixed precision, gradient accumulation for fast training |

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA GPU recommended (RTX 3060+ or cloud GPU)
- ~10GB disk space for datasets

### Installation

```bash
git clone https://github.com/your-org/viralflip.git
cd viralflip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e ".[dev]"
```

### Train the Model

```bash
# One command training
python scripts/train.py --max-accuracy
```

This will:
1. ✅ Load processed health datasets
2. ✅ Train with optimized GPU settings
3. ✅ Save the best model to `runs/`

### Check Model Accuracy

```bash
# Evaluate on validation set
python scripts/check_accuracy.py --model runs/YOUR_RUN/best_model.pt

# Auto-find latest model
python scripts/check_accuracy.py --split test
```

---

## Using a Trained Model

```python
import torch
import numpy as np
from viralflip import ViralFlip

# 1. Create model with same settings as training
model = ViralFlip(
    feature_dims={"voice": 30, "cough": 30, "rppg": 5},
    horizons=[24, 48, 72],
    use_virus_classifier=True,
)

# 2. Load trained weights
checkpoint = torch.load("runs/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Make predictions
output = model.predict(
    drift_dict={
        "voice": np.random.randn(5, 30),  # (seq_len, features)
        "cough": np.random.randn(5, 30),
    },
    quality_scores={"voice": 0.9, "cough": 0.85},
    user_id="user_123",
)

# 4. Access results
print(f"24h Risk: {output.risks[24]:.1%}")
print(f"72h Risk: {output.risks[72]:.1%}")
print(f"Virus: {output.virus_prediction} ({output.virus_confidence:.0%})")
print(f"Alert: {output.should_alert}")
print(f"Uncertainty: {output.aleatoric_uncertainty:.2f}")
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `risks` | `dict[int, float]` | Risk probability per horizon |
| `confidences` | `dict[int, float]` | Confidence per horizon |
| `virus_prediction` | `str` | Predicted illness type |
| `virus_confidence` | `float` | Confidence in virus type |
| `should_alert` | `bool` | Whether to trigger alert |
| `drift_scores` | `dict[str, float]` | Per-modality drift scores |
| `aleatoric_uncertainty` | `float` | Data uncertainty |
| `epistemic_uncertainty` | `float` | Model uncertainty |
| `conformal_lower` | `dict` | Lower confidence bounds |
| `conformal_upper` | `dict` | Upper confidence bounds |

---

## Model Architecture

```
Phone Sensors → Feature Extraction → Personal Baseline Memory
                                            ↓
                              ┌─────────────────────────────┐
                              │   Drift Score Module (φ)    │
                              │  [Optional: Encoder-backed] │
                              └─────────────────────────────┘
                                            ↓
                              ┌─────────────────────────────┐
                              │  Behavior-Drift Debiasing   │
                              └─────────────────────────────┘
                                            ↓
                              ┌─────────────────────────────┐
                              │   Lag-Lattice Hazard Model  │
                              │   + Interaction Module      │
                              └─────────────────────────────┘
                                     ↙           ↘
                    ┌────────────────────┐  ┌────────────────────┐
                    │  Risk Prediction   │  │  Virus Classifier  │
                    │   (24/48/72h)      │  │    (7 classes)     │
                    └────────────────────┘  └────────────────────┘
                              ↓                       ↓
                    ┌────────────────────┐  ┌────────────────────┐
                    │ Conformal Bounds   │  │  Uncertainty Est.  │
                    └────────────────────┘  └────────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **ViralFlip** | Unified model combining all features |
| **DriftScoreModule** | Compresses modality drifts to scalar scores |
| **LagLatticeHazardModel** | Multi-horizon prediction with temporal structure |
| **InteractionModule** | Captures modality correlations (voice+cough) |
| **VirusClassifier** | MLP head for illness type classification |
| **ConfidenceScorer** | Data quality-based confidence |
| **PersonalizationLayer** | Per-user calibration |

---

## Project Structure

```
viralflip/
├── configs/
│   └── high_performance.yaml    # GPU-optimized training config
├── src/viralflip/
│   ├── model/
│   │   ├── viralflip.py         # Unified ViralFlip model
│   │   ├── drift_score.py       # Drift score compression
│   │   ├── lag_lattice.py       # Multi-horizon hazard model
│   │   ├── interactions.py      # Modality interactions
│   │   ├── personalization.py   # Per-user calibration
│   │   └── virus_types.py       # Virus type definitions
│   ├── data/
│   │   ├── dataset.py           # PyTorch dataset
│   │   └── real_data_loader.py  # Health dataset loaders
│   ├── train/
│   │   ├── trainer.py           # Training loop
│   │   ├── losses.py            # Focal loss + virus classification
│   │   └── build_sequences.py   # Sequence construction
│   ├── baseline/                # Personal Baseline Memory
│   ├── debias/                  # Behavior-Drift Debiasing
│   ├── pretrain/                # Masked autoencoder pretraining
│   ├── conformal/               # Conformal prediction
│   ├── explain/                 # Explainability engine
│   └── eval/                    # Metrics, calibration
├── scripts/
│   ├── train.py                 # Main training script
│   ├── check_accuracy.py        # Evaluate model accuracy
│   ├── train_viralflip_x.py     # Advanced training with pretraining
│   ├── evaluate.py              # Full evaluation pipeline
│   └── download_more_data.py    # Download health datasets
├── tests/                       # Unit tests
└── data/
    └── processed/               # Training-ready data
```

---

## Virus Types Detected

| Virus | Key Signals | Typical Onset |
|-------|-------------|---------------|
| **COVID-19** | Voice changes, dry cough, elevated HR | 48h gradual |
| **Influenza** | High fever (HR spike), fatigue, muscle aches | 12h rapid |
| **Common Cold** | Nasal voice, mild cough | 36h gradual |
| **RSV** | Wheezing cough, respiratory distress | 24h moderate |
| **Pneumonia** | Severe cough, high HR, significant fatigue | 48h gradual |
| **General** | Mixed respiratory symptoms | Variable |

---

## Configuration

High-performance config (`configs/high_performance.yaml`):

```yaml
training:
  batch_size: 128
  gradient_accumulation_steps: 2  # Effective batch = 256
  epochs: 300
  learning_rate: 0.0005
  use_amp: true  # Mixed precision (FP16)

model:
  max_lag_bins: 16           # 96h lookback
  use_interactions: true
  use_virus_classifier: true
  use_encoder: false         # Set true for pretrained encoder
  virus_classifier:
    hidden_dim: 128
    n_classes: 7

data:
  horizons: [24, 48, 72]
```

---

## GPU Requirements

| GPU | VRAM | Batch Size | Training Time |
|-----|------|------------|---------------|
| RTX 3060 | 12GB | 64 | ~4 hours |
| RTX 3080 | 10GB | 96 | ~2.5 hours |
| RTX 4090 | 24GB | 256 | ~1 hour |
| A100 | 40GB | 512 | ~30 min |

---

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **AUPRC** | >0.70 | Area under precision-recall curve |
| **AUROC** | >0.85 | Area under ROC curve |
| **Virus Accuracy** | >60% | Correct virus type classification |
| **Lead Time** | >24h | Early warning before symptoms |

---

## Advanced Features

### Pretrained Encoder Mode

```python
from viralflip import ViralFlip, MultimodalTimeSeriesEncoder

# Create encoder
encoder = MultimodalTimeSeriesEncoder(
    modality_dims={"voice": 30, "cough": 30},
    embed_dim=128,
    n_layers=4,
)

# Use encoder-backed model
model = ViralFlip(
    feature_dims={"voice": 30, "cough": 30},
    use_encoder=True,
    encoder=encoder,
    encoder_embed_dim=128,
)
```

### Conformal Prediction

```python
# Calibrate conformal predictor
model.calibrate_conformal(
    calibration_predictions={24: preds_24, 48: preds_48, 72: preds_72},
    calibration_labels={24: labels_24, 48: labels_48, 72: labels_72},
)

# Get prediction with uncertainty bounds
output = model.predict(drift_dict)
print(output.conformal_lower)  # {24: 0.35, 48: 0.55, 72: 0.68}
print(output.conformal_upper)  # {24: 0.49, 48: 0.75, 72: 0.88}
```

### Explainability

```python
from viralflip.explain import ExplanationEngine

engine = ExplanationEngine(model, top_k=5)
explanation = engine.explain(drift_dict, horizon=72)

print(explanation.summary_text())
# Risk (72h horizon): 78.2% (confidence: 85.3%)
# 
# Top Contributors:
#   1. voice (current): score=2.14, Δ=-15.3%
#        - f0_mean: z=2.8
#        - jitter: z=2.1
#   2. cough (current): score=1.87, Δ=-12.1%
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/viralflip --cov-report=html
```

---

## API Reference

### ViralFlip

```python
ViralFlip(
    feature_dims: dict[str, int],      # Modality -> feature dimension
    horizons: list[int] = [24, 48, 72],
    max_lag: int = 12,
    
    # Encoder options
    use_encoder: bool = False,
    encoder: nn.Module = None,
    encoder_embed_dim: int = 128,
    
    # Regularization
    l1_lambda_drift: float = 0.01,
    l1_lambda_lattice: float = 0.01,
    
    # Features
    use_interactions: bool = False,
    use_personalization: bool = True,
    use_virus_classifier: bool = True,
    use_conformal: bool = True,
    use_environment_detection: bool = True,
)
```

### ViralFlipOutput

```python
@dataclass
class ViralFlipOutput:
    risks: dict[int, float]           # horizon -> probability
    confidences: dict[int, float]     # horizon -> confidence
    raw_logits: dict[int, float]
    drift_scores: dict[str, float]    # modality -> score
    
    # Virus classification
    virus_probs: dict[str, float]     # virus_name -> probability
    virus_prediction: str             # Most likely virus
    virus_confidence: float
    
    # Uncertainty
    conformal_lower: dict[int, float]
    conformal_upper: dict[int, float]
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    
    # Alerting
    should_alert: bool
    alert_confidence: float
```

---

## Citation

```bibtex
@software{viralflip2024,
  title={ViralFlip: Predictive Viral Illness Forecasting with Phone Sensors},
  year={2024},
  url={https://github.com/your-org/viralflip}
}
```

---

## License

MIT License - see LICENSE file.

---

## Acknowledgments

Built on research in mobile health sensing and digital biomarkers. Thanks to the creators of COUGHVID, Coswara, Virufy, DiCOVA, FluSense, and other public health datasets.
