#!/usr/bin/env python3
"""Evaluate trained ViralFlip model.

Usage:
    python scripts/evaluate.py --run_dir runs/20241224_120000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from viralflip.eval.metrics import evaluate_model, EvaluationResults
from viralflip.eval.calibration import compute_calibration_metrics
from viralflip.model.viralflip import ViralFlip
from viralflip.baseline.pbm import PersonalBaselineMemory
from viralflip.debias.ridge import BehaviorDriftDebiaser
from viralflip.utils.io import load_config, load_pickle, ensure_dir
from viralflip.utils.logging import setup_logging, get_logger
from viralflip.utils.seed import set_seed


logger = get_logger(__name__)


FEATURE_DIMS = {
    "voice": 24,
    "cough": 6,
    "tap": 6,
    "gait_active": 8,
    "rppg": 5,
    "imu_passive": 6,
    "gps": 5,
    "light": 4,
    "baro": 4,
    "screen": 5,
}


def load_model(run_dir: Path, device: str = "cpu") -> tuple[ViralFlip, dict]:
    """Load trained model from run directory.
    
    Args:
        run_dir: Path to run directory.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, config).
    """
    # Load config
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_cfg = config.get("model", {})
    model = ViralFlip(
        feature_dims=FEATURE_DIMS,
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        max_lag=model_cfg.get("max_lag_bins", 12),
        use_interactions=model_cfg.get("use_interactions", False),
        use_missing_indicators=model_cfg.get("use_missing_indicators", True),
        use_personalization=config.get("personalization", {}).get("enabled", True),
    )
    
    # Load weights
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, config


def load_test_data(data_dir: Path, test_user_ids: list[str]) -> dict:
    """Load test data for evaluation.
    
    Args:
        data_dir: Data directory.
        test_user_ids: List of test user IDs.
        
    Returns:
        Dict with test data.
    """
    from viralflip.scripts.train import load_synthetic_data
    
    all_users = load_synthetic_data(data_dir)
    test_users = [u for u in all_users if u.user_id in test_user_ids]
    
    return test_users


def evaluate_on_test(
    model: ViralFlip,
    test_users: list,
    pbm: PersonalBaselineMemory,
    debiaser: BehaviorDriftDebiaser,
    config: dict,
    device: str = "cpu",
) -> dict:
    """Evaluate model on test set.
    
    Args:
        model: Trained model.
        test_users: List of test user data.
        pbm: Personal baseline memory.
        debiaser: Behavior drift debiaser.
        config: Configuration dict.
        device: Evaluation device.
        
    Returns:
        Dict with evaluation results.
    """
    model.eval()
    horizons = model.horizons
    
    # Collect predictions and labels
    all_predictions = {h: [] for h in horizons}
    all_labels = {h: [] for h in horizons}
    all_onset_indices = []
    
    with torch.no_grad():
        for user_data in test_users:
            onset_offset = len(all_predictions[horizons[0]])
            
            for bin_data in user_data.bins:
                if bin_data.in_washout:
                    continue
                
                # Compute drifts
                drift_dict = {}
                missing_mask = {}
                quality_scores = {}
                
                for modality in FEATURE_DIMS:
                    if modality in bin_data.features and not bin_data.missing.get(modality, False):
                        drift = pbm.compute_drift(
                            user_data.user_id,
                            modality,
                            bin_data.features[modality],
                        )
                        drift_dict[modality] = drift
                        missing_mask[modality] = False
                        quality_scores[modality] = float(bin_data.quality.get(modality, [1.0])[0])
                    else:
                        missing_mask[modality] = True
                
                # Apply debiasing
                if debiaser.is_fitted and drift_dict:
                    behavior_drift = {m: drift_dict[m] for m in debiaser.behavior_blocks if m in drift_dict}
                    physiology_drift = {m: drift_dict[m] for m in debiaser.physiology_blocks if m in drift_dict}
                    debiased = debiaser.debias(behavior_drift, physiology_drift)
                    for m, d in debiased.items():
                        drift_dict[m] = d
                
                if not drift_dict:
                    continue
                
                # Get prediction
                output = model.predict_single(
                    drift_dict, missing_mask, quality_scores, user_data.user_id
                )
                
                for h in horizons:
                    all_predictions[h].append(output.risks[h])
                    all_labels[h].append(bin_data.labels.get(h, 0))
            
            # Track onset indices (adjusted for offset)
            for onset in user_data.onset_timestamps:
                all_onset_indices.append(int(onset) + onset_offset)
    
    # Convert to arrays
    for h in horizons:
        all_predictions[h] = np.array(all_predictions[h])
        all_labels[h] = np.array(all_labels[h])
    
    # Evaluate
    results = evaluate_model(
        predictions=all_predictions,
        labels=all_labels,
        onset_indices=all_onset_indices,
        horizons=horizons,
    )
    
    # Calibration metrics
    calibration = {}
    for h in horizons:
        cal = compute_calibration_metrics(all_labels[h], all_predictions[h])
        calibration[h] = cal.to_dict()
    
    return {
        "metrics": results.to_dict(),
        "calibration": calibration,
        "n_samples": len(all_predictions[horizons[0]]),
        "n_positive": {h: int(all_labels[h].sum()) for h in horizons},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViralFlip model")
    parser.add_argument("--run_dir", "-r", type=str, required=True,
                       help="Path to run directory")
    parser.add_argument("--data", "-d", type=str, default=None,
                       help="Data directory (default: from config)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Evaluation device")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file (default: run_dir/evaluation.json)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    setup_logging()
    
    # Load model and config
    model, config = load_model(run_dir, args.device)
    
    # Determine data directory
    if args.data is not None:
        data_dir = Path(args.data)
    else:
        data_dir = Path(config.get("paths", {}).get("data_dir", "data")) / "synthetic"
    
    # Load PBM and debiaser
    pbm = PersonalBaselineMemory(feature_dims=FEATURE_DIMS)
    pbm_state = load_pickle(run_dir / "pbm_state.pkl")
    pbm.load_state_dict(pbm_state)
    
    debiaser = BehaviorDriftDebiaser(feature_dims=FEATURE_DIMS)
    debiaser_state = load_pickle(run_dir / "debiaser_state.pkl")
    debiaser.load_state_dict(debiaser_state)
    
    # Get test user IDs
    seed = config.get("training", {}).get("seed", 42)
    splits_cfg = config.get("splits", {})
    
    # Load all data to get user IDs
    from viralflip.scripts.train import load_synthetic_data
    all_users = load_synthetic_data(data_dir)
    
    rng = np.random.default_rng(seed)
    user_ids = [u.user_id for u in all_users]
    shuffled = rng.permutation(user_ids)
    
    n_train = int(len(all_users) * splits_cfg.get("train_user_frac", 0.7))
    n_val = int(len(all_users) * splits_cfg.get("val_user_frac", 0.15))
    
    test_ids = set(shuffled[n_train + n_val:])
    test_users = [u for u in all_users if u.user_id in test_ids]
    
    logger.info(f"Evaluating on {len(test_users)} test users")
    
    # Evaluate
    results = evaluate_on_test(
        model, test_users, pbm, debiaser, config, args.device
    )
    
    # Print results
    metrics = results["metrics"]
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Samples: {results['n_samples']}")
    logger.info(f"Positives: {results['n_positive']}")
    logger.info(f"\nAUPRC: {metrics['auprc']}")
    logger.info(f"AUROC: {metrics['auroc']}")
    logger.info(f"Lead Time Frac: {metrics['lead_time_frac']}")
    logger.info(f"False Alarms/Week: {metrics['false_alarms_per_week']}")
    logger.info(f"\nMean AUPRC: {metrics['auprc_mean']:.4f}")
    logger.info(f"Mean AUROC: {metrics['auroc_mean']:.4f}")
    
    # Save results
    output_path = Path(args.output) if args.output else run_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

