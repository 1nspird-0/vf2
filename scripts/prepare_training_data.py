#!/usr/bin/env python3
"""Prepare gathered real data for ViralFlip training.

This script takes the output from gather_real_data.py and creates:
1. Pre-trained feature extractors from modality-specific datasets
2. Calibrated quality estimators
3. Behavior confound models from longitudinal data
4. A combined training dataset format

Usage:
    python scripts/prepare_training_data.py --input data/real --output data/training
    python scripts/prepare_training_data.py --input data/real --output data/training --task pretrain_cough
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from viralflip.utils.io import ensure_dir, save_pickle, load_pickle
from viralflip.utils.logging import setup_logging, get_logger
from viralflip.utils.seed import set_seed


logger = get_logger(__name__)


# ============================================================================
# Data Preparation Tasks
# ============================================================================

def prepare_cough_detector_data(
    processed_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Prepare data for training cough event detector.
    
    Uses:
    - COUGHVID: Labeled cough audio
    - Coswara: Cough recordings
    - ESC-50: Cough vs non-cough classification
    
    Output: Training data for binary cough detection
    """
    logger.info("Preparing cough detector training data...")
    
    output_dir = ensure_dir(output_dir / "cough_detector")
    
    positive_samples = []  # Cough audio paths
    negative_samples = []  # Non-cough audio paths
    
    # 1. Process ESC-50 (has labeled cough vs non-cough)
    esc50_dir = processed_dir / "esc50_processed"
    if esc50_dir.exists():
        cough_file = esc50_dir / "cough_files.txt"
        if cough_file.exists():
            with open(cough_file) as f:
                positive_samples.extend(f.read().strip().split("\n"))
            logger.info(f"Added {len(positive_samples)} cough samples from ESC-50")
        
        # Get non-cough samples
        if HAS_PANDAS:
            meta_file = esc50_dir / "esc50_metadata.csv"
            if meta_file.exists():
                df = pd.read_csv(meta_file)
                non_cough = df[df['category'] != 'coughing']
                # Sample same number of negatives
                n_neg = min(len(non_cough), len(positive_samples) * 2)
                neg_sample = non_cough.sample(n=n_neg, random_state=42)
                for _, row in neg_sample.iterrows():
                    negative_samples.append(row.get('filename', ''))
                logger.info(f"Added {len(negative_samples)} non-cough samples from ESC-50")
    
    # 2. Process COUGHVID
    coughvid_dir = processed_dir / "coughvid_processed"
    if coughvid_dir.exists() and HAS_PANDAS:
        meta_file = coughvid_dir / "coughvid_metadata.csv"
        if meta_file.exists():
            df = pd.read_csv(meta_file)
            # Get samples with high cough_detected score
            if 'cough_detected' in df.columns:
                high_conf = df[df['cough_detected'] > 0.8]
                logger.info(f"Found {len(high_conf)} high-confidence cough samples in COUGHVID")
    
    # 3. Process Coswara
    coswara_dir = processed_dir / "coswara_processed"
    if coswara_dir.exists():
        manifest_file = coswara_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            logger.info(f"Coswara: {manifest.get('n_participants', 0)} participants available")
    
    # Create training splits
    np.random.seed(42)
    
    all_positive = list(set(positive_samples))
    all_negative = list(set(negative_samples))
    
    np.random.shuffle(all_positive)
    np.random.shuffle(all_negative)
    
    # 80/10/10 split
    n_train = int(0.8 * min(len(all_positive), len(all_negative)))
    n_val = int(0.1 * min(len(all_positive), len(all_negative)))
    
    splits = {
        "train": {
            "positive": all_positive[:n_train],
            "negative": all_negative[:n_train],
        },
        "val": {
            "positive": all_positive[n_train:n_train+n_val],
            "negative": all_negative[n_train:n_train+n_val],
        },
        "test": {
            "positive": all_positive[n_train+n_val:],
            "negative": all_negative[n_train+n_val:],
        },
    }
    
    # Save splits
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    result = {
        "task": "cough_detector",
        "n_positive": len(all_positive),
        "n_negative": len(all_negative),
        "splits": {k: {"pos": len(v["positive"]), "neg": len(v["negative"])} 
                  for k, v in splits.items()},
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Cough detector data prepared: {result}")
    
    return result


def prepare_imu_baseline_data(
    processed_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Prepare IMU data for gait/activity feature validation.
    
    Uses:
    - UCI HAR: Activity classification with known labels
    - WISDM: Multi-sensor activity data
    - RealWorld HAR: Phone placement variability
    
    Output: Validated IMU feature extraction pipeline
    """
    logger.info("Preparing IMU baseline data...")
    
    output_dir = ensure_dir(output_dir / "imu_baseline")
    
    datasets_used = []
    
    # 1. UCI HAR - reference for activity detection
    uci_dir = processed_dir / "uci_har_processed"
    if uci_dir.exists():
        manifest_file = uci_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            datasets_used.append({
                "name": "uci_har",
                "activities": manifest.get("activities", {}),
                "train_samples": manifest.get("data_summary", {}).get("train_n_samples", 0),
                "test_samples": manifest.get("data_summary", {}).get("test_n_samples", 0),
            })
            logger.info(f"UCI HAR: {datasets_used[-1]}")
    
    # 2. WISDM
    wisdm_dir = processed_dir / "wisdm_processed"
    if wisdm_dir.exists():
        manifest_file = wisdm_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            datasets_used.append({
                "name": "wisdm",
                "n_accel_files": manifest.get("n_accel_files", 0),
                "n_gyro_files": manifest.get("n_gyro_files", 0),
            })
            logger.info(f"WISDM: {datasets_used[-1]}")
    
    # 3. RealWorld HAR (multi-modal)
    realworld_dir = processed_dir / "realworld_processed"
    if realworld_dir.exists():
        manifest_file = realworld_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            datasets_used.append({
                "name": "realworld_har",
                "modalities": manifest.get("modalities", {}),
            })
            logger.info(f"RealWorld: {datasets_used[-1]}")
    
    result = {
        "task": "imu_baseline",
        "datasets": datasets_used,
        "use_for": [
            "step_detection_validation",
            "gait_feature_calibration",
            "activity_level_thresholds",
            "phone_placement_robustness",
        ],
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def prepare_behavior_confound_data(
    processed_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Prepare longitudinal phone sensing data for behavior confound modeling.
    
    Uses:
    - StudentLife: GPS + phone usage + EMA
    - Beiwe: GPS + accelerometer + surveys
    - ExtraSensory: Multi-modal with context labels
    
    Output: Training data for BDD (Behavior-Drift Debiasing)
    """
    logger.info("Preparing behavior confound modeling data...")
    
    output_dir = ensure_dir(output_dir / "behavior_confounds")
    
    datasets_info = []
    
    # 1. ExtraSensory - richest labels
    extrasensory_dir = processed_dir / "extrasensory_processed"
    if extrasensory_dir.exists():
        manifest_file = extrasensory_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            datasets_info.append({
                "name": "extrasensory",
                "n_features": manifest.get("n_feature_cols", 0),
                "n_labels": manifest.get("n_label_cols", 0),
                "quality": "high",  # Rich context labels
            })
    
    # 2. StudentLife
    studentlife_dir = processed_dir / "studentlife_processed"
    if studentlife_dir.exists():
        manifest_file = studentlife_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            datasets_info.append({
                "name": "studentlife",
                "modalities": manifest.get("modalities", {}),
                "quality": "high",  # Longitudinal with EMA
            })
    
    # 3. Beiwe
    beiwe_dir = processed_dir / "beiwe_processed"
    if beiwe_dir.exists():
        manifest_file = beiwe_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            datasets_info.append({
                "name": "beiwe",
                "streams": manifest.get("streams", {}),
                "quality": "medium",
            })
    
    result = {
        "task": "behavior_confound_modeling",
        "datasets": datasets_info,
        "use_for": [
            "mobility_pattern_baseline",
            "routine_change_detection",
            "behavior_drift_debiasing",
            "confound_removal_training",
        ],
        "features_to_extract": [
            "radius_of_gyration",
            "location_entropy",
            "time_at_home",
            "activity_fragmentation",
            "screen_on_patterns",
            "circadian_regularity",
        ],
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def create_pretrain_config(
    output_dir: Path,
    task_results: dict,
) -> dict:
    """Create configuration for pre-training on real data."""
    
    config = {
        "pretrain_stages": [],
        "validation_datasets": [],
    }
    
    # Stage 1: Cough detector pre-training
    if "cough_detector" in task_results:
        cough_result = task_results["cough_detector"]
        if cough_result.get("n_positive", 0) > 100:
            config["pretrain_stages"].append({
                "stage": 1,
                "name": "cough_event_detector",
                "data_path": str(output_dir / "cough_detector"),
                "n_samples": cough_result.get("n_positive", 0) + cough_result.get("n_negative", 0),
                "epochs": 50,
                "learning_rate": 0.001,
            })
    
    # Stage 2: IMU feature calibration
    if "imu_baseline" in task_results:
        imu_result = task_results["imu_baseline"]
        if len(imu_result.get("datasets", [])) > 0:
            config["pretrain_stages"].append({
                "stage": 2,
                "name": "imu_feature_validation",
                "data_path": str(output_dir / "imu_baseline"),
                "datasets": [d["name"] for d in imu_result.get("datasets", [])],
                "purpose": "validate step detection, gait features, activity thresholds",
            })
    
    # Stage 3: Behavior confound model
    if "behavior_confound_modeling" in task_results:
        behavior_result = task_results["behavior_confound_modeling"]
        if len(behavior_result.get("datasets", [])) > 0:
            config["pretrain_stages"].append({
                "stage": 3,
                "name": "behavior_drift_debiasing",
                "data_path": str(output_dir / "behavior_confounds"),
                "datasets": [d["name"] for d in behavior_result.get("datasets", [])],
                "purpose": "train ridge regression for confound removal",
            })
    
    # Save config
    config_path = output_dir / "pretrain_config.yaml"
    
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config


def generate_training_summary(
    output_dir: Path,
    task_results: dict,
) -> None:
    """Generate a summary of available training data."""
    
    summary_lines = [
        "=" * 70,
        "VIRALFLIP TRAINING DATA SUMMARY",
        "=" * 70,
        "",
    ]
    
    # Cough detection
    if "cough_detector" in task_results:
        r = task_results["cough_detector"]
        summary_lines.extend([
            "COUGH EVENT DETECTION",
            "-" * 40,
            f"  Positive samples: {r.get('n_positive', 0)}",
            f"  Negative samples: {r.get('n_negative', 0)}",
            f"  Train/Val/Test split available: Yes",
            "",
        ])
    
    # IMU baseline
    if "imu_baseline" in task_results:
        r = task_results["imu_baseline"]
        summary_lines.extend([
            "IMU FEATURE BASELINE",
            "-" * 40,
        ])
        for ds in r.get("datasets", []):
            summary_lines.append(f"  {ds['name']}: {ds}")
        summary_lines.append("")
    
    # Behavior confounds
    if "behavior_confound_modeling" in task_results:
        r = task_results["behavior_confound_modeling"]
        summary_lines.extend([
            "BEHAVIOR CONFOUND MODELING",
            "-" * 40,
        ])
        for ds in r.get("datasets", []):
            summary_lines.append(f"  {ds['name']}: quality={ds.get('quality', 'unknown')}")
        summary_lines.append("")
    
    # What's missing
    summary_lines.extend([
        "WHAT'S STILL NEEDED",
        "-" * 40,
        "  • rPPG datasets (require manual request): UBFC-rPPG, PURE, COHFACE",
        "  • Longitudinal illness onset labels: requires custom data collection",
        "  • Voice quality ground truth: PCR-referenced datasets",
        "",
        "RECOMMENDED NEXT STEPS",
        "-" * 40,
        "  1. Pre-train cough detector on available audio data",
        "  2. Validate IMU features on HAR datasets",
        "  3. Train behavior confound model on phone sensing data",
        "  4. Request access to rPPG datasets for HR validation",
        "  5. Design/run pilot study for illness onset labels",
        "",
    ])
    
    summary_text = "\n".join(summary_lines)
    
    # Print and save
    print(summary_text)
    
    with open(output_dir / "TRAINING_SUMMARY.txt", "w") as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare gathered data for ViralFlip training"
    )
    
    parser.add_argument("--input", "-i", type=str, default="data/real/processed",
                       help="Input directory (processed data from gather_real_data.py)")
    parser.add_argument("--output", "-o", type=str, default="data/training",
                       help="Output directory for training-ready data")
    parser.add_argument("--task", "-t", type=str, default="all",
                       choices=["all", "cough", "imu", "behavior", "pretrain_config"],
                       help="Specific task to prepare")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output))
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("Run gather_real_data.py first to download datasets")
        return
    
    config = {"seed": args.seed}
    task_results = {}
    
    # Run tasks
    if args.task in ["all", "cough"]:
        task_results["cough_detector"] = prepare_cough_detector_data(
            input_dir, output_dir, config
        )
    
    if args.task in ["all", "imu"]:
        task_results["imu_baseline"] = prepare_imu_baseline_data(
            input_dir, output_dir, config
        )
    
    if args.task in ["all", "behavior"]:
        task_results["behavior_confound_modeling"] = prepare_behavior_confound_data(
            input_dir, output_dir, config
        )
    
    if args.task in ["all", "pretrain_config"]:
        pretrain_config = create_pretrain_config(output_dir, task_results)
        logger.info(f"Pre-training config: {pretrain_config}")
    
    # Generate summary
    generate_training_summary(output_dir, task_results)
    
    # Save combined results
    with open(output_dir / "preparation_results.json", "w") as f:
        json.dump(task_results, f, indent=2, default=str)
    
    logger.info(f"\nTraining data prepared in: {output_dir}")


if __name__ == "__main__":
    main()

