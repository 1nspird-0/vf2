#!/usr/bin/env python3
"""Train ViralFlip model on real health datasets.

Usage:
    # First, download and prepare real data:
    python scripts/download_more_data.py --health --parallel 4
    python scripts/prepare_real_data.py
    
    # Then train:
    python scripts/train.py --config configs/high_performance.yaml
    python scripts/train.py --max-accuracy
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import yaml

from viralflip.model.viralflip import ViralFlip
from viralflip.model.virus_types import VirusType, NUM_VIRUS_CLASSES, VIRUS_NAMES
from viralflip.data.dataset import ViralFlipDataset, collate_viralflip_batch
from viralflip.train.trainer import ViralFlipTrainer
from viralflip.utils.io import load_config, save_pickle, ensure_dir
from viralflip.utils.logging import setup_logging, get_logger
from viralflip.utils.seed import set_seed

try:
    from viralflip.utils.gpu import setup_gpu, print_gpu_info, get_gpu_info
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


logger = get_logger(__name__)


# Feature dimensions for real data (audio-based)
FEATURE_DIMS = {
    "voice": 30,      # MFCCs + spectral features
    "cough": 30,
    "breathing": 20,
    "rppg": 5,
    "activity": 6,
}

# Physiology modalities used for drift detection
PHYSIOLOGY_MODALITIES = ["voice", "cough", "breathing", "rppg"]


def check_data_exists(data_dir: Path) -> bool:
    """Check if processed data exists."""
    required_files = ["train.json", "val.json", "metadata.yaml"]
    return all((data_dir / f).exists() for f in required_files)


def print_data_stats(data_dir: Path):
    """Print dataset statistics."""
    metadata_path = data_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Train samples: {metadata.get('n_train', 0):,}")
        print(f"Val samples: {metadata.get('n_val', 0):,}")
        print(f"Test samples: {metadata.get('n_test', 0):,}")
        
        virus_counts = metadata.get("virus_counts", {})
        if virus_counts:
            print("\nVirus type distribution:")
            for vt, count in sorted(virus_counts.items(), key=lambda x: -x[1]):
                if count > 0:
                    print(f"  {vt}: {count}")
        
        print(f"\nDatasets used: {metadata.get('datasets_used', [])}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train ViralFlip model on real health data")
    parser.add_argument("--config", "-c", type=str, default="configs/high_performance.yaml",
                       help="Path to config file")
    parser.add_argument("--data", "-d", type=str, default="data/processed/",
                       help="Path to processed data directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: runs/TIMESTAMP)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'auto', 'cpu', 'cuda'")
    parser.add_argument("--max-accuracy", action="store_true",
                       help="Use high-performance config for maximum accuracy")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--gpu-info", action="store_true",
                       help="Print GPU info and exit")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    
    args = parser.parse_args()
    
    # Print GPU info if requested
    if args.gpu_info:
        if GPU_UTILS_AVAILABLE:
            print_gpu_info()
        else:
            print("GPU utilities not available")
            if torch.cuda.is_available():
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return
    
    # Override config for max accuracy mode
    if args.max_accuracy:
        args.config = "configs/high_performance.yaml"
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script
        config_path = Path(__file__).parent.parent / args.config
    
    if not config_path.exists():
        print(f"Config not found: {args.config}")
        print("Using default configuration...")
        config = {
            "training": {
                "epochs": 100,
                "batch_size": 64,
                "learning_rate": 0.001,
                "use_amp": True,
            },
            "model": {
                "max_lag_bins": 12,
                "use_interactions": True,
                "use_virus_classifier": True,
            },
            "data": {
                "horizons": [24, 48, 72],
            }
        }
    else:
        config = load_config(str(config_path))
    
    # Apply command-line overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    
    # Check for processed data
    data_dir = Path(args.data)
    if not check_data_exists(data_dir):
        print("\n" + "!"*60)
        print("ERROR: Processed data not found!")
        print("!"*60)
        print(f"\nExpected data in: {data_dir}")
        print("\nPlease run the following commands first:")
        print("\n  1. Download health datasets:")
        print("     python scripts/download_more_data.py --health --parallel 4")
        print("\n  2. Prepare data for training:")
        print("     python scripts/prepare_real_data.py")
        print("\nThen re-run this training script.")
        return
    
    # Print data stats
    print_data_stats(data_dir)
    
    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get("paths", {}).get("output_dir", "runs")) / timestamp
    else:
        output_dir = Path(args.output)
    
    output_dir = ensure_dir(output_dir)
    
    # Setup logging
    setup_logging(log_file=output_dir / "train.log")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using config: {args.config}")
    
    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    train_dataset = ViralFlipDataset(
        data_path=data_dir,
        split="train",
        feature_dims=FEATURE_DIMS,
        augment=True,
    )
    
    val_dataset = ViralFlipDataset(
        data_path=data_dir,
        split="val",
        feature_dims=FEATURE_DIMS,
        augment=False,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model_cfg = config.get("model", {})
    virus_cfg = model_cfg.get("virus_classifier", {})
    
    model = ViralFlip(
        feature_dims=FEATURE_DIMS,
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        max_lag=model_cfg.get("max_lag_bins", 12),
        l1_lambda_drift=config.get("drift_score", {}).get("l1_lambda", 0.01),
        l1_lambda_lattice=model_cfg.get("l1_lambda_w", 0.01),
        use_interactions=model_cfg.get("use_interactions", True),
        use_missing_indicators=model_cfg.get("use_missing_indicators", True),
        use_personalization=config.get("personalization", {}).get("enabled", True),
        use_virus_classifier=model_cfg.get("use_virus_classifier", True),
        virus_classifier_hidden=virus_cfg.get("hidden_dim", 128),
        virus_classifier_dropout=virus_cfg.get("dropout", 0.3),
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Print GPU info
    if GPU_UTILS_AVAILABLE and args.device != "cpu":
        print_gpu_info()
    elif torch.cuda.is_available() and args.device != "cpu":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create trainer
    trainer = ViralFlipTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=args.device,
    )
    
    # Resume from checkpoint if specified
    resume_path = Path(args.resume) if args.resume else None
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"  Epochs: {config.get('training', {}).get('epochs', 100)}")
    print(f"  Batch size: {config.get('training', {}).get('batch_size', 64)}")
    print(f"  Learning rate: {config.get('training', {}).get('learning_rate', 0.001)}")
    print(f"  Mixed precision: {config.get('training', {}).get('use_amp', False)}")
    print(f"  Virus classifier: {model_cfg.get('use_virus_classifier', True)}")
    print("="*60 + "\n")
    
    history = trainer.train(train_dataset, val_dataset, resume_from=resume_path)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"  Best validation AUPRC: {max(history['val_auprc']):.4f}")
    if 'virus_accuracy' in history and history.get('virus_accuracy'):
        print(f"  Best virus accuracy: {max(history.get('virus_accuracy', [0])):.2%}")
    print(f"  Model saved to: {output_dir}")
    print("="*60)
    
    # Print virus classification capabilities
    print("\nVirus types the model can detect:")
    for vt in VirusType:
        if vt != VirusType.HEALTHY:
            print(f"  - {VIRUS_NAMES[vt]}")


if __name__ == "__main__":
    main()
