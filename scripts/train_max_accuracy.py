#!/usr/bin/env python3
"""One-command maximum accuracy training for ViralFlip.

This script runs the full pipeline with REAL health data:
1. Downloads all health datasets (with resume support)
2. Prepares and processes the real data
3. Trains with high-performance config
4. Outputs comprehensive metrics

Usage:
    python scripts/train_max_accuracy.py
    python scripts/train_max_accuracy.py --skip-download  # Skip data download
    python scripts/train_max_accuracy.py --resume         # Resume from last checkpoint
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str, allow_fail: bool = False) -> bool:
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n[OK] {description} completed in {elapsed:.1f}s")
        return True
    else:
        if allow_fail:
            print(f"\n[WARN] {description} had issues (continuing...)")
            return True
        else:
            print(f"\n[ERROR] {description} failed (exit code {result.returncode})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Full max-accuracy training pipeline with REAL data")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--skip-prepare", action="store_true",
                       help="Skip data preparation (use existing processed data)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: runs/max_accuracy_TIMESTAMP)")
    parser.add_argument("--parallel", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of training epochs")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/max_accuracy_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          VIRALFLIP MAXIMUM ACCURACY TRAINING                 ║
║                   (REAL DATA ONLY)                           ║
╠══════════════════════════════════════════════════════════════╣
║  Output: {str(output_dir):<50} ║
║  Config: configs/high_performance.yaml                       ║
║                                                              ║
║  Optimized for:                                              ║
║  - RTX 5070 (12GB VRAM)                                      ║
║  - Mixed precision (FP16)                                    ║
║  - Large batch (128 x 2 accum = 256 effective)              ║
║  - Cosine warmup LR schedule                                 ║
║  - Full interaction modeling                                 ║
║  - Virus type classification (7 classes)                     ║
║                                                              ║
║  Real Datasets Used:                                         ║
║  - COUGHVID: 25K+ COVID coughs with test results            ║
║  - Coswara: Breathing/cough/voice + COVID labels            ║
║  - Virufy: PCR-confirmed COVID coughs                       ║
║  - DiCOVA: Respiratory illness detection                    ║
║  - FluSense: Flu detection from hospital data               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    
    # Step 1: Download health datasets
    if not args.skip_download:
        success = run_command(
            [sys.executable, "scripts/download_more_data.py", 
             "--health", f"--parallel={args.parallel}"],
            "Downloading real health datasets with illness labels",
            allow_fail=True  # Some datasets may fail but we can continue
        )
        if not success:
            print("\nWarning: Some downloads may have failed. Continuing with available data...")
    else:
        print("\n[SKIP] Dataset download skipped")
    
    # Step 2: Prepare real data
    if not args.skip_prepare:
        success = run_command(
            [sys.executable, "scripts/prepare_real_data.py",
             "--data-dir", "data/",
             "--output", "data/processed/",
             f"--parallel={args.parallel}"],
            "Processing real health data (extracting features, creating splits)"
        )
        if not success:
            print("\nFailed to prepare data!")
            print("Make sure health datasets were downloaded successfully.")
            print("Run: python scripts/download_more_data.py --health --list")
            return 1
    else:
        print("\n[SKIP] Data preparation skipped")
    
    # Check that processed data exists
    processed_dir = Path("data/processed")
    if not (processed_dir / "train.json").exists():
        print("\nERROR: Processed training data not found!")
        print("Please run: python scripts/prepare_real_data.py")
        return 1
    
    # Step 3: Train model
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", "configs/high_performance.yaml",
        "--data", "data/processed/",
        "--output", str(output_dir),
    ]
    
    if args.epochs:
        train_cmd.extend(["--epochs", str(args.epochs)])
    
    if args.resume:
        checkpoint = output_dir / "checkpoint.pt"
        if checkpoint.exists():
            train_cmd.extend(["--resume", str(checkpoint)])
    
    success = run_command(
        train_cmd,
        "Training with high-performance configuration on REAL data"
    )
    
    if not success:
        print("Training failed")
        return 1
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE!                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total time: {hours:02d}h {minutes:02d}m                                          ║
║  Output: {str(output_dir):<50} ║
║                                                              ║
║  Key files:                                                  ║
║  - best_model.pt       : Best trained model                  ║
║  - training_history.pkl: Training curves                     ║
║  - config.yaml         : Configuration used                  ║
║  - train.log           : Full training log                   ║
║                                                              ║
║  The model can now predict:                                  ║
║  - 24h/48h/72h illness risk probability                      ║
║  - Virus type: COVID, Flu, Cold, RSV, Pneumonia, General     ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Print results summary
    try:
        import pickle
        history_path = output_dir / "training_history.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            
            best_auprc = max(history.get('val_auprc', [0]))
            best_auroc = max(history.get('val_auroc', [0]))
            
            print(f"RESULTS:")
            print(f"  Best Validation AUPRC: {best_auprc:.4f}")
            print(f"  Best Validation AUROC: {best_auroc:.4f}")
            
            if 'virus_accuracy' in history:
                best_virus_acc = max(history.get('virus_accuracy', [0]))
                print(f"  Best Virus Classification Accuracy: {best_virus_acc:.2%}")
    except Exception:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
