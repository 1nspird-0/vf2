#!/usr/bin/env python3
"""Run ablation studies for ViralFlip.

Usage:
    python scripts/run_ablations.py --run_dir runs/20241224_120000
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from viralflip.eval.ablations import AblationRunner, AblationResult
from viralflip.utils.logging import setup_logging, get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ViralFlip ablation studies")
    parser.add_argument("--run_dir", "-r", type=str, required=True,
                       help="Path to baseline run directory")
    parser.add_argument("--ablations", "-a", type=str, nargs="+", default=None,
                       help="Specific ablations to run (default: all)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file (default: run_dir/ablations.json)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    setup_logging()
    
    # Load baseline config
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Define simplified train and eval functions
    # In practice, these would fully retrain the model
    # Here we provide a stub that demonstrates the interface
    
    def train_fn(config: dict) -> Any:
        """Train model with given config (stub)."""
        logger.info(f"Training with config modifications...")
        # In real implementation:
        # 1. Initialize model with config
        # 2. Train on data
        # 3. Return trained model
        return None
    
    def eval_fn(model: Any) -> dict:
        """Evaluate model (stub)."""
        # In real implementation:
        # 1. Load test data
        # 2. Run evaluation
        # 3. Return metrics dict
        return {
            "auprc_mean": np.random.uniform(0.2, 0.6),
            "auroc_mean": np.random.uniform(0.6, 0.8),
        }
    
    # Create ablation runner
    output_dir = run_dir / "ablations"
    runner = AblationRunner(
        train_fn=train_fn,
        eval_fn=eval_fn,
        base_config=base_config,
        output_dir=output_dir,
    )
    
    # Determine which ablations to run
    if args.ablations:
        ablation_keys = args.ablations
    else:
        ablation_keys = list(AblationRunner.ABLATIONS.keys())
    
    logger.info(f"Running {len(ablation_keys)} ablations: {ablation_keys}")
    
    # Note: For a real implementation, you would:
    # 1. Load the baseline model and get actual metrics
    # 2. For each ablation, retrain the model and evaluate
    # 
    # Here we demonstrate the structure but use stubs
    
    logger.info("\n" + "=" * 60)
    logger.info("NOTE: This is a demonstration script.")
    logger.info("For real ablations, implement train_fn and eval_fn properly.")
    logger.info("=" * 60 + "\n")
    
    # Run ablations
    results = runner.run_all_ablations(ablation_keys)
    
    # Print summary
    logger.info("\n=== Ablation Summary ===")
    for result in sorted(results, key=lambda r: r.delta_auprc):
        logger.info(
            f"{result.name:40s} | "
            f"ΔAUPRC: {result.delta_auprc:+.4f} | "
            f"ΔAUROC: {result.delta_auroc:+.4f}"
        )
    
    # Save results
    output_path = Path(args.output) if args.output else run_dir / "ablations.json"
    runner.save_results(str(output_path))
    
    # Also save detailed summary
    summary = runner.get_summary()
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print key findings
    logger.info("\n=== Key Findings ===")
    if summary.get("most_impactful"):
        logger.info(f"Most impactful ablation: {summary['most_impactful']}")
    if summary.get("least_impactful"):
        logger.info(f"Least impactful ablation: {summary['least_impactful']}")


if __name__ == "__main__":
    main()

