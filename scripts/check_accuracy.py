#!/usr/bin/env python3
"""Check accuracy of a trained ViralFlip model using real dataset samples.

Usage:
    python scripts/check_accuracy.py --model runs/20241225_120000/best_model.pt
    python scripts/check_accuracy.py --model runs/best_model.pt --data data/processed --split test
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from viralflip.model.viralflip import ViralFlip
from viralflip.model.virus_types import VirusType, VIRUS_NAMES, NUM_VIRUS_CLASSES
from viralflip.data.dataset import ViralFlipDataset, collate_viralflip_batch


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_metric(name: str, value: float, fmt: str = ".4f"):
    """Print formatted metric."""
    print(f"  {name:.<40} {value:{fmt}}")


def find_latest_run() -> Path:
    """Find the most recent training run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    runs = sorted(
        [r for r in runs_dir.iterdir() if r.is_dir() and (r / "best_model.pt").exists()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return runs[0] / "best_model.pt" if runs else None


def load_model(model_path: Path, device: str = "cpu") -> tuple[ViralFlip, dict]:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to load config from checkpoint or run directory
    config = checkpoint.get("config", {})
    
    if not config:
        config_path = model_path.parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
    
    # Get feature dims from config or use defaults
    feature_dims = config.get("feature_dims", {
        "voice": 30,
        "cough": 30,
        "breathing": 20,
        "rppg": 5,
        "activity": 6,
    })
    
    # Create model
    model_cfg = config.get("model", {})
    model = ViralFlip(
        feature_dims=feature_dims,
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        max_lag=model_cfg.get("max_lag_bins", 12),
        use_interactions=model_cfg.get("use_interactions", True),
        use_missing_indicators=model_cfg.get("use_missing_indicators", True),
        use_personalization=config.get("personalization", {}).get("enabled", True),
        use_virus_classifier=model_cfg.get("use_virus_classifier", True),
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_with_dataloader(
    model: ViralFlip,
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Evaluate model using a DataLoader.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test samples
        device: Evaluation device
        
    Returns:
        Tuple of (risk_metrics, virus_metrics)
    """
    model.eval()
    
    all_risk_preds = []
    all_risk_labels = []
    all_virus_preds = []
    all_virus_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move to device
            features = {k: v.to(device) for k, v in batch["features"].items()}
            missing = {k: v.to(device) for k, v in batch["missing"].items()}
            quality = {k: v.to(device) for k, v in batch["quality"].items()}
            labels = batch["labels"]
            virus_types = batch["virus_type"]
            user_ids = batch["user_ids"]
            
            # Forward pass
            risk_probs, confidence, virus_logits, _ = model(
                features, missing, quality, user_ids
            )
            
            # Get predictions for last timestep
            risk_preds = risk_probs[:, -1, :].cpu().numpy()
            all_risk_preds.append(risk_preds)
            all_risk_labels.append(labels.numpy())
            
            # Virus predictions
            if virus_logits is not None:
                virus_preds = virus_logits.argmax(dim=-1).cpu().numpy()
                all_virus_preds.append(virus_preds)
                all_virus_labels.append(virus_types.numpy())
    
    # Concatenate all predictions
    all_risk_preds = np.concatenate(all_risk_preds, axis=0)
    all_risk_labels = np.concatenate(all_risk_labels, axis=0)
    
    # Compute risk metrics
    risk_metrics = {}
    for i, h in enumerate(model.horizons):
        preds = all_risk_preds[:, i]
        true_labels = all_risk_labels[:, i]
        
        # Binary predictions
        pred_binary = (preds >= 0.5).astype(int)
        
        n_pos = int(true_labels.sum())
        n_neg = len(true_labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            risk_metrics[f"auprc_{h}h"] = float("nan")
            risk_metrics[f"auroc_{h}h"] = float("nan")
        else:
            risk_metrics[f"auprc_{h}h"] = average_precision_score(true_labels, preds)
            risk_metrics[f"auroc_{h}h"] = roc_auc_score(true_labels, preds)
        
        risk_metrics[f"accuracy_{h}h"] = accuracy_score(true_labels, pred_binary)
        risk_metrics[f"precision_{h}h"] = precision_score(true_labels, pred_binary, zero_division=0)
        risk_metrics[f"recall_{h}h"] = recall_score(true_labels, pred_binary, zero_division=0)
        risk_metrics[f"f1_{h}h"] = f1_score(true_labels, pred_binary, zero_division=0)
        risk_metrics[f"n_positive_{h}h"] = n_pos
        risk_metrics[f"n_negative_{h}h"] = n_neg
    
    # Compute virus metrics
    virus_metrics = {"enabled": False}
    if all_virus_preds:
        all_virus_preds = np.concatenate(all_virus_preds)
        all_virus_labels = np.concatenate(all_virus_labels)
        
        virus_metrics = {
            "enabled": True,
            "accuracy": accuracy_score(all_virus_labels, all_virus_preds),
            "f1_macro": f1_score(all_virus_labels, all_virus_preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(all_virus_labels, all_virus_preds, average="weighted", zero_division=0),
        }
        
        # Illness-only metrics
        illness_mask = all_virus_labels > 0
        if illness_mask.sum() > 0:
            virus_metrics["illness_accuracy"] = accuracy_score(
                all_virus_labels[illness_mask], all_virus_preds[illness_mask]
            )
            virus_metrics["illness_f1_macro"] = f1_score(
                all_virus_labels[illness_mask], all_virus_preds[illness_mask],
                average="macro", zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(all_virus_labels, all_virus_preds, labels=list(range(NUM_VIRUS_CLASSES)))
        virus_metrics["confusion_matrix"] = cm.tolist()
        
        # Per-class report
        class_names = [VIRUS_NAMES[VirusType(i)] for i in range(NUM_VIRUS_CLASSES)]
        report = classification_report(
            all_virus_labels, all_virus_preds,
            labels=list(range(NUM_VIRUS_CLASSES)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        virus_metrics["per_class"] = report
    
    return risk_metrics, virus_metrics


def main():
    parser = argparse.ArgumentParser(description="Check ViralFlip model accuracy on dataset")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Path to trained model checkpoint (default: latest run)")
    parser.add_argument("--data", "-d", type=str, default="data/processed",
                       help="Path to processed data directory")
    parser.add_argument("--split", "-s", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Data split to evaluate on")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for evaluation (cpu or cuda)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--workers", "-w", type=int, default=0,
                       help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Find model
    if args.model is None:
        model_path = find_latest_run()
        if model_path is None:
            print("ERROR: No trained model found.")
            print("\nTrain a model first with:")
            print("  python scripts/train.py --max-accuracy")
            return
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nAvailable models:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for run in sorted(runs_dir.iterdir(), reverse=True)[:5]:
                best = run / "best_model.pt"
                if best.exists():
                    print(f"  {best}")
        return
    
    # Check data exists
    data_path = Path(args.data)
    split_file = data_path / f"{args.split}.json"
    
    if not split_file.exists():
        print(f"ERROR: Data not found at {split_file}")
        print("\nPrepare data first with:")
        print("  python scripts/download_more_data.py --health --parallel 4")
        print("  python scripts/prepare_real_data.py")
        return
    
    print_header("ViralFlip Model Accuracy Check")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Split: {args.split}")
    print(f"  Device: {args.device}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(model_path, args.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Horizons: {model.horizons}")
    print(f"  Modalities: {model.physiology_modalities}")
    print(f"  Virus classifier: {model.use_virus_classifier}")
    
    # Get feature dims from model
    feature_dims = {m: model.feature_dims[m] for m in model.physiology_modalities}
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    try:
        dataset = ViralFlipDataset(
            data_path=str(data_path),
            split=args.split,
            feature_dims=feature_dims,
            augment=False,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    print(f"  Samples: {len(dataset)}")
    
    # Count class distribution
    virus_counts = {}
    risk_positive = 0
    for i in range(min(len(dataset), 1000)):  # Sample up to 1000
        sample = dataset[i]
        vt = sample["virus_type"].item()
        virus_counts[vt] = virus_counts.get(vt, 0) + 1
        if sample["labels"][0] > 0.5:
            risk_positive += 1
    
    sample_count = min(len(dataset), 1000)
    print(f"  Positive rate: {risk_positive / sample_count:.1%}")
    print(f"  Virus distribution (sample of {sample_count}):")
    for vt_idx, count in sorted(virus_counts.items()):
        vt_name = VIRUS_NAMES[VirusType(vt_idx)] if vt_idx < NUM_VIRUS_CLASSES else f"Unknown({vt_idx})"
        print(f"    {vt_name}: {count}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_viralflip_batch,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    
    # Evaluate
    print("\nEvaluating model...")
    risk_metrics, virus_metrics = evaluate_with_dataloader(model, dataloader, args.device)
    
    # Print risk metrics
    print_header("Risk Prediction Metrics")
    
    for h in model.horizons:
        print(f"\n  {h}h Horizon:")
        print(f"    Samples: {risk_metrics[f'n_positive_{h}h']} pos / {risk_metrics[f'n_negative_{h}h']} neg")
        
        auprc = risk_metrics[f"auprc_{h}h"]
        auroc = risk_metrics[f"auroc_{h}h"]
        if not np.isnan(auprc):
            print(f"    AUPRC: {auprc:.4f}")
            print(f"    AUROC: {auroc:.4f}")
        else:
            print(f"    AUPRC: N/A (single class)")
            print(f"    AUROC: N/A (single class)")
        
        print(f"    Accuracy: {risk_metrics[f'accuracy_{h}h']:.4f}")
        print(f"    Precision: {risk_metrics[f'precision_{h}h']:.4f}")
        print(f"    Recall: {risk_metrics[f'recall_{h}h']:.4f}")
        print(f"    F1 Score: {risk_metrics[f'f1_{h}h']:.4f}")
    
    # Summary of risk metrics
    valid_auprc = [risk_metrics[f"auprc_{h}h"] for h in model.horizons 
                   if not np.isnan(risk_metrics[f"auprc_{h}h"])]
    valid_auroc = [risk_metrics[f"auroc_{h}h"] for h in model.horizons 
                   if not np.isnan(risk_metrics[f"auroc_{h}h"])]
    
    print(f"\n  Mean AUPRC: {np.mean(valid_auprc):.4f}" if valid_auprc else "\n  Mean AUPRC: N/A")
    print(f"  Mean AUROC: {np.mean(valid_auroc):.4f}" if valid_auroc else "  Mean AUROC: N/A")
    
    # Print virus metrics
    if virus_metrics["enabled"]:
        print_header("Virus Classification Metrics")
        
        print(f"\n  Overall Accuracy: {virus_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {virus_metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted: {virus_metrics['f1_weighted']:.4f}")
        
        if "illness_accuracy" in virus_metrics:
            print(f"\n  Illness-Only Accuracy: {virus_metrics['illness_accuracy']:.4f}")
            print(f"  Illness-Only F1 Macro: {virus_metrics['illness_f1_macro']:.4f}")
        
        # Per-class breakdown
        print("\n  Per-Class Performance:")
        per_class = virus_metrics["per_class"]
        for vt in VirusType:
            name = VIRUS_NAMES[vt]
            if name in per_class:
                cls = per_class[name]
                support = cls.get("support", 0)
                if support > 0:
                    print(f"    {name:.<25} P={cls['precision']:.2f} R={cls['recall']:.2f} F1={cls['f1-score']:.2f} (n={int(support)})")
    
    # Overall summary
    print_header("Summary")
    
    print(f"  Dataset: {data_path}")
    print(f"  Split: {args.split} ({len(dataset)} samples)")
    
    if valid_auprc:
        mean_auprc = np.mean(valid_auprc)
        if mean_auprc >= 0.7:
            print("  ✅ AUPRC is GOOD (≥0.70)")
        elif mean_auprc >= 0.5:
            print("  ⚠️  AUPRC is MODERATE (0.50-0.70)")
        else:
            print("  ❌ AUPRC is LOW (<0.50)")
    
    if virus_metrics["enabled"]:
        acc = virus_metrics["accuracy"]
        if acc >= 0.6:
            print("  ✅ Virus classification is GOOD (≥60%)")
        elif acc >= 0.4:
            print("  ⚠️  Virus classification is MODERATE (40-60%)")
        else:
            print("  ❌ Virus classification is LOW (<40%)")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"accuracy_{args.split}.json"
    
    results = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "split": args.split,
        "n_samples": len(dataset),
        "timestamp": datetime.now().isoformat(),
        "risk_metrics": {k: v if not (isinstance(v, float) and np.isnan(v)) else None 
                         for k, v in risk_metrics.items()},
        "virus_metrics": virus_metrics,
    }
    
    # Remove non-serializable items
    if "confusion_matrix" in results["virus_metrics"]:
        pass  # Already a list
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
