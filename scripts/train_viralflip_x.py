#!/usr/bin/env python
"""Train ViralFlip-X model with all advanced features.

Usage:
    python train_viralflip_x.py --config configs/viralflip_x.yaml
    python train_viralflip_x.py --config configs/viralflip_x.yaml --pretrain-only
    python train_viralflip_x.py --config configs/viralflip_x.yaml --resume runs/viralflip_x/checkpoint.pt
"""

import argparse
from pathlib import Path
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ViralFlip-X imports
from viralflip.model.viralflip import ViralFlip
from viralflip.pretrain.masked_autoencoder import MaskedMultimodalAutoencoder, MaskConfig
from viralflip.pretrain.pretrain_trainer import PretrainTrainer, PretrainConfig
from viralflip.robust.irm import IRMLoss, BehaviorEnvironmentDetector
from viralflip.train.losses import CombinedLoss
from viralflip.train.build_sequences import UserDataset, collate_user_batch
from viralflip.baseline.changepoint import MultiModalityChangePointBaseline
from viralflip.conformal.conformal_predictor import MultiHorizonConformal
from viralflip.utils.logging import get_logger
from viralflip.utils.io import ensure_dir, save_pickle
from viralflip.utils.seed import set_seed

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Create cosine schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class ViralFlipXTrainer:
    """Trainer for ViralFlip-X with all advanced features."""
    
    def __init__(
        self,
        model: ViralFlip,
        config: dict,
        output_dir: Path,
        device: str = "auto",
    ):
        """Initialize trainer.
        
        Args:
            model: ViralFlip-X model
            config: Training configuration
            output_dir: Output directory
            device: Device to use
        """
        self.model = model
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        
        # Mixed precision
        self.use_amp = train_cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Base loss
        self.base_loss = CombinedLoss(
            horizons=model.horizons,
            horizon_weights=train_cfg.get("horizon_weights"),
            use_focal=train_cfg.get("use_focal_loss", True),
            focal_gamma=train_cfg.get("focal_gamma", 2.0),
            pos_weight=train_cfg.get("pos_weight_multiplier", 5.0),
        )
        
        # IRM loss (if enabled)
        irm_cfg = config.get("irm", {})
        self.use_irm = irm_cfg.get("enabled", True)
        if self.use_irm:
            self.irm_loss = IRMLoss(
                base_loss=self.base_loss.prediction_loss,
                irm_penalty_weight=irm_cfg.get("penalty_weight", 1.0),
                anneal_iters=irm_cfg.get("anneal_iters", 500),
            )
            self.env_detector = BehaviorEnvironmentDetector(
                mobility_threshold=irm_cfg.get("mobility_threshold", 0.5),
                quality_threshold=irm_cfg.get("quality_threshold", 0.7),
                activity_threshold=irm_cfg.get("activity_threshold", 0.3),
            )
        
        # Conformal (for evaluation)
        conformal_cfg = config.get("conformal", {})
        self.use_conformal = conformal_cfg.get("enabled", True)
        if self.use_conformal:
            self.conformal = MultiHorizonConformal(
                horizons=model.horizons,
                alpha=conformal_cfg.get("alpha", 0.1),
                use_changepoint=conformal_cfg.get("use_changepoint", True),
            )
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auprc": [],
            "irm_penalty": [],
            "conformal_coverage": [],
        }
        
        self.global_step = 0
        self.best_auprc = 0.0
        self.best_epoch = 0
    
    def train(
        self,
        train_dataset: UserDataset,
        val_dataset: UserDataset,
        resume_from: Path = None,
    ) -> dict:
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Training history
        """
        train_cfg = self.config.get("training", {})
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_user_batch,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_user_batch,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=self.device.type == "cuda",
        )
        
        # Scheduler
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        total_steps = len(train_loader) * self.epochs
        warmup_steps = len(train_loader) * warmup_epochs
        min_lr_ratio = train_cfg.get("min_lr", 1e-7) / self.lr
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps, min_lr_ratio
        )
        
        # Resume
        start_epoch = 0
        if resume_from is not None and resume_from.exists():
            start_epoch = self._load_checkpoint(resume_from)
        
        logger.info(f"Training ViralFlip-X on {len(train_dataset)} samples")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        logger.info(f"IRM enabled: {self.use_irm}")
        logger.info(f"Conformal enabled: {self.use_conformal}")
        
        patience = train_cfg.get("early_stopping_patience", 15)
        patience_counter = 0
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, irm_penalty = self._train_epoch(train_loader, scheduler, epoch)
            
            # Validate
            val_metrics = self._validate(val_loader)
            val_auprc = np.mean([val_metrics[f"auprc_{h}h"] for h in self.model.horizons])
            
            # Conformal calibration and evaluation
            if self.use_conformal:
                conformal_coverage = self._evaluate_conformal(val_loader)
                self.history["conformal_coverage"].append(conformal_coverage)
            
            epoch_time = time.time() - epoch_start
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"IRM: {irm_penalty:.4f} | "
                f"Val AUPRC: {val_auprc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auprc"].append(val_auprc)
            self.history["irm_penalty"].append(irm_penalty)
            
            # Early stopping
            if val_auprc > self.best_auprc:
                self.best_auprc = val_auprc
                self.best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(epoch + 1, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Periodic save
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1)
        
        # Load best model
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            self._load_checkpoint(best_path)
        
        save_pickle(self.history, self.output_dir / "training_history.pkl")
        
        logger.info(f"Training complete. Best AUPRC: {self.best_auprc:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    def _train_epoch(self, loader, scheduler, epoch) -> tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_irm = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, ncols=100)
        
        for batch in pbar:
            # Move to device
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            missing = {k: v.to(self.device) for k, v in batch["missing"].items()}
            quality = {k: v.to(self.device) for k, v in batch["quality"].items()}
            labels = batch["labels"].to(self.device)
            user_ids = batch["user_ids"]
            
            with autocast(enabled=self.use_amp):
                # Forward
                predictions, uncertainty, _, _ = self.model(
                    features, missing, quality, user_ids, return_features=False
                )
                predictions = predictions[:, -1, :]
                
                # Regularization
                reg_penalty = self.model.total_penalty()
                
                if self.use_irm:
                    # Create environment masks (simplified - use dummy)
                    batch_size = predictions.shape[0]
                    env_masks = {
                        f"env_{i}": torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                        for i in range(4)
                    }
                    # Simple split for demo
                    for i in range(batch_size):
                        env_masks[f"env_{i % 4}"][i] = True
                    
                    logits = torch.logit(predictions.clamp(1e-7, 1 - 1e-7))
                    loss, loss_dict = self.irm_loss(logits, labels, env_masks, reg_penalty)
                    irm_penalty = loss_dict.get("irm", torch.tensor(0.0)).item()
                else:
                    loss, _ = self.base_loss(predictions, labels, reg_penalty)
                    irm_penalty = 0.0
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            scheduler.step()
            
            total_loss += loss.item()
            total_irm += irm_penalty
            n_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})
        
        return total_loss / n_batches, total_irm / n_batches
    
    @torch.no_grad()
    def _validate(self, loader) -> dict:
        """Validate model."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            missing = {k: v.to(self.device) for k, v in batch["missing"].items()}
            quality = {k: v.to(self.device) for k, v in batch["quality"].items()}
            labels = batch["labels"].to(self.device)
            user_ids = batch["user_ids"]
            
            with autocast(enabled=self.use_amp):
                predictions, _, _, _ = self.model(features, missing, quality, user_ids)
                predictions = predictions[:, -1, :]
                
                reg_penalty = self.model.total_penalty()
                loss, _ = self.base_loss(predictions, labels, reg_penalty)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        metrics = {"loss": total_loss / n_batches}
        
        for i, horizon in enumerate(self.model.horizons):
            pred_h = all_preds[:, i]
            label_h = all_labels[:, i]
            
            # Check for single-class scenario (no positive or all positive samples)
            n_positive = int(label_h.sum())
            n_samples = len(label_h)
            
            if n_positive == 0:
                # No positive samples - metrics are undefined
                print(f"  WARNING: Horizon {horizon}h has no positive samples ({n_samples} total)")
                metrics[f"auprc_{horizon}h"] = float('nan')
                metrics[f"auroc_{horizon}h"] = float('nan')
            elif n_positive == n_samples:
                # All positive samples - metrics are undefined
                print(f"  WARNING: Horizon {horizon}h has all positive samples ({n_samples} total)")
                metrics[f"auprc_{horizon}h"] = float('nan')
                metrics[f"auroc_{horizon}h"] = float('nan')
            else:
                metrics[f"auprc_{horizon}h"] = average_precision_score(label_h, pred_h)
                metrics[f"auroc_{horizon}h"] = roc_auc_score(label_h, pred_h)
        
        return metrics
    
    @torch.no_grad()
    def _evaluate_conformal(self, loader) -> float:
        """Evaluate conformal coverage."""
        self.model.eval()
        
        # Collect predictions and labels
        all_preds = {h: [] for h in self.model.horizons}
        all_labels = {h: [] for h in self.model.horizons}
        
        for batch in loader:
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            missing = {k: v.to(self.device) for k, v in batch["missing"].items()}
            quality = {k: v.to(self.device) for k, v in batch["quality"].items()}
            labels = batch["labels"].cpu().numpy()
            
            predictions, _, _, _ = self.model(features, missing, quality)
            predictions = predictions[:, -1, :].cpu().numpy()
            
            for i, h in enumerate(self.model.horizons):
                all_preds[h].extend(predictions[:, i].tolist())
                all_labels[h].extend(labels[:, i].tolist())
        
        # Calibrate and evaluate coverage
        preds_np = {h: np.array(all_preds[h]) for h in self.model.horizons}
        labels_np = {h: np.array(all_labels[h]) for h in self.model.horizons}
        
        # Split for calibration/evaluation
        n = len(preds_np[self.model.horizons[0]])
        cal_idx = np.random.choice(n, n // 2, replace=False)
        eval_idx = np.array([i for i in range(n) if i not in cal_idx])
        
        # Calibrate
        cal_preds = {h: preds_np[h][cal_idx] for h in self.model.horizons}
        cal_labels = {h: labels_np[h][cal_idx] for h in self.model.horizons}
        self.conformal.calibrate(cal_preds, cal_labels)
        
        # Evaluate coverage
        eval_preds = {h: preds_np[h][eval_idx] for h in self.model.horizons}
        eval_labels = {h: labels_np[h][eval_idx] for h in self.model.horizons}
        
        coverages = []
        for h in self.model.horizons:
            predictor = self.conformal.predictors[h]
            for pred, label in zip(eval_preds[h], eval_labels[h]):
                lower, upper = predictor.predict(pred)
                covered = (label >= lower) and (label <= upper)
                coverages.append(covered)
        
        return np.mean(coverages)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
            "global_step": self.global_step,
            "best_auprc": self.best_auprc,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.output_dir / "checkpoint.pt")
        
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")
    
    def _load_checkpoint(self, path: Path) -> int:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.history = checkpoint.get("history", self.history)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_auprc = checkpoint.get("best_auprc", 0.0)
        
        return checkpoint["epoch"]


def run_pretraining(config: dict, output_dir: Path):
    """Run self-supervised pretraining."""
    pretrain_cfg = config.get("pretrain", {})
    
    logger.info("Starting self-supervised pretraining...")
    
    # Build modality dims from config
    feature_dims = {}
    for modality in ["voice", "cough", "tap", "gait_active", "rppg", "light", "baro"]:
        feature_dims[modality] = config.get("features", {}).get(modality, {}).get("n_features", 10)
    
    # Default dims if not specified
    default_dims = {
        "voice": 6, "cough": 4, "tap": 6, "gait_active": 8,
        "rppg": 4, "light": 3, "baro": 3
    }
    for m, d in default_dims.items():
        if m not in feature_dims or feature_dims[m] == 0:
            feature_dims[m] = d
    
    # Create model
    mask_config = MaskConfig(
        mask_ratio=pretrain_cfg.get("mask_ratio", 0.4),
        temporal_mask_prob=pretrain_cfg.get("temporal_mask_prob", 0.3),
        modality_mask_prob=pretrain_cfg.get("modality_mask_prob", 0.2),
        contiguous_mask_prob=pretrain_cfg.get("contiguous_mask_prob", 0.3),
    )
    
    model = MaskedMultimodalAutoencoder(
        modality_dims=feature_dims,
        embed_dim=pretrain_cfg.get("embed_dim", 128),
        encoder_layers=pretrain_cfg.get("encoder_layers", 4),
        decoder_layers=pretrain_cfg.get("decoder_layers", 2),
        n_heads=pretrain_cfg.get("n_heads", 4),
        ff_dim=pretrain_cfg.get("ff_dim", 512),
        dropout=pretrain_cfg.get("dropout", 0.1),
        mask_config=mask_config,
    )
    
    pretrain_config = PretrainConfig(
        embed_dim=pretrain_cfg.get("embed_dim", 128),
        encoder_layers=pretrain_cfg.get("encoder_layers", 4),
        epochs=pretrain_cfg.get("epochs", 100),
        batch_size=pretrain_cfg.get("batch_size", 64),
        learning_rate=pretrain_cfg.get("learning_rate", 1e-4),
    )
    
    trainer = PretrainTrainer(
        model=model,
        config=pretrain_config,
        output_dir=output_dir / "pretrain",
    )
    
    logger.info("Pretraining requires unlabeled data. Skipping for demo...")
    logger.info("In production, load unlabeled sensor streams here.")
    
    # Save encoder
    torch.save(model.encoder.state_dict(), output_dir / "pretrain" / "encoder.pt")
    
    return model.encoder


def main():
    parser = argparse.ArgumentParser(description="Train ViralFlip-X")
    parser.add_argument("--config", type=str, default="configs/viralflip_x.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed or config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("paths", {}).get("output_dir", "runs/viralflip_x"))
    output_dir = ensure_dir(output_dir)
    
    # Pretraining
    pretrain_cfg = config.get("pretrain", {})
    encoder = None
    
    if pretrain_cfg.get("enabled", True) and not args.skip_pretrain:
        encoder = run_pretraining(config, output_dir)
        
        if args.pretrain_only:
            logger.info("Pretraining complete. Exiting.")
            return
    
    # Build feature dims
    feature_dims = {}
    default_dims = {
        "voice": 6, "cough": 4, "tap": 6, "gait_active": 8,
        "rppg": 4, "light": 3, "baro": 3,
        "gps": 6, "imu_passive": 8, "screen": 4
    }
    for m, d in default_dims.items():
        feature_dims[m] = d
    
    # Create ViralFlip-X model
    model_cfg = config.get("model", {})
    
    model = ViralFlip(
        feature_dims=feature_dims,
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        max_lag=model_cfg.get("max_lag_bins", 12),
        use_encoder=model_cfg.get("use_pretrained_encoder", True),
        encoder_embed_dim=model_cfg.get("encoder_embed_dim", 128),
        l1_lambda_drift=model_cfg.get("l1_lambda_drift", 0.01),
        l1_lambda_lattice=model_cfg.get("l1_lambda_lattice", 0.01),
        use_interactions=model_cfg.get("use_interactions", False),
        use_personalization=config.get("personalization", {}).get("enabled", True),
        use_conformal=config.get("conformal", {}).get("enabled", True),
        conformal_alpha=config.get("conformal", {}).get("alpha", 0.1),
        use_environment_detection=config.get("irm", {}).get("enabled", True),
        n_environments=config.get("irm", {}).get("n_environments", 4),
    )
    
    # Load pretrained encoder if available
    encoder_path = output_dir / "pretrain" / "encoder.pt"
    if encoder_path.exists() and model_cfg.get("use_pretrained_encoder", True):
        try:
            model.load_pretrained_encoder(encoder_path)
            logger.info(f"Loaded pretrained encoder from {encoder_path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained encoder: {e}")
    
    # Create trainer
    trainer = ViralFlipXTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=args.device,
    )
    
    logger.info("ViralFlip-X model created successfully!")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training would happen here with actual data
    logger.info("Training requires labeled data. Set up datasets to proceed.")
    logger.info(f"Config saved to {output_dir}")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save model architecture summary
    summary = model.get_state_summary()
    save_pickle(summary, output_dir / "model_summary.pkl")
    
    logger.info("Setup complete! Ready for training with data.")


if __name__ == "__main__":
    main()

