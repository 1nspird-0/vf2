"""Training loop for ViralFlip model.

Key features:
- Mixed precision (FP16) training for GPU acceleration
- Gradient accumulation for effective larger batches
- Cosine annealing with warmup
- User-level and temporal splits
- Early stopping on validation AUPRC
- Checkpoint saving with resume capability
- Comprehensive logging with TensorBoard support
"""

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, LambdaLR
)
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from viralflip.model.viralflip import ViralFlip
from viralflip.train.losses import CombinedLoss
from viralflip.train.build_sequences import UserDataset, collate_user_batch
from viralflip.utils.logging import get_logger
from viralflip.utils.io import save_pickle, ensure_dir


logger = get_logger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Create a cosine annealing schedule with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum change to count as improvement.
            mode: 'max' for metrics where higher is better, 'min' otherwise.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            
        Returns:
            True if training should stop.
        """
        # Handle NaN scores - treat as no improvement
        if np.isnan(score):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ViralFlipTrainer:
    """Trainer for ViralFlip model with GPU optimization."""
    
    def __init__(
        self,
        model: ViralFlip,
        config: dict,
        output_dir: Path,
        device: str = "auto",
    ):
        """Initialize trainer.
        
        Args:
            model: ViralFlip model instance.
            config: Training configuration dict.
            output_dir: Directory for outputs.
            device: Device to use ('auto', 'cpu', 'cuda').
        """
        self.model = model
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # GPU optimizations
        gpu_cfg = config.get("gpu", {})
        if self.device.type == "cuda":
            if gpu_cfg.get("cudnn_benchmark", True):
                torch.backends.cudnn.benchmark = True
            if not gpu_cfg.get("deterministic", False):
                torch.backends.cudnn.deterministic = False
            
            # Try to compile model with torch.compile (PyTorch 2.0+)
            if gpu_cfg.get("compile_model", False):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
        
        # Training settings
        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 100))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.lr = float(train_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        
        # Mixed precision
        self.use_amp = train_cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("Using automatic mixed precision (FP16)")
        
        # Data loading
        self.num_workers = train_cfg.get("num_workers", 0)
        self.pin_memory = train_cfg.get("pin_memory", True) and self.device.type == "cuda"
        self.prefetch_factor = train_cfg.get("prefetch_factor", 2)
        
        # Loss (with virus classification if model supports it)
        use_virus_loss = getattr(model, 'use_virus_classifier', False)
        self.loss_fn = CombinedLoss(
            horizons=model.horizons,
            horizon_weights=train_cfg.get("horizon_weights"),
            use_focal=train_cfg.get("use_focal_loss", True),
            focal_gamma=train_cfg.get("focal_gamma", 2.0),
            pos_weight=train_cfg.get("pos_weight_multiplier", 5.0),
            use_virus_loss=use_virus_loss,
            virus_loss_weight=train_cfg.get("virus_loss_weight", 0.5),
        )
        
        # Optimizer (AdamW for better weight decay handling)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        self.lr_scheduler_type = train_cfg.get("lr_scheduler", "reduce_on_plateau")
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        self.min_lr = float(train_cfg.get("min_lr", 1e-7))
        self.scheduler = None  # Will be set in train()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 15),
            mode="max",
        )
        
        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auprc": [],
            "val_auroc": [],
            "lr": [],
            "epoch_time": [],
        }
        self.best_model_state = None
        self.best_epoch = 0
        self.global_step = 0
        
        # TensorBoard (optional)
        self.tensorboard_writer = None
        if config.get("logging", {}).get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
                logger.info(f"TensorBoard logging to {self.output_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available")
    
    def _create_scheduler(self, steps_per_epoch: int):
        """Create learning rate scheduler based on config."""
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        
        if self.lr_scheduler_type == "cosine_warmup":
            min_lr_ratio = self.min_lr / self.lr
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif self.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr,
            )
        else:  # reduce_on_plateau
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.min_lr,
            )
    
    def _check_class_distribution(self, dataset, split_name: str):
        """Check and log class distribution in dataset.
        
        Warns if only one class is present per horizon.
        """
        try:
            # Sample labels from dataset
            all_labels = []
            for i in range(min(len(dataset), 1000)):  # Sample up to 1000
                sample = dataset[i]
                all_labels.append(sample["labels"].numpy())
            
            all_labels = np.stack(all_labels)
            
            logger.info(f"Class distribution in {split_name} set ({len(all_labels)} samples):")
            has_issue = False
            
            for i, horizon in enumerate(self.model.horizons):
                labels = all_labels[:, i]
                n_pos = int(labels.sum())
                n_neg = len(labels) - n_pos
                pos_rate = n_pos / len(labels) * 100
                
                status = ""
                if n_pos == 0:
                    status = " ⚠️ NO POSITIVE SAMPLES - METRICS WILL BE UNDEFINED!"
                    has_issue = True
                elif n_neg == 0:
                    status = " ⚠️ NO NEGATIVE SAMPLES - METRICS WILL BE UNDEFINED!"
                    has_issue = True
                elif n_pos < 10 or n_neg < 10:
                    status = " ⚠️ Very few samples of one class"
                    has_issue = True
                
                logger.info(f"  {horizon}h: {n_pos} positive ({pos_rate:.1f}%), {n_neg} negative{status}")
            
            if has_issue:
                logger.warning(
                    f"Class imbalance detected in {split_name} set! "
                    "This will cause AUPRC/AUROC to be undefined or unreliable. "
                    "Consider: (1) collecting more data, (2) adjusting train/val split, "
                    "(3) using stratified sampling."
                )
        except Exception as e:
            logger.debug(f"Could not check class distribution: {e}")
    
    def train(
        self,
        train_dataset: UserDataset,
        val_dataset: UserDataset,
        resume_from: Optional[Path] = None,
    ) -> dict:
        """Train the model.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            resume_from: Optional checkpoint path to resume from.
            
        Returns:
            Training history dict.
        """
        # Data loaders with optimized settings
        loader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": collate_user_batch,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,  # Drop last incomplete batch for consistent gradients
            **loader_kwargs,
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
        
        # Create scheduler
        steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        self.scheduler = self._create_scheduler(steps_per_epoch)
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from is not None and resume_from.exists():
            start_epoch = self._load_checkpoint(resume_from)
            logger.info(f"Resumed from epoch {start_epoch}")
        
        logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size} x {self.gradient_accumulation_steps} accum = {self.batch_size * self.gradient_accumulation_steps} effective")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Workers: {self.num_workers}, Pin memory: {self.pin_memory}")
        
        # Check class distribution in validation set
        self._check_class_distribution(val_dataset, "validation")
        
        # Print GPU info
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Compute mean AUPRC (use nanmean to handle undefined metrics)
            auprc_values = [val_metrics[f"auprc_{h}h"] for h in self.model.horizons]
            val_auprc_mean = np.nanmean(auprc_values) if not all(np.isnan(auprc_values)) else 0.0
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_auprc_mean if not np.isnan(val_auprc_mean) else 0.0)
            # Note: step-based schedulers are updated in _train_epoch
            
            epoch_time = time.time() - epoch_start
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUPRC: {val_auprc_mean:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tensorboard_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.tensorboard_writer.add_scalar("AUPRC/val_mean", val_auprc_mean, epoch)
                self.tensorboard_writer.add_scalar("LR", current_lr, epoch)
                for h in self.model.horizons:
                    self.tensorboard_writer.add_scalar(f"AUPRC/val_{h}h", val_metrics[f"auprc_{h}h"], epoch)
                
                # Virus classification metrics
                if "virus_accuracy" in val_metrics:
                    self.tensorboard_writer.add_scalar("Virus/accuracy", val_metrics["virus_accuracy"], epoch)
                if "virus_f1_macro" in val_metrics:
                    self.tensorboard_writer.add_scalar("Virus/f1_macro", val_metrics["virus_f1_macro"], epoch)
            
            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auprc"].append(val_auprc_mean)
            auroc_values = [val_metrics[f"auroc_{h}h"] for h in self.model.horizons]
            val_auroc_mean = np.nanmean(auroc_values) if not all(np.isnan(auroc_values)) else 0.5
            self.history["val_auroc"].append(val_auroc_mean)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            
            # Early stopping check
            if self.early_stopping(val_auprc_mean):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save best model (skip if metric is undefined/NaN)
            if not np.isnan(val_auprc_mean) and val_auprc_mean >= self.early_stopping.best_score:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch + 1, is_best=True)
            
            # Periodic checkpoint
            save_every = self.config.get("logging", {}).get("save_every_n_epochs", 10)
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1, is_best=False)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model from epoch {self.best_epoch}")
        
        # Save final history
        save_pickle(self.history, self.output_dir / "training_history.pkl")
        
        # Close TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        # Print summary
        total_time = sum(self.history["epoch_time"])
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best AUPRC: {self.early_stopping.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and mixed precision.
        
        Args:
            loader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            features = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["features"].items()
            }
            missing = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["missing"].items()
            }
            quality = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["quality"].items()
            }
            labels = batch["labels"].to(self.device, non_blocking=True)
            user_ids = batch["user_ids"]
            
            # Virus labels (optional)
            virus_labels = None
            if "virus_type" in batch:
                virus_labels = batch["virus_type"].to(self.device, non_blocking=True)
            
            # Forward with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                predictions, confidence, virus_logits, _ = self.model(
                    features, missing, quality, user_ids
                )
                
                # Use last timestep for loss
                predictions = predictions[:, -1, :]
                
                # Loss (scaled for gradient accumulation)
                reg_penalty = self.model.total_penalty()
                loss, _ = self.loss_fn(
                    predictions, labels, reg_penalty,
                    virus_logits=virus_logits,
                    virus_targets=virus_labels,
                )
                loss = loss / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Step scheduler if not ReduceLROnPlateau
                if not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        """Validate the model.
        
        Args:
            loader: Validation data loader.
            
        Returns:
            Dict of validation metrics.
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        
        all_virus_preds = []
        all_virus_labels = []
        
        for batch in loader:
            features = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["features"].items()
            }
            missing = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["missing"].items()
            }
            quality = {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch["quality"].items()
            }
            labels = batch["labels"].to(self.device, non_blocking=True)
            user_ids = batch["user_ids"]
            
            # Virus labels (optional)
            virus_labels = None
            if "virus_type" in batch:
                virus_labels = batch["virus_type"].to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.use_amp):
                predictions, _, virus_logits, _ = self.model(features, missing, quality, user_ids)
                predictions = predictions[:, -1, :]
                
                reg_penalty = self.model.total_penalty()
                loss, _ = self.loss_fn(
                    predictions, labels, reg_penalty,
                    virus_logits=virus_logits,
                    virus_targets=virus_labels,
                )
            
            all_preds.append(predictions.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Collect virus predictions
            if virus_logits is not None and virus_labels is not None:
                all_virus_preds.append(virus_logits.argmax(dim=-1).cpu().numpy())
                all_virus_labels.append(virus_labels.cpu().numpy())
            
            total_loss += loss.item()
            n_batches += 1
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
        
        metrics = {"loss": total_loss / n_batches}
        
        for i, horizon in enumerate(self.model.horizons):
            pred_h = all_preds[:, i]
            label_h = all_labels[:, i]
            
            # Check for single-class scenario (no positive or all positive samples)
            n_positive = int(label_h.sum())
            n_samples = len(label_h)
            
            if n_positive == 0:
                # No positive samples - metrics are undefined
                logger.warning(
                    f"Horizon {horizon}h: No positive samples in validation set ({n_samples} total). "
                    "AUPRC/AUROC undefined - check your data split!"
                )
                metrics[f"auprc_{horizon}h"] = float('nan')
                metrics[f"auroc_{horizon}h"] = float('nan')
            elif n_positive == n_samples:
                # All positive samples - metrics are undefined
                logger.warning(
                    f"Horizon {horizon}h: All samples are positive ({n_samples} total). "
                    "AUPRC/AUROC undefined - check your data split!"
                )
                metrics[f"auprc_{horizon}h"] = float('nan')
                metrics[f"auroc_{horizon}h"] = float('nan')
            else:
                metrics[f"auprc_{horizon}h"] = average_precision_score(label_h, pred_h)
                metrics[f"auroc_{horizon}h"] = roc_auc_score(label_h, pred_h)
        
        # Virus classification metrics
        if all_virus_preds and all_virus_labels:
            virus_preds = np.concatenate(all_virus_preds)
            virus_labels = np.concatenate(all_virus_labels)
            
            # Filter to samples with illness (virus_type > 0)
            illness_mask = virus_labels > 0
            if illness_mask.sum() > 0:
                metrics["virus_accuracy"] = accuracy_score(
                    virus_labels[illness_mask], 
                    virus_preds[illness_mask]
                )
                metrics["virus_f1_macro"] = f1_score(
                    virus_labels[illness_mask], 
                    virus_preds[illness_mask],
                    average="macro",
                    zero_division=0,
                )
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "history": self.history,
            "config": self.config,
            "global_step": self.global_step,
            "best_score": self.early_stopping.best_score,
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.output_dir / "checkpoint.pt")
        
        # Save best model separately
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")
    
    def _load_checkpoint(self, path: Path) -> int:
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
            
        Returns:
            Epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.history = checkpoint.get("history", self.history)
        self.global_step = checkpoint.get("global_step", 0)
        self.early_stopping.best_score = checkpoint.get("best_score", self.early_stopping.best_score)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint["epoch"]
