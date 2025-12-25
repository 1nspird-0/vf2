"""Self-supervised pretraining trainer for masked multimodal autoencoder.

Trains the encoder on unlabeled sensor streams to learn robust representations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from viralflip.pretrain.masked_autoencoder import MaskedMultimodalAutoencoder, MaskConfig
from viralflip.utils.logging import get_logger
from viralflip.utils.io import ensure_dir

logger = get_logger(__name__)


@dataclass
class PretrainConfig:
    """Configuration for pretraining."""
    
    # Model
    embed_dim: int = 128
    encoder_layers: int = 4
    decoder_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 512
    patch_size: int = 1
    dropout: float = 0.1
    
    # Masking
    mask_ratio: float = 0.4
    temporal_mask_prob: float = 0.3
    modality_mask_prob: float = 0.2
    contiguous_mask_prob: float = 0.3
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    max_grad_norm: float = 1.0
    
    # Hardware
    use_amp: bool = True
    num_workers: int = 4
    
    # Logging
    log_every: int = 100
    save_every: int = 10


class UnlabeledSensorDataset(Dataset):
    """Dataset for unlabeled sensor streams."""
    
    def __init__(
        self,
        data: dict[str, np.ndarray],
        seq_len: int = 24,
        stride: int = 6,
    ):
        """Initialize dataset.
        
        Args:
            data: Dict mapping modality to array (n_samples, features)
            seq_len: Sequence length to extract
            stride: Stride between sequences
        """
        self.data = {k: torch.from_numpy(v).float() for k, v in data.items()}
        self.seq_len = seq_len
        self.stride = stride
        
        # Compute number of sequences
        first_mod = list(self.data.keys())[0]
        n_samples = self.data[first_mod].shape[0]
        self.n_sequences = (n_samples - seq_len) // stride + 1
        
        self.modalities = list(data.keys())
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        
        return {
            mod: self.data[mod][start:end]
            for mod in self.modalities
        }


def collate_pretrain_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for pretraining."""
    modalities = list(batch[0].keys())
    
    return {
        mod: torch.stack([b[mod] for b in batch])
        for mod in modalities
    }


class PretrainTrainer:
    """Trainer for self-supervised pretraining."""
    
    def __init__(
        self,
        model: MaskedMultimodalAutoencoder,
        config: PretrainConfig,
        output_dir: Path,
        device: str = "auto",
    ):
        """Initialize trainer.
        
        Args:
            model: Masked autoencoder model
            config: Training configuration
            output_dir: Output directory
            device: Device to use
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
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # History
        self.history = {
            "loss": [],
            "lr": [],
        }
    
    def _create_scheduler(self, steps_per_epoch: int):
        """Create learning rate scheduler with warmup."""
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(
                self.config.min_lr / self.config.learning_rate,
                0.5 * (1 + np.cos(np.pi * progress))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> dict:
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_pretrain_batch,
            drop_last=True,
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_pretrain_batch,
            )
        
        # Create scheduler
        scheduler = self._create_scheduler(len(train_loader))
        
        logger.info(f"Starting pretraining on {len(train_dataset)} samples")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, scheduler, epoch)
            self.history["loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Log
            log_msg = (
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f}"
            )
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | Time: {epoch_time:.1f}s"
            logger.info(log_msg)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)
            
            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch + 1, is_best=True)
        
        # Save final model
        self._save_checkpoint(self.config.epochs, is_best=False)
        
        return self.history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        scheduler: LambdaLR,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Forward with mixed precision
            with autocast(enabled=self.use_amp):
                reconstructions, mask, targets = self.model(batch)
                loss, loss_dict = self.model.compute_loss(reconstructions, targets, mask)
            
            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                reconstructions, mask, targets = self.model(batch)
                loss, _ = self.model.compute_loss(reconstructions, targets, mask)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.output_dir / "checkpoint.pt")
        
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_pretrain.pt")
        
        # Also save encoder separately for easy loading
        torch.save(
            self.model.encoder.state_dict(),
            self.output_dir / "encoder.pt"
        )
    
    def load_checkpoint(self, path: Path) -> int:
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.history = checkpoint.get("history", self.history)
        
        return checkpoint["epoch"]

