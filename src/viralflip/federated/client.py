"""Federated learning client for on-device training.

Each client (device/user) performs local training and sends
model updates to the server without sharing raw data.

Reference: On-device Federated Learning Feasibility (PMC 2023)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClientUpdate:
    """Update from a federated client."""
    
    # Client identifier
    client_id: str
    
    # Model state dict (or delta from global model)
    state_dict: dict[str, torch.Tensor]
    
    # Number of local samples
    n_samples: int
    
    # Local training metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Whether this is a delta (difference from global)
    is_delta: bool = False
    
    # Training metadata
    n_epochs: int = 1
    learning_rate: float = 0.01


@dataclass
class ClientConfig:
    """Configuration for federated client."""
    
    # Training
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    
    # FedProx regularization (if using)
    proximal_mu: float = 0.0
    
    # Privacy
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Device
    device: str = "auto"
    
    # Personalization
    personalization_layers: list[str] = field(default_factory=list)
    freeze_encoder: bool = False


class LocalTrainer:
    """Trainer for local (on-device) model updates."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ClientConfig,
    ):
        """Initialize local trainer.
        
        Args:
            model: Local model copy
            config: Client configuration
        """
        self.model = model
        self.config = config
        
        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Freeze encoder if configured
        if config.freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
    
    def train(
        self,
        dataset: Dataset,
        global_model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Perform local training.
        
        Args:
            dataset: Local training data
            global_model: Global model (for FedProx regularization)
            loss_fn: Loss function
            
        Returns:
            Tuple of (state_dict, metrics)
        """
        self.model.train()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=len(dataset) > self.config.batch_size,
        )
        
        # Optimizer
        optimizer = SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            momentum=0.9,
        )
        
        # Save initial model for FedProx
        if self.config.proximal_mu > 0 and global_model is not None:
            global_params = {
                name: param.data.clone()
                for name, param in global_model.named_parameters()
            }
        else:
            global_params = None
        
        # Training loop
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                # Move to device
                batch = self._to_device(batch)
                
                # Forward
                outputs = self.model(**batch)
                
                if loss_fn is not None:
                    loss = loss_fn(outputs, batch.get('labels'))
                    if isinstance(loss, tuple):
                        loss = loss[0]
                else:
                    loss = outputs.get('loss', outputs)
                
                # FedProx regularization
                if global_params is not None and self.config.proximal_mu > 0:
                    prox_term = self._compute_proximal_term(global_params)
                    loss = loss + (self.config.proximal_mu / 2) * prox_term
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (for DP or stability)
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / max(n_batches, 1)
        metrics = {
            "loss": avg_loss,
            "n_batches": n_batches,
            "n_epochs": self.config.local_epochs,
        }
        
        return self.model.state_dict(), metrics
    
    def _compute_proximal_term(
        self,
        global_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute FedProx proximal regularization term."""
        prox_term = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in global_params:
                prox_term = prox_term + torch.sum(
                    (param - global_params[name]) ** 2
                )
        
        return prox_term
    
    def _to_device(self, batch: dict) -> dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


class FederatedClient:
    """Federated learning client.
    
    Represents a single participant (device/user) in federated learning.
    Handles:
    - Receiving global model updates
    - Local training
    - Computing and sending updates
    - On-device personalization
    """
    
    def __init__(
        self,
        client_id: str,
        model_class: type,
        model_kwargs: dict,
        config: ClientConfig,
    ):
        """Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model_class: Class for creating model
            model_kwargs: Kwargs for model construction
            config: Client configuration
        """
        self.client_id = client_id
        self.config = config
        
        # Create local model
        self.model = model_class(**model_kwargs)
        
        # Local trainer
        self.trainer = LocalTrainer(self.model, config)
        
        # Personalization layers (kept local, not sent to server)
        self.personal_state: Optional[dict[str, torch.Tensor]] = None
        
        # Training history
        self.update_history: list[ClientUpdate] = []
        self.n_rounds = 0
    
    def receive_global_model(
        self,
        global_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Receive and load global model from server.
        
        Args:
            global_state_dict: Global model state dict
        """
        # Load global weights
        self.model.load_state_dict(global_state_dict, strict=False)
        
        # Restore personal layers if any
        if self.personal_state is not None:
            for name, param in self.personal_state.items():
                if name in dict(self.model.named_parameters()):
                    self.model.state_dict()[name].copy_(param)
    
    def local_train(
        self,
        dataset: Dataset,
        global_model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> ClientUpdate:
        """Perform local training and create update.
        
        Args:
            dataset: Local training data
            global_model: Global model reference
            loss_fn: Loss function
            
        Returns:
            ClientUpdate to send to server
        """
        # Store personal layers before training
        self._save_personal_state()
        
        # Train
        state_dict, metrics = self.trainer.train(
            dataset, global_model, loss_fn
        )
        
        # Compute delta if configured
        if global_model is not None:
            delta_dict = self._compute_delta(
                state_dict,
                global_model.state_dict()
            )
            is_delta = True
        else:
            delta_dict = state_dict
            is_delta = False
        
        # Remove personal layers from update
        filtered_dict = self._filter_personal_layers(delta_dict)
        
        # Create update
        update = ClientUpdate(
            client_id=self.client_id,
            state_dict=filtered_dict,
            n_samples=len(dataset),
            metrics=metrics,
            is_delta=is_delta,
            n_epochs=self.config.local_epochs,
            learning_rate=self.config.learning_rate,
        )
        
        self.update_history.append(update)
        self.n_rounds += 1
        
        return update
    
    def _save_personal_state(self) -> None:
        """Save personal layer state."""
        if not self.config.personalization_layers:
            return
        
        self.personal_state = {}
        for name, param in self.model.named_parameters():
            for pattern in self.config.personalization_layers:
                if pattern in name:
                    self.personal_state[name] = param.data.clone()
                    break
    
    def _compute_delta(
        self,
        local_state: dict[str, torch.Tensor],
        global_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute parameter delta from global model."""
        delta = {}
        
        for name, param in local_state.items():
            if name in global_state:
                delta[name] = param - global_state[name]
            else:
                delta[name] = param
        
        return delta
    
    def _filter_personal_layers(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter out personalization layers from update."""
        if not self.config.personalization_layers:
            return state_dict
        
        filtered = {}
        for name, param in state_dict.items():
            is_personal = False
            for pattern in self.config.personalization_layers:
                if pattern in name:
                    is_personal = True
                    break
            
            if not is_personal:
                filtered[name] = param
        
        return filtered
    
    @torch.no_grad()
    def evaluate(
        self,
        dataset: Dataset,
        loss_fn: Optional[nn.Module] = None,
    ) -> dict[str, float]:
        """Evaluate model on local data.
        
        Args:
            dataset: Evaluation dataset
            loss_fn: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        n_correct = 0
        n_samples = 0
        
        device = self.trainer.device
        
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            outputs = self.model(**batch)
            
            if loss_fn is not None:
                loss = loss_fn(outputs, batch.get('labels'))
                if isinstance(loss, tuple):
                    loss = loss[0]
                total_loss += loss.item() * len(batch)
            
            n_samples += len(batch)
        
        return {
            "eval_loss": total_loss / max(n_samples, 1),
            "n_samples": n_samples,
        }
    
    def get_model(self) -> nn.Module:
        """Get current local model."""
        return self.model
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "client_id": self.client_id,
            "n_rounds": self.n_rounds,
            "n_updates": len(self.update_history),
            "has_personal_state": self.personal_state is not None,
        }

