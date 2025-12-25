"""Federated learning server for aggregating client updates.

The server coordinates federated training:
- Distributes global model to clients
- Aggregates client updates
- Manages training rounds
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import copy

import numpy as np
import torch
import torch.nn as nn

from viralflip.federated.client import ClientUpdate


@dataclass
class ServerConfig:
    """Configuration for federated server."""
    
    # Aggregation
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, scaffold
    min_clients_per_round: int = 2
    client_fraction: float = 1.0  # Fraction of clients to sample per round
    
    # Momentum (for FedAvgM)
    server_momentum: float = 0.0
    
    # FedProx
    proximal_mu: float = 0.0
    
    # Privacy
    use_secure_aggregation: bool = False
    use_differential_privacy: bool = False
    dp_noise_multiplier: float = 1.0
    
    # Training
    n_rounds: int = 100
    eval_every: int = 5


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""
    
    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate client updates into new global model.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
            
        Returns:
            New global state dict
        """
        pass


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg).
    
    Weighted average of client parameters based on sample count.
    """
    
    def __init__(self, momentum: float = 0.0):
        """Initialize FedAvg.
        
        Args:
            momentum: Server-side momentum (for FedAvgM)
        """
        self.momentum = momentum
        self.velocity = None
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate using weighted averaging."""
        if not client_updates:
            return global_model.state_dict()
        
        # Total samples
        total_samples = sum(u.n_samples for u in client_updates)
        
        # Initialize aggregated state
        aggregated = {}
        global_state = global_model.state_dict()
        
        for name, param in global_state.items():
            aggregated[name] = torch.zeros_like(param, dtype=torch.float32)
        
        # Weighted average
        for update in client_updates:
            weight = update.n_samples / total_samples
            
            for name, param in update.state_dict.items():
                if name in aggregated:
                    if update.is_delta:
                        # Add weighted delta
                        aggregated[name] += weight * param.float()
                    else:
                        # Weighted average of full parameters
                        aggregated[name] += weight * param.float()
        
        # Convert back and add to global if delta
        new_state = {}
        for name, param in global_state.items():
            if name in aggregated:
                if client_updates[0].is_delta:
                    # Apply delta to global
                    update_value = aggregated[name]
                    
                    # Apply momentum if configured
                    if self.momentum > 0:
                        if self.velocity is None:
                            self.velocity = {}
                        if name not in self.velocity:
                            self.velocity[name] = torch.zeros_like(update_value)
                        
                        self.velocity[name] = (
                            self.momentum * self.velocity[name] + update_value
                        )
                        update_value = self.velocity[name]
                    
                    new_state[name] = param + update_value.to(param.dtype)
                else:
                    new_state[name] = aggregated[name].to(param.dtype)
            else:
                new_state[name] = param
        
        return new_state


class FedProx(AggregationStrategy):
    """FedProx aggregation.
    
    Same as FedAvg but clients use proximal regularization.
    """
    
    def __init__(self, mu: float = 0.01):
        """Initialize FedProx.
        
        Args:
            mu: Proximal regularization weight
        """
        self.mu = mu
        self.fedavg = FedAvg()
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate using FedAvg (regularization is client-side)."""
        return self.fedavg.aggregate(global_model, client_updates)


class SecureAggregation:
    """Secure aggregation for privacy-preserving updates.
    
    Simulates secure aggregation protocol where server only
    sees the aggregate, not individual updates.
    
    In practice, this would use cryptographic protocols.
    """
    
    def __init__(self, threshold: int = 2):
        """Initialize secure aggregation.
        
        Args:
            threshold: Minimum clients for aggregation
        """
        self.threshold = threshold
    
    def aggregate(
        self,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Securely aggregate updates.
        
        Args:
            updates: Client updates
            
        Returns:
            Aggregated update (sum)
        """
        if len(updates) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} clients for secure aggregation"
            )
        
        # In real implementation, this would use secure multi-party computation
        # Here we simulate by just summing
        
        aggregated = {}
        for update in updates:
            for name, param in update.state_dict.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(param, dtype=torch.float32)
                aggregated[name] += param.float()
        
        return aggregated


class FederatedServer:
    """Federated learning server.
    
    Coordinates federated training:
    - Manages global model
    - Selects clients per round
    - Aggregates updates
    - Tracks training progress
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ServerConfig,
    ):
        """Initialize server.
        
        Args:
            model: Global model
            config: Server configuration
        """
        self.global_model = model
        self.config = config
        
        # Aggregation strategy
        if config.aggregation_strategy == "fedavg":
            self.aggregator = FedAvg(momentum=config.server_momentum)
        elif config.aggregation_strategy == "fedprox":
            self.aggregator = FedProx(mu=config.proximal_mu)
        else:
            self.aggregator = FedAvg()
        
        # Secure aggregation
        if config.use_secure_aggregation:
            self.secure_agg = SecureAggregation(
                threshold=config.min_clients_per_round
            )
        else:
            self.secure_agg = None
        
        # Training state
        self.current_round = 0
        self.history = {
            "rounds": [],
            "n_clients": [],
            "aggregated_metrics": [],
        }
    
    def get_global_model(self) -> nn.Module:
        """Get current global model."""
        return self.global_model
    
    def get_global_state_dict(self) -> dict[str, torch.Tensor]:
        """Get global model state dict for distribution."""
        return copy.deepcopy(self.global_model.state_dict())
    
    def select_clients(
        self,
        available_clients: list[str],
    ) -> list[str]:
        """Select clients for current round.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            Selected client IDs
        """
        n_available = len(available_clients)
        n_select = max(
            self.config.min_clients_per_round,
            int(n_available * self.config.client_fraction)
        )
        n_select = min(n_select, n_available)
        
        # Random selection
        selected = np.random.choice(
            available_clients,
            size=n_select,
            replace=False
        ).tolist()
        
        return selected
    
    def aggregate_round(
        self,
        client_updates: list[ClientUpdate],
    ) -> dict[str, float]:
        """Aggregate client updates for a round.
        
        Args:
            client_updates: Updates from participating clients
            
        Returns:
            Aggregated metrics
        """
        if len(client_updates) < self.config.min_clients_per_round:
            raise ValueError(
                f"Need at least {self.config.min_clients_per_round} clients"
            )
        
        # Aggregate
        new_state = self.aggregator.aggregate(self.global_model, client_updates)
        
        # Add DP noise if configured
        if self.config.use_differential_privacy:
            new_state = self._add_dp_noise(new_state)
        
        # Update global model
        self.global_model.load_state_dict(new_state)
        
        # Aggregate metrics
        total_samples = sum(u.n_samples for u in client_updates)
        avg_loss = sum(
            u.metrics.get("loss", 0) * u.n_samples for u in client_updates
        ) / total_samples
        
        metrics = {
            "n_clients": len(client_updates),
            "total_samples": total_samples,
            "avg_loss": avg_loss,
        }
        
        # Update history
        self.current_round += 1
        self.history["rounds"].append(self.current_round)
        self.history["n_clients"].append(len(client_updates))
        self.history["aggregated_metrics"].append(metrics)
        
        return metrics
    
    def _add_dp_noise(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add differential privacy noise to aggregated parameters."""
        noisy_state = {}
        
        for name, param in state_dict.items():
            noise = torch.randn_like(param) * self.config.dp_noise_multiplier
            noisy_state[name] = param + noise
        
        return noisy_state
    
    def run_round(
        self,
        clients: dict,  # client_id -> FederatedClient
        datasets: dict,  # client_id -> Dataset
        loss_fn: Optional[nn.Module] = None,
    ) -> dict[str, float]:
        """Run a complete federated round.
        
        Args:
            clients: Dict of client_id -> FederatedClient
            datasets: Dict of client_id -> local dataset
            loss_fn: Loss function
            
        Returns:
            Round metrics
        """
        # Select clients
        available = list(clients.keys())
        selected = self.select_clients(available)
        
        # Distribute global model
        global_state = self.get_global_state_dict()
        for client_id in selected:
            clients[client_id].receive_global_model(global_state)
        
        # Collect updates
        updates = []
        for client_id in selected:
            client = clients[client_id]
            dataset = datasets.get(client_id)
            
            if dataset is None or len(dataset) == 0:
                continue
            
            update = client.local_train(
                dataset,
                global_model=self.global_model,
                loss_fn=loss_fn,
            )
            updates.append(update)
        
        # Aggregate
        metrics = self.aggregate_round(updates)
        
        return metrics
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "current_round": self.current_round,
            "n_rounds_completed": len(self.history["rounds"]),
            "avg_clients_per_round": np.mean(self.history["n_clients"]) if self.history["n_clients"] else 0,
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save server checkpoint."""
        import torch
        
        checkpoint = {
            "global_model": self.global_model.state_dict(),
            "current_round": self.current_round,
            "history": self.history,
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load server checkpoint."""
        import torch
        
        checkpoint = torch.load(path, map_location="cpu")
        
        self.global_model.load_state_dict(checkpoint["global_model"])
        self.current_round = checkpoint["current_round"]
        self.history = checkpoint["history"]

