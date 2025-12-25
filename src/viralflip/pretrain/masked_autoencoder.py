"""Masked Multimodal Autoencoder for Self-Supervised Pretraining.

Implements masked time-series modeling for multi-sensor health data:
1. Patch-based tokenization of time-series across modalities
2. Random masking of patches (time + sensor channels)
3. Transformer encoder with cross-modal attention
4. Lightweight decoder for reconstruction

This learns robust representations that handle:
- Missing sensors
- Noisy phone data
- Distribution shifts

Reference: Multi-scale Spatial-temporal Masked Self-supervised Pre-training (IJCAI 2025)
"""

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskConfig:
    """Configuration for masking strategy."""
    
    mask_ratio: float = 0.4  # Fraction of patches to mask
    temporal_mask_prob: float = 0.3  # Prob of masking entire time column
    modality_mask_prob: float = 0.2  # Prob of masking entire modality row
    contiguous_mask_prob: float = 0.3  # Prob of contiguous temporal masking
    min_contiguous_len: int = 2
    max_contiguous_len: int = 6


def create_mask_strategy(config: MaskConfig):
    """Create a masking strategy from config."""
    return MaskingStrategy(
        mask_ratio=config.mask_ratio,
        temporal_mask_prob=config.temporal_mask_prob,
        modality_mask_prob=config.modality_mask_prob,
        contiguous_mask_prob=config.contiguous_mask_prob,
        min_contiguous_len=config.min_contiguous_len,
        max_contiguous_len=config.max_contiguous_len,
    )


class MaskingStrategy(nn.Module):
    """Generates masks for time x modality patches.
    
    Supports multiple masking patterns:
    - Random: Random patches masked
    - Temporal: Entire time columns masked (simulates missing data windows)
    - Modality: Entire modality rows masked (simulates missing sensors)
    - Contiguous: Contiguous temporal blocks masked
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.4,
        temporal_mask_prob: float = 0.3,
        modality_mask_prob: float = 0.2,
        contiguous_mask_prob: float = 0.3,
        min_contiguous_len: int = 2,
        max_contiguous_len: int = 6,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.temporal_mask_prob = temporal_mask_prob
        self.modality_mask_prob = modality_mask_prob
        self.contiguous_mask_prob = contiguous_mask_prob
        self.min_contiguous_len = min_contiguous_len
        self.max_contiguous_len = max_contiguous_len
    
    def forward(
        self,
        batch_size: int,
        n_time_patches: int,
        n_modalities: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate mask for batch.
        
        Args:
            batch_size: Number of samples
            n_time_patches: Number of time patches
            n_modalities: Number of modalities
            device: Device for tensors
            
        Returns:
            Boolean mask of shape (batch, n_time_patches, n_modalities)
            True = masked (to predict), False = visible
        """
        masks = []
        
        for _ in range(batch_size):
            # Choose masking strategy
            rand = torch.rand(1).item()
            
            if rand < self.temporal_mask_prob:
                mask = self._temporal_mask(n_time_patches, n_modalities, device)
            elif rand < self.temporal_mask_prob + self.modality_mask_prob:
                mask = self._modality_mask(n_time_patches, n_modalities, device)
            elif rand < self.temporal_mask_prob + self.modality_mask_prob + self.contiguous_mask_prob:
                mask = self._contiguous_mask(n_time_patches, n_modalities, device)
            else:
                mask = self._random_mask(n_time_patches, n_modalities, device)
            
            masks.append(mask)
        
        return torch.stack(masks)
    
    def _random_mask(
        self, n_time: int, n_mod: int, device: torch.device
    ) -> torch.Tensor:
        """Random patch masking."""
        n_patches = n_time * n_mod
        n_mask = int(n_patches * self.mask_ratio)
        
        # Random permutation
        indices = torch.randperm(n_patches, device=device)
        mask_flat = torch.zeros(n_patches, dtype=torch.bool, device=device)
        mask_flat[indices[:n_mask]] = True
        
        return mask_flat.view(n_time, n_mod)
    
    def _temporal_mask(
        self, n_time: int, n_mod: int, device: torch.device
    ) -> torch.Tensor:
        """Mask entire time columns (simulates missing windows)."""
        n_mask_cols = max(1, int(n_time * self.mask_ratio))
        
        indices = torch.randperm(n_time, device=device)[:n_mask_cols]
        mask = torch.zeros(n_time, n_mod, dtype=torch.bool, device=device)
        mask[indices, :] = True
        
        return mask
    
    def _modality_mask(
        self, n_time: int, n_mod: int, device: torch.device
    ) -> torch.Tensor:
        """Mask entire modality rows (simulates missing sensors)."""
        n_mask_rows = max(1, int(n_mod * self.mask_ratio))
        
        indices = torch.randperm(n_mod, device=device)[:n_mask_rows]
        mask = torch.zeros(n_time, n_mod, dtype=torch.bool, device=device)
        mask[:, indices] = True
        
        return mask
    
    def _contiguous_mask(
        self, n_time: int, n_mod: int, device: torch.device
    ) -> torch.Tensor:
        """Mask contiguous temporal blocks."""
        mask = torch.zeros(n_time, n_mod, dtype=torch.bool, device=device)
        n_target = int(n_time * n_mod * self.mask_ratio)
        n_masked = 0
        
        while n_masked < n_target:
            # Random start position
            t_start = torch.randint(0, n_time, (1,)).item()
            m_idx = torch.randint(0, n_mod, (1,)).item()
            
            # Random length
            length = torch.randint(
                self.min_contiguous_len, 
                self.max_contiguous_len + 1, 
                (1,)
            ).item()
            
            t_end = min(t_start + length, n_time)
            mask[t_start:t_end, m_idx] = True
            n_masked = mask.sum().item()
        
        return mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time dimension."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Embed time-series patches into tokens.
    
    Each modality has its own linear projection, but they share
    the same embedding dimension for cross-modal attention.
    """
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        patch_size: int = 1,  # Number of time steps per patch
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize patch embedding.
        
        Args:
            modality_dims: Dict mapping modality name to feature dimension
            patch_size: Number of time steps per patch
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.n_modalities = len(self.modalities)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Per-modality projection
        self.projections = nn.ModuleDict()
        for mod, dim in modality_dims.items():
            input_dim = dim * patch_size
            self.projections[mod] = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        
        # Modality embedding (learnable)
        self.modality_embed = nn.Embedding(self.n_modalities, embed_dim)
        
        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
    
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed multimodal time series into patch tokens.
        
        Args:
            x_dict: Dict mapping modality to tensor (batch, seq, features)
            missing_mask: Optional dict mapping modality to missing mask (batch, seq)
            
        Returns:
            tokens: (batch, n_patches, n_modalities, embed_dim)
            mask: (batch, n_patches, n_modalities) True = missing
        """
        # Get shapes from first modality
        first_mod = self.modalities[0]
        batch_size, seq_len, _ = x_dict[first_mod].shape
        n_patches = seq_len // self.patch_size
        device = x_dict[first_mod].device
        
        # Initialize output tensors
        tokens = torch.zeros(
            batch_size, n_patches, self.n_modalities, self.embed_dim,
            device=device
        )
        mask = torch.zeros(
            batch_size, n_patches, self.n_modalities,
            dtype=torch.bool, device=device
        )
        
        for m_idx, modality in enumerate(self.modalities):
            if modality not in x_dict:
                mask[:, :, m_idx] = True
                continue
            
            x = x_dict[modality]  # (batch, seq, features)
            
            # Reshape into patches
            x = x[:, :n_patches * self.patch_size, :]
            x = x.view(batch_size, n_patches, -1)  # (batch, n_patches, patch_size * features)
            
            # Project
            x = self.projections[modality](x)  # (batch, n_patches, embed_dim)
            
            # Add modality embedding
            mod_emb = self.modality_embed(
                torch.tensor([m_idx], device=device)
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, embed_dim)
            x = x + mod_emb.squeeze(2)
            
            tokens[:, :, m_idx, :] = x
            
            # Handle missing mask
            if missing_mask is not None and modality in missing_mask:
                m = missing_mask[modality]
                # Downsample mask to patch level
                m = m[:, :n_patches * self.patch_size]
                m = m.view(batch_size, n_patches, self.patch_size).any(dim=-1)
                mask[:, :, m_idx] = m
        
        # Add temporal positional encoding
        # Reshape to (batch * n_mod, n_patches, embed_dim) for pos encoding
        tokens = tokens.permute(0, 2, 1, 3)  # (batch, n_mod, n_patches, embed_dim)
        tokens = tokens.reshape(batch_size * self.n_modalities, n_patches, self.embed_dim)
        tokens = self.pos_encoding(tokens)
        tokens = tokens.view(batch_size, self.n_modalities, n_patches, self.embed_dim)
        tokens = tokens.permute(0, 2, 1, 3)  # (batch, n_patches, n_mod, embed_dim)
        
        return tokens, mask


class CrossModalAttention(nn.Module):
    """Cross-modal attention layer.
    
    Allows information flow between modalities at the same time step.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-modal attention.
        
        Args:
            x: (batch, n_patches, n_modalities, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output of same shape
        """
        batch, n_patches, n_mod, embed_dim = x.shape
        
        # Reshape: treat each time step as a separate sequence
        x = x.view(batch * n_patches, n_mod, embed_dim)
        
        # Self-attention across modalities
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        
        return x.view(batch, n_patches, n_mod, embed_dim)


class TemporalAttention(nn.Module):
    """Temporal self-attention layer.
    
    Captures temporal dependencies within each modality.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal attention.
        
        Args:
            x: (batch, n_patches, n_modalities, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output of same shape
        """
        batch, n_patches, n_mod, embed_dim = x.shape
        
        # Reshape: process each modality separately
        x = x.permute(0, 2, 1, 3).reshape(batch * n_mod, n_patches, embed_dim)
        
        # Self-attention across time
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        
        x = x.view(batch, n_mod, n_patches, embed_dim).permute(0, 2, 1, 3)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with cross-modal and temporal attention."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_cross_modal: bool = True,
    ):
        super().__init__()
        
        self.use_cross_modal = use_cross_modal
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(embed_dim, n_heads, dropout)
        
        # Cross-modal attention (optional)
        if use_cross_modal:
            self.cross_modal_attn = CrossModalAttention(embed_dim, n_heads, dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, n_patches, n_modalities, embed_dim)
            mask: Optional mask
            
        Returns:
            Output of same shape
        """
        # Temporal attention
        x = self.temporal_attn(x, mask)
        
        # Cross-modal attention
        if self.use_cross_modal:
            x = self.cross_modal_attn(x, mask)
        
        # Feed-forward
        x = x + self.ff(x)
        x = self.ff_norm(x)
        
        return x


class MultimodalTimeSeriesEncoder(nn.Module):
    """Transformer encoder for multimodal time series.
    
    Processes multiple sensor modalities with:
    - Per-modality patch embedding
    - Cross-modal attention for sensor fusion
    - Temporal attention for sequence modeling
    """
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 512,
        patch_size: int = 1,
        dropout: float = 0.1,
        use_cross_modal: bool = True,
    ):
        """Initialize encoder.
        
        Args:
            modality_dims: Dict mapping modality name to feature dimension
            embed_dim: Embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            patch_size: Time steps per patch
            dropout: Dropout rate
            use_cross_modal: Whether to use cross-modal attention
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            modality_dims, patch_size, embed_dim, dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout, use_cross_modal)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
        return_all_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal time series.
        
        Args:
            x_dict: Dict mapping modality to tensor (batch, seq, features)
            missing_mask: Optional dict mapping modality to missing mask
            return_all_tokens: If True, return all tokens; else return pooled
            
        Returns:
            If return_all_tokens:
                tokens: (batch, n_patches, n_modalities, embed_dim)
                mask: (batch, n_patches, n_modalities)
            Else:
                pooled: (batch, embed_dim)
                tokens: (batch, n_patches, n_modalities, embed_dim)
        """
        # Patch embedding
        tokens, mask = self.patch_embed(x_dict, missing_mask)
        
        # Transformer layers
        for layer in self.layers:
            tokens = layer(tokens, mask)
        
        tokens = self.norm(tokens)
        
        if return_all_tokens:
            return tokens, mask
        
        # Pool: mean over visible tokens
        visible_mask = ~mask  # (batch, n_patches, n_mod)
        visible_tokens = tokens * visible_mask.unsqueeze(-1).float()
        
        # Mean pooling
        n_visible = visible_mask.sum(dim=(1, 2), keepdim=True).unsqueeze(-1).clamp(min=1)
        pooled = visible_tokens.sum(dim=(1, 2)) / n_visible.squeeze()
        
        return pooled, tokens


class TimeSeriesDecoder(nn.Module):
    """Lightweight decoder for masked reconstruction.
    
    Reconstructs original features from masked tokens.
    Uses a simple MLP per modality.
    """
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        embed_dim: int = 128,
        hidden_dim: int = 256,
        patch_size: int = 1,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize decoder.
        
        Args:
            modality_dims: Dict mapping modality name to feature dimension
            embed_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            patch_size: Time steps per patch
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Shared decoder transformer
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=4, ff_dim=hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Per-modality reconstruction heads
        self.heads = nn.ModuleDict()
        for mod, dim in modality_dims.items():
            output_dim = dim * patch_size
            self.heads[mod] = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def forward(
        self,
        encoder_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode masked tokens.
        
        Args:
            encoder_tokens: (batch, n_patches, n_modalities, embed_dim)
            mask: (batch, n_patches, n_modalities) True = masked
            
        Returns:
            Dict mapping modality to reconstructed features
            Each tensor: (batch, n_patches, patch_size * features)
        """
        batch, n_patches, n_mod, embed_dim = encoder_tokens.shape
        
        # Replace masked tokens with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(encoder_tokens)
        mask_tokens = self.mask_token.expand(batch, n_patches, n_mod, -1)
        
        decoder_input = torch.where(mask_expanded, mask_tokens, encoder_tokens)
        
        # Decoder transformer
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input)
        
        # Per-modality reconstruction
        reconstructions = {}
        for m_idx, modality in enumerate(self.modalities):
            mod_tokens = decoder_input[:, :, m_idx, :]  # (batch, n_patches, embed_dim)
            reconstructions[modality] = self.heads[modality](mod_tokens)
        
        return reconstructions


class MaskedMultimodalAutoencoder(nn.Module):
    """Masked Autoencoder for multimodal time series pretraining.
    
    Combines:
    - Multimodal transformer encoder
    - Masking strategy (time + modality + contiguous)
    - Lightweight decoder for reconstruction
    
    The encoder learns representations that are robust to:
    - Missing sensors
    - Noisy data
    - Temporal gaps
    """
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        embed_dim: int = 128,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 512,
        patch_size: int = 1,
        dropout: float = 0.1,
        mask_config: Optional[MaskConfig] = None,
    ):
        """Initialize masked autoencoder.
        
        Args:
            modality_dims: Dict mapping modality name to feature dimension
            embed_dim: Embedding dimension
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            n_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            patch_size: Time steps per patch
            dropout: Dropout rate
            mask_config: Masking configuration
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Encoder
        self.encoder = MultimodalTimeSeriesEncoder(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            n_layers=encoder_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = TimeSeriesDecoder(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            hidden_dim=ff_dim,
            patch_size=patch_size,
            n_layers=decoder_layers,
            dropout=dropout,
        )
        
        # Masking
        self.mask_config = mask_config or MaskConfig()
        self.masking = create_mask_strategy(self.mask_config)
    
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
        apply_mask: bool = True,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with masking and reconstruction.
        
        Args:
            x_dict: Dict mapping modality to tensor (batch, seq, features)
            missing_mask: Optional dict mapping modality to missing mask
            apply_mask: Whether to apply masking (False for inference)
            
        Returns:
            reconstructions: Dict mapping modality to reconstructed features
            mask: (batch, n_patches, n_modalities) mask used
            targets: Dict mapping modality to original patch features
        """
        # Get dimensions
        first_mod = self.modalities[0]
        batch_size, seq_len, _ = x_dict[first_mod].shape
        n_patches = seq_len // self.patch_size
        device = x_dict[first_mod].device
        
        # Encode
        tokens, data_mask = self.encoder(x_dict, missing_mask, return_all_tokens=True)
        
        # Generate mask
        if apply_mask:
            mask = self.masking(batch_size, n_patches, len(self.modalities), device)
            # Combine with data mask (don't try to reconstruct actually missing data)
            mask = mask | data_mask
        else:
            mask = data_mask
        
        # Decode
        reconstructions = self.decoder(tokens, mask)
        
        # Build targets (original patches)
        targets = {}
        for modality in self.modalities:
            if modality not in x_dict:
                continue
            x = x_dict[modality]
            x = x[:, :n_patches * self.patch_size, :]
            x = x.view(batch_size, n_patches, -1)
            targets[modality] = x
        
        return reconstructions, mask, targets
    
    def compute_loss(
        self,
        reconstructions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reconstruction loss on masked patches only.
        
        Args:
            reconstructions: Dict mapping modality to reconstructed patches
            targets: Dict mapping modality to original patches
            mask: (batch, n_patches, n_modalities) True = masked
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Per-modality losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=mask.device)
        
        for m_idx, modality in enumerate(self.modalities):
            if modality not in reconstructions or modality not in targets:
                continue
            
            recon = reconstructions[modality]
            target = targets[modality]
            mod_mask = mask[:, :, m_idx]  # (batch, n_patches)
            
            # MSE loss on masked patches only
            mse = F.mse_loss(recon, target, reduction='none')
            mse = mse.mean(dim=-1)  # (batch, n_patches)
            
            # Apply mask
            masked_mse = mse * mod_mask.float()
            n_masked = mod_mask.float().sum().clamp(min=1)
            
            mod_loss = masked_mse.sum() / n_masked
            losses[modality] = mod_loss
            total_loss = total_loss + mod_loss
        
        # Average across modalities
        total_loss = total_loss / max(len(losses), 1)
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def get_encoder(self) -> MultimodalTimeSeriesEncoder:
        """Get the pretrained encoder for downstream tasks."""
        return self.encoder
    
    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode input to representation (for downstream use).
        
        Args:
            x_dict: Dict mapping modality to tensor (batch, seq, features)
            missing_mask: Optional dict mapping modality to missing mask
            
        Returns:
            Pooled representation: (batch, embed_dim)
        """
        pooled, _ = self.encoder(x_dict, missing_mask, return_all_tokens=False)
        return pooled

