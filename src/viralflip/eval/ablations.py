"""Ablation study runner for ViralFlip.

Mandatory ablations:
A) No PBM (population-only normalization)
B) No BDD (no confound removal)
C) Remove each modality family
D) No lag lattice (only lag=0)
E) No interactions (if used)
F) No personalization
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from viralflip.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation."""
    
    name: str
    config_changes: dict
    metrics: dict
    delta_auprc: float  # Change from baseline
    delta_auroc: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config_changes": self.config_changes,
            "metrics": self.metrics,
            "delta_auprc": self.delta_auprc,
            "delta_auroc": self.delta_auroc,
        }


class AblationRunner:
    """Run ablation studies."""
    
    # Standard ablation configurations
    ABLATIONS = {
        "no_pbm": {
            "name": "No Personal Baseline Memory",
            "description": "Use population-level normalization instead of PBM",
            "config": {"pbm": {"enabled": False}},
        },
        "no_bdd": {
            "name": "No Behavior-Drift Debiasing",
            "description": "Skip confound removal",
            "config": {"bdd": {"enabled": False}},
        },
        "no_voice": {
            "name": "Remove Voice Modality",
            "config": {"modalities": {"voice": False}},
        },
        "no_cough": {
            "name": "Remove Cough Modality",
            "config": {"modalities": {"cough": False}},
        },
        "no_rppg": {
            "name": "Remove rPPG Modality",
            "config": {"modalities": {"rppg": False}},
        },
        "no_gait_tap": {
            "name": "Remove Gait + Tapping",
            "config": {"modalities": {"gait_active": False, "tap": False}},
        },
        "no_gps": {
            "name": "Remove GPS Mobility",
            "config": {"modalities": {"gps": False}},
        },
        "no_ambient": {
            "name": "Remove Ambient (Light + Baro)",
            "config": {"modalities": {"light": False, "baro": False}},
        },
        "no_screen": {
            "name": "Remove Screen Events",
            "config": {"modalities": {"screen": False}},
        },
        "no_lag": {
            "name": "No Lag Lattice (lag=0 only)",
            "config": {"model": {"max_lag_bins": 0}},
        },
        "no_interactions": {
            "name": "No Interactions",
            "config": {"model": {"use_interactions": False}},
        },
        "no_personalization": {
            "name": "No Personalization",
            "config": {"personalization": {"enabled": False}},
        },
    }
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        base_config: dict,
        output_dir: Path,
    ):
        """Initialize ablation runner.
        
        Args:
            train_fn: Function that trains model given config, returns model.
            eval_fn: Function that evaluates model, returns metrics dict.
            base_config: Base configuration dict.
            output_dir: Directory for outputs.
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_metrics: Optional[dict] = None
        self.results: list[AblationResult] = []
    
    def run_baseline(self) -> dict:
        """Run baseline (full model) and store metrics.
        
        Returns:
            Baseline metrics dict.
        """
        logger.info("Running baseline model...")
        
        model = self.train_fn(self.base_config)
        self.baseline_metrics = self.eval_fn(model)
        
        logger.info(f"Baseline AUPRC: {self.baseline_metrics.get('auprc_mean', 0):.4f}")
        logger.info(f"Baseline AUROC: {self.baseline_metrics.get('auroc_mean', 0):.4f}")
        
        return self.baseline_metrics
    
    def run_ablation(
        self,
        ablation_key: str,
        ablation_config: Optional[dict] = None,
    ) -> AblationResult:
        """Run a single ablation.
        
        Args:
            ablation_key: Key from ABLATIONS dict.
            ablation_config: Optional custom ablation config.
            
        Returns:
            AblationResult object.
        """
        if self.baseline_metrics is None:
            self.run_baseline()
        
        # Get ablation config
        if ablation_config is None:
            if ablation_key not in self.ABLATIONS:
                raise ValueError(f"Unknown ablation: {ablation_key}")
            ablation_info = self.ABLATIONS[ablation_key]
        else:
            ablation_info = ablation_config
        
        name = ablation_info.get("name", ablation_key)
        config_changes = ablation_info.get("config", {})
        
        logger.info(f"Running ablation: {name}")
        
        # Merge config
        ablation_cfg = self._merge_config(self.base_config, config_changes)
        
        # Train and evaluate
        try:
            model = self.train_fn(ablation_cfg)
            metrics = self.eval_fn(model)
        except Exception as e:
            logger.error(f"Ablation {name} failed: {e}")
            metrics = {"auprc_mean": 0.0, "auroc_mean": 0.0}
        
        # Compute deltas
        delta_auprc = metrics.get("auprc_mean", 0) - self.baseline_metrics.get("auprc_mean", 0)
        delta_auroc = metrics.get("auroc_mean", 0) - self.baseline_metrics.get("auroc_mean", 0)
        
        result = AblationResult(
            name=name,
            config_changes=config_changes,
            metrics=metrics,
            delta_auprc=delta_auprc,
            delta_auroc=delta_auroc,
        )
        
        self.results.append(result)
        
        logger.info(f"  AUPRC: {metrics.get('auprc_mean', 0):.4f} (Δ={delta_auprc:+.4f})")
        logger.info(f"  AUROC: {metrics.get('auroc_mean', 0):.4f} (Δ={delta_auroc:+.4f})")
        
        return result
    
    def run_all_ablations(
        self,
        ablation_keys: Optional[list[str]] = None,
    ) -> list[AblationResult]:
        """Run all specified ablations.
        
        Args:
            ablation_keys: List of ablation keys to run. If None, runs all.
            
        Returns:
            List of AblationResult objects.
        """
        if ablation_keys is None:
            ablation_keys = list(self.ABLATIONS.keys())
        
        # Run baseline first
        if self.baseline_metrics is None:
            self.run_baseline()
        
        # Run each ablation
        for key in tqdm(ablation_keys, desc="Ablations"):
            self.run_ablation(key)
        
        return self.results
    
    def _merge_config(self, base: dict, changes: dict) -> dict:
        """Deep merge config changes into base config.
        
        Args:
            base: Base configuration.
            changes: Changes to apply.
            
        Returns:
            Merged configuration.
        """
        import copy
        result = copy.deepcopy(base)
        
        def _merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    _merge(d1[k], v)
                else:
                    d1[k] = v
        
        _merge(result, changes)
        return result
    
    def get_summary(self) -> dict:
        """Get summary of all ablation results.
        
        Returns:
            Summary dict.
        """
        summary = {
            "baseline": self.baseline_metrics,
            "ablations": [r.to_dict() for r in self.results],
        }
        
        # Find most impactful ablations
        if self.results:
            sorted_by_impact = sorted(self.results, key=lambda r: r.delta_auprc)
            summary["most_impactful"] = sorted_by_impact[0].name
            summary["least_impactful"] = sorted_by_impact[-1].name
        
        return summary
    
    def save_results(self, filename: str = "ablation_results.json") -> None:
        """Save results to JSON file.
        
        Args:
            filename: Output filename.
        """
        import json
        
        summary = self.get_summary()
        
        with open(self.output_dir / filename, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved ablation results to {self.output_dir / filename}")

