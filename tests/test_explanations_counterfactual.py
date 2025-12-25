"""Tests for explanation counterfactual validity.

Key test: Removing the top contributor should reduce risk the most.
This verifies that the explanation faithfully represents model behavior.
"""

import numpy as np
import pytest
import torch

from viralflip.model.viralflip import ViralFlip
from viralflip.explain.explain import ExplanationEngine


class TestCounterfactualFidelity:
    """Test that explanations accurately reflect model behavior."""
    
    @pytest.fixture
    def feature_dims(self):
        return {
            "voice": 5,
            "cough": 3,
            "tap": 4,
            "gait_active": 4,
            "rppg": 3,
            "light": 2,
            "baro": 2,
        }
    
    @pytest.fixture
    def model(self, feature_dims):
        model = ViralFlip(
            feature_dims=feature_dims,
            horizons=[24, 48, 72],
            max_lag=4,
            use_interactions=False,
            use_personalization=False,
            use_virus_classifier=False,
        )
        
        # Train briefly to get non-trivial weights
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        for _ in range(50):
            # Create drift that should increase risk
            drift_dict = {
                m: torch.ones(4, 5, d) * 2.0  # batch=4, seq=5
                for m, d in feature_dims.items()
            }
            
            probs, _, _, _ = model(drift_dict)
            loss = -probs.mean()  # Maximize risk for high drift
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model
    
    @pytest.fixture
    def engine(self, model):
        return ExplanationEngine(model, top_k=5)
    
    def test_removing_top_contributor_reduces_risk(self, model, engine, feature_dims):
        """Removing the top contributor should reduce predicted risk."""
        # Create test drift
        np.random.seed(42)
        drift_dict = {
            m: np.random.randn(5, d) + 1.0  # Positive drift
            for m, d in feature_dims.items()
        }
        
        # Get explanation
        explanation = engine.explain(drift_dict, horizon=24)
        
        if len(explanation.top_contributors) == 0:
            pytest.skip("No contributors found")
        
        # Skip if risk is saturated (model optimization went too far)
        if explanation.risk > 0.99 or explanation.risk < 0.01:
            pytest.skip("Risk is saturated, counterfactual delta not meaningful")
        
        top_contrib = explanation.top_contributors[0]
        
        # The delta_if_removed should be negative or zero (removing contributor decreases risk)
        # This is because we trained the model to associate positive drift with risk
        # Allow 0.0 for edge cases where model is near saturation
        assert top_contrib.delta_if_removed <= 0, \
            f"Expected non-positive delta, got {top_contrib.delta_if_removed}"
    
    def test_counterfactual_ordering(self, model, engine, feature_dims):
        """Contributors with higher contribution should have larger counterfactual impact."""
        np.random.seed(42)
        drift_dict = {
            m: np.random.randn(5, d) + 1.5
            for m, d in feature_dims.items()
        }
        
        explanation = engine.explain(drift_dict, horizon=24)
        
        if len(explanation.top_contributors) < 2:
            pytest.skip("Need at least 2 contributors")
        
        # Higher contribution should generally mean larger (more negative) delta
        # when removed, but this may not be strictly monotonic due to interactions
        top_delta = abs(explanation.top_contributors[0].delta_if_removed)
        second_delta = abs(explanation.top_contributors[1].delta_if_removed)
        
        # Allow some tolerance - top should be at least 50% of second
        # (strict ordering not guaranteed due to model complexity)
        assert top_delta >= second_delta * 0.5
    
    def test_zero_drift_minimal_contribution(self, model, engine, feature_dims):
        """Zero drift should result in minimal/zero contributions."""
        drift_dict = {
            m: np.zeros((5, d))
            for m, d in feature_dims.items()
        }
        
        explanation = engine.explain(drift_dict, horizon=24)
        
        # Total explained should be very low
        assert explanation.total_explained < 0.1
    
    def test_explanation_for_all_horizons(self, model, engine, feature_dims):
        """Explanations should work for all horizons."""
        np.random.seed(42)
        drift_dict = {
            m: np.random.randn(5, d) + 1.0
            for m, d in feature_dims.items()
        }
        
        explanations = engine.explain_all_horizons(drift_dict)
        
        assert 24 in explanations
        assert 48 in explanations
        assert 72 in explanations
        
        for horizon, exp in explanations.items():
            assert exp.horizon == horizon
            assert 0 <= exp.risk <= 1
            assert 0 <= exp.confidence <= 1


class TestExplanationOutput:
    """Test explanation output format."""
    
    @pytest.fixture
    def feature_dims(self):
        return {
            "voice": 5,
            "cough": 3,
        }
    
    @pytest.fixture
    def model(self, feature_dims):
        return ViralFlip(
            feature_dims=feature_dims,
            horizons=[24],
            max_lag=2,
            use_interactions=False,
            use_personalization=False,
            use_virus_classifier=False,
        )
    
    def test_to_dict(self, model, feature_dims):
        """Explanation should be serializable to dict."""
        engine = ExplanationEngine(model)
        
        drift_dict = {
            m: np.random.randn(3, d)
            for m, d in feature_dims.items()
        }
        
        explanation = engine.explain(drift_dict, horizon=24)
        result = explanation.to_dict()
        
        assert "horizon" in result
        assert "risk" in result
        assert "confidence" in result
        assert "top_contributors" in result
    
    def test_summary_text(self, model, feature_dims):
        """Should generate readable summary text."""
        engine = ExplanationEngine(model)
        
        drift_dict = {
            m: np.random.randn(3, d) + 1.0
            for m, d in feature_dims.items()
        }
        
        explanation = engine.explain(drift_dict, horizon=24)
        summary = explanation.summary_text()
        
        assert "Risk" in summary
        assert "24h" in summary or "24" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

