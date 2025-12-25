"""Tests for Lag Lattice Hazard Model tensor shapes."""

import numpy as np
import pytest
import torch

from viralflip.model.lag_lattice import LagLatticeHazardModel


class TestLagLatticeShapes:
    """Test tensor shapes in lag lattice model."""
    
    @pytest.fixture
    def model(self):
        return LagLatticeHazardModel(
            n_modalities=7,
            horizons=[24, 48, 72],
            max_lag=12,
            l1_lambda=0.01,
            use_missing_indicators=True,
        )
    
    def test_forward_shape(self, model):
        """Output shape should be (batch, seq, n_horizons)."""
        batch_size = 8
        seq_len = 20
        n_mod = 7
        
        drift_scores = torch.randn(batch_size, seq_len, n_mod)
        
        output = model(drift_scores)
        
        assert output.shape == (batch_size, seq_len, 3)  # 3 horizons
    
    def test_forward_with_missing_indicators(self, model):
        """Missing indicators should have same shape as drift scores."""
        batch_size = 4
        seq_len = 15
        n_mod = 7
        
        drift_scores = torch.randn(batch_size, seq_len, n_mod)
        missing_indicators = torch.zeros(batch_size, seq_len, n_mod)
        missing_indicators[:, :, 0] = 1  # First modality missing
        
        output = model(drift_scores, missing_indicators)
        
        assert output.shape == (batch_size, seq_len, 3)
    
    def test_weights_shape(self, model):
        """Weight tensor should be (n_horizons, n_modalities, max_lag+1)."""
        weights = model.get_weights()
        
        assert weights.shape == (3, 7, 13)  # 3 horizons, 7 mods, 13 lags (0-12)
    
    def test_output_is_probability(self, model):
        """Output should be in [0, 1] range."""
        drift_scores = torch.randn(4, 10, 7)
        
        output = model(drift_scores)
        
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_higher_drift_increases_risk(self, model):
        """Higher drift scores should generally increase risk."""
        # Low drift
        low_drift = torch.zeros(1, 5, 7)
        # High drift
        high_drift = torch.ones(1, 5, 7) * 3.0
        
        # Train briefly to make weights nonzero
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(10):
            output = model(high_drift)
            loss = -output.mean()  # Encourage higher output for high drift
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            risk_low = model(low_drift)
            risk_high = model(high_drift)
        
        # High drift should produce higher risk (on average)
        assert risk_high.mean() >= risk_low.mean()
    
    def test_l1_penalty_nonzero_after_training(self, model):
        """L1 penalty should be non-zero after weights become nonzero."""
        # Initially, weights are near zero so penalty is low
        initial_penalty = model.l1_penalty()
        
        # Train to make weights nonzero
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        drift = torch.ones(4, 10, 7)
        
        for _ in range(20):
            output = model(drift)
            loss = -output.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_penalty = model.l1_penalty()
        
        # Penalty should increase as weights grow
        assert final_penalty > initial_penalty
    
    def test_single_sample(self, model):
        """Should handle single sample (batch=1)."""
        drift_scores = torch.randn(1, 10, 7)
        
        output = model(drift_scores)
        
        assert output.shape == (1, 10, 3)
    
    def test_single_timestep(self, model):
        """Should handle single timestep (seq=1)."""
        drift_scores = torch.randn(4, 1, 7)
        
        output = model(drift_scores)
        
        assert output.shape == (4, 1, 3)


class TestLagLatticeContributors:
    """Test top contributor extraction."""
    
    @pytest.fixture
    def model(self):
        model = LagLatticeHazardModel(
            n_modalities=5,
            horizons=[24, 48, 72],
            max_lag=4,
        )
        
        # Set some weights manually
        with torch.no_grad():
            model._weights_raw.data[0, 0, 0] = 2.0  # Horizon 0, mod 0, lag 0
            model._weights_raw.data[0, 1, 1] = 1.5  # Horizon 0, mod 1, lag 1
        
        return model
    
    def test_get_top_contributors(self, model):
        """Should return top contributing (modality, lag) pairs."""
        drift_scores = torch.zeros(1, 5, 5)
        drift_scores[0, -1, 0] = 3.0  # High drift for mod 0
        drift_scores[0, -1, 1] = 2.0  # Medium drift for mod 1
        
        contributors = model.get_top_contributors(drift_scores, horizon_idx=0, top_k=3)
        
        assert len(contributors) <= 3
        # First contributor should be (mod 0, lag 0) due to high weight and drift
        assert contributors[0][0] == 0  # modality index
        assert contributors[0][1] == 0  # lag


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

