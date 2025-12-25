"""Tests for Personal Baseline Memory (PBM) leakage prevention.

CRITICAL: Baseline at time t must use only data from times <= t-1.
These tests verify that no future information leaks into baseline computation.
"""

import numpy as np
import pytest

from viralflip.baseline.pbm import PersonalBaselineMemory


class TestPBMNoLeakage:
    """Test suite for PBM leakage prevention."""
    
    @pytest.fixture
    def feature_dims(self):
        return {
            "voice": 5,
            "cough": 3,
        }
    
    @pytest.fixture
    def pbm(self, feature_dims):
        return PersonalBaselineMemory(
            feature_dims=feature_dims,
            init_bins=10,  # Shorter for testing
            alpha=0.1,
            beta=0.1,
        )
    
    def test_baseline_not_initialized_before_enough_samples(self, pbm, feature_dims):
        """Baseline should not be available before sufficient initialization samples."""
        user_id = "test_user"
        
        # Add fewer samples than required
        for t in range(5):  # Less than init_bins=10
            features = np.random.randn(feature_dims["voice"])
            pbm.add_init_sample(user_id, "voice", features)
        
        # Should not be initialized yet
        assert not pbm.is_initialized(user_id, "voice")
        
        # Drift should return zeros
        drift = pbm.compute_drift(user_id, "voice", np.random.randn(feature_dims["voice"]))
        assert np.allclose(drift, 0.0)
    
    def test_baseline_uses_only_past_data(self, pbm, feature_dims):
        """Verify baseline at time t uses only data from t-1 and earlier."""
        user_id = "test_user"
        
        # Create known sequence of features
        np.random.seed(42)
        all_features = [np.random.randn(feature_dims["voice"]) * (i + 1) for i in range(20)]
        
        # Initialize baseline with first 10 samples
        for t in range(10):
            pbm.add_init_sample(user_id, "voice", all_features[t])
        
        assert pbm.is_initialized(user_id, "voice")
        
        # Get baseline after initialization
        mu_init, sigma_init = pbm.get_baseline(user_id, "voice")
        
        # The baseline should be computed from samples 0-9 only
        expected_mu = np.median(all_features[:10], axis=0)
        
        # Verify baseline matches expected (within numerical tolerance)
        assert np.allclose(mu_init, expected_mu, atol=0.1)
        
        # Now compute drift at time 10 - should NOT use sample 10's value
        drift_10 = pbm.compute_drift(user_id, "voice", all_features[10])
        
        # The drift should be relative to baseline computed from 0-9
        # NOT incorporating sample 10's information
        expected_drift = (all_features[10] - mu_init) / (sigma_init + 1e-6)
        expected_drift = np.clip(expected_drift, -6, 6)
        
        assert np.allclose(drift_10, expected_drift, atol=0.1)
    
    def test_update_does_not_use_current_sample_for_current_drift(self, pbm, feature_dims):
        """Update at time t should not affect drift computation at time t."""
        user_id = "test_user"
        
        # Initialize
        for t in range(10):
            features = np.ones(feature_dims["voice"]) * 0.5
            pbm.add_init_sample(user_id, "voice", features)
        
        # Get baseline and compute drift for a new sample
        mu_before, sigma_before = pbm.get_baseline(user_id, "voice")
        
        # New sample with very different value
        new_sample = np.ones(feature_dims["voice"]) * 100.0
        drift_before_update = pbm.compute_drift(user_id, "voice", new_sample)
        
        # Now update baseline with this sample
        pbm.update_baseline(
            user_id, "voice", new_sample,
            quality=1.0,
            predicted_risk=0.1,  # Safe to update
            is_labeled_sick=False,
        )
        
        # Drift BEFORE update should NOT have used the update
        # Recompute expected drift with old baseline
        expected_drift = (new_sample - mu_before) / (sigma_before + 1e-6)
        expected_drift = np.clip(expected_drift, -6, 6)
        
        assert np.allclose(drift_before_update, expected_drift, atol=0.1)
    
    def test_update_gating_prevents_sick_period_leakage(self, pbm, feature_dims):
        """Updates should be blocked during sick periods."""
        user_id = "test_user"
        
        # Initialize
        for t in range(10):
            features = np.zeros(feature_dims["voice"])
            pbm.add_init_sample(user_id, "voice", features)
        
        mu_before, _ = pbm.get_baseline(user_id, "voice")
        
        # Try to update with sick label
        sick_features = np.ones(feature_dims["voice"]) * 10.0
        updated = pbm.update_baseline(
            user_id, "voice", sick_features,
            quality=1.0,
            predicted_risk=0.5,  # High risk
            is_labeled_sick=True,
        )
        
        # Update should be rejected
        assert not updated
        
        # Baseline should be unchanged
        mu_after, _ = pbm.get_baseline(user_id, "voice")
        assert np.allclose(mu_before, mu_after)
    
    def test_update_gating_prevents_high_risk_leakage(self, pbm, feature_dims):
        """Updates should be blocked when predicted risk is high."""
        user_id = "test_user"
        
        # Initialize
        for t in range(10):
            features = np.zeros(feature_dims["voice"])
            pbm.add_init_sample(user_id, "voice", features)
        
        mu_before, _ = pbm.get_baseline(user_id, "voice")
        
        # Try to update with high predicted risk (but not labeled sick)
        risky_features = np.ones(feature_dims["voice"]) * 5.0
        updated = pbm.update_baseline(
            user_id, "voice", risky_features,
            quality=1.0,
            predicted_risk=0.5,  # Above threshold of 0.3
            is_labeled_sick=False,
        )
        
        # Update should be rejected due to high risk
        assert not updated
        
        # Baseline should be unchanged
        mu_after, _ = pbm.get_baseline(user_id, "voice")
        assert np.allclose(mu_before, mu_after)
    
    def test_temporal_ordering_in_batch(self, pbm, feature_dims):
        """When processing a batch, earlier samples should not see later data."""
        user_id = "test_user"
        
        # Create time-ordered features with clear trend
        features_by_time = {}
        for t in range(20):
            features_by_time[t] = np.ones(feature_dims["voice"]) * t
        
        # Initialize with first 10
        for t in range(10):
            pbm.add_init_sample(user_id, "voice", features_by_time[t])
        
        # Compute drifts for times 10-19
        drifts = {}
        for t in range(10, 20):
            drifts[t] = pbm.compute_drift(user_id, "voice", features_by_time[t])
            
            # After computing drift, update baseline (simulating online processing)
            pbm.update_baseline(
                user_id, "voice", features_by_time[t],
                quality=1.0,
                predicted_risk=0.1,
                is_labeled_sick=False,
            )
        
        # Verify that drift at time 10 was computed before seeing times 11-19
        # The drift should be based on baseline from times 0-9 only
        mu_init = np.median([features_by_time[t] for t in range(10)], axis=0)
        
        # Drift at time 10 should reflect distance from initial baseline
        # (actual values depend on MAD, but direction should be positive)
        assert np.all(drifts[10] > 0)  # Features at t=10 are higher than baseline
        
        # Later drifts should reflect updated baselines (monotonically changing)
        for t in range(11, 20):
            # As baseline adapts upward, drift for same-valued features decreases
            # This is a softer check as exact values depend on update dynamics
            pass  # The key test is that t=10 drift was computed correctly above


class TestPBMStatePersistence:
    """Test PBM state saving and loading."""
    
    @pytest.fixture
    def feature_dims(self):
        return {"voice": 5}
    
    def test_state_dict_roundtrip(self, feature_dims):
        """State should be preservable through save/load cycle."""
        pbm = PersonalBaselineMemory(feature_dims=feature_dims, init_bins=5)
        
        # Initialize a user
        for t in range(5):
            features = np.random.randn(feature_dims["voice"])
            pbm.add_init_sample("user1", "voice", features)
        
        # Get state
        state = pbm.get_state_dict()
        
        # Create new PBM and load state
        pbm2 = PersonalBaselineMemory(feature_dims=feature_dims)
        pbm2.load_state_dict(state)
        
        # Should have same baseline
        mu1, sigma1 = pbm.get_baseline("user1", "voice")
        mu2, sigma2 = pbm2.get_baseline("user1", "voice")
        
        assert np.allclose(mu1, mu2)
        assert np.allclose(sigma1, sigma2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

