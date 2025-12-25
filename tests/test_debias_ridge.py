"""Tests for Behavior-Drift Debiasing (BDD) module."""

import numpy as np
import pytest

from viralflip.debias.ridge import BehaviorDriftDebiaser


class TestBehaviorDriftDebiaser:
    """Test suite for BDD module."""
    
    @pytest.fixture
    def feature_dims(self):
        return {
            "voice": 5,
            "cough": 3,
            "rppg": 4,
            "gps": 5,
            "imu_passive": 6,
            "screen": 5,
        }
    
    @pytest.fixture
    def debiaser(self, feature_dims):
        return BehaviorDriftDebiaser(
            feature_dims=feature_dims,
            ridge_lambda=1.0,
            behavior_blocks=["gps", "imu_passive", "screen"],
            physiology_blocks=["voice", "cough", "rppg"],
        )
    
    def test_fit_and_transform(self, debiaser, feature_dims):
        """Test fitting and transforming with debiaser."""
        n_samples = 100
        np.random.seed(42)
        
        # Create behavior drifts
        behavior_drifts = {
            "gps": np.random.randn(n_samples, feature_dims["gps"]),
            "imu_passive": np.random.randn(n_samples, feature_dims["imu_passive"]),
            "screen": np.random.randn(n_samples, feature_dims["screen"]),
        }
        
        # Create physiology drifts with correlation to behavior
        physiology_drifts = {
            "voice": np.random.randn(n_samples, feature_dims["voice"]),
            "cough": np.random.randn(n_samples, feature_dims["cough"]),
            "rppg": np.random.randn(n_samples, feature_dims["rppg"]),
        }
        
        # Add correlation: physiology partially depends on behavior
        behavior_concat = np.hstack([
            behavior_drifts["gps"],
            behavior_drifts["imu_passive"],
            behavior_drifts["screen"],
        ])
        
        # Make voice correlated with behavior
        physiology_drifts["voice"] += 0.5 * behavior_concat[:, :feature_dims["voice"]]
        
        # Fit debiaser
        debiaser.fit(behavior_drifts, physiology_drifts)
        
        assert debiaser.is_fitted
        
        # Transform a single sample
        sample_behavior = {
            "gps": behavior_drifts["gps"][0],
            "imu_passive": behavior_drifts["imu_passive"][0],
            "screen": behavior_drifts["screen"][0],
        }
        sample_physiology = {
            "voice": physiology_drifts["voice"][0],
            "cough": physiology_drifts["cough"][0],
            "rppg": physiology_drifts["rppg"][0],
        }
        
        results = debiaser.transform(sample_behavior, sample_physiology)
        
        # Check that debiased values are returned
        assert "voice" in results
        assert "cough" in results
        assert "rppg" in results
        
        # Debiased values should differ from original
        assert not np.allclose(results["voice"].debiased, results["voice"].original)
    
    def test_removes_behavior_correlation(self, debiaser, feature_dims):
        """Debiasing should reduce correlation with behavior."""
        n_samples = 500
        np.random.seed(42)
        
        # Create behavior with structure
        behavior_drifts = {
            "gps": np.random.randn(n_samples, feature_dims["gps"]),
            "imu_passive": np.random.randn(n_samples, feature_dims["imu_passive"]),
            "screen": np.random.randn(n_samples, feature_dims["screen"]),
        }
        
        behavior_concat = np.hstack([
            behavior_drifts["gps"],
            behavior_drifts["imu_passive"],
            behavior_drifts["screen"],
        ])
        
        # Create physiology strongly correlated with behavior
        voice_original = 2.0 * behavior_concat[:, :feature_dims["voice"]] + \
                        0.1 * np.random.randn(n_samples, feature_dims["voice"])
        
        physiology_drifts = {
            "voice": voice_original,
            "cough": np.random.randn(n_samples, feature_dims["cough"]),
            "rppg": np.random.randn(n_samples, feature_dims["rppg"]),
        }
        
        # Fit debiaser
        debiaser.fit(behavior_drifts, physiology_drifts)
        
        # Check correlation before and after for each sample
        corr_before = []
        corr_after = []
        
        for i in range(min(100, n_samples)):
            sample_behavior = {
                "gps": behavior_drifts["gps"][i],
                "imu_passive": behavior_drifts["imu_passive"][i],
                "screen": behavior_drifts["screen"][i],
            }
            sample_physiology = {
                "voice": physiology_drifts["voice"][i],
            }
            
            results = debiaser.transform(sample_behavior, sample_physiology)
            
            # Compute simple correlation (dot product as proxy)
            behavior_vec = np.hstack([
                sample_behavior["gps"],
                sample_behavior["imu_passive"],
                sample_behavior["screen"],
            ])
            
            original_voice = results["voice"].original
            debiased_voice = results["voice"].debiased
            
            corr_before.append(np.abs(np.corrcoef(
                behavior_vec[:len(original_voice)], original_voice
            )[0, 1]))
            corr_after.append(np.abs(np.corrcoef(
                behavior_vec[:len(debiased_voice)], debiased_voice
            )[0, 1]))
        
        # After debiasing, correlation should be reduced on average
        mean_corr_before = np.nanmean(corr_before)
        mean_corr_after = np.nanmean(corr_after)
        
        assert mean_corr_after < mean_corr_before
    
    def test_state_dict_roundtrip(self, debiaser, feature_dims):
        """State should be preservable through save/load cycle."""
        n_samples = 50
        np.random.seed(42)
        
        behavior_drifts = {
            "gps": np.random.randn(n_samples, feature_dims["gps"]),
            "imu_passive": np.random.randn(n_samples, feature_dims["imu_passive"]),
            "screen": np.random.randn(n_samples, feature_dims["screen"]),
        }
        physiology_drifts = {
            "voice": np.random.randn(n_samples, feature_dims["voice"]),
            "cough": np.random.randn(n_samples, feature_dims["cough"]),
            "rppg": np.random.randn(n_samples, feature_dims["rppg"]),
        }
        
        debiaser.fit(behavior_drifts, physiology_drifts)
        
        # Get state
        state = debiaser.get_state_dict()
        
        # Create new debiaser and load state
        debiaser2 = BehaviorDriftDebiaser(feature_dims=feature_dims)
        debiaser2.load_state_dict(state)
        
        # Transform should give same results
        sample_b = {k: v[0] for k, v in behavior_drifts.items()}
        sample_p = {k: v[0] for k, v in physiology_drifts.items()}
        
        result1 = debiaser.debias(sample_b, sample_p)
        result2 = debiaser2.debias(sample_b, sample_p)
        
        for modality in result1:
            assert np.allclose(result1[modality], result2[modality])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

