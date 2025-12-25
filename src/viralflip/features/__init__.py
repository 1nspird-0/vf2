"""Feature extraction modules for phone sensor data."""

from viralflip.features.voice import VoiceFeatureExtractor
from viralflip.features.cough import CoughFeatureExtractor
from viralflip.features.tapping import TappingFeatureExtractor
from viralflip.features.gait_active import GaitActiveFeatureExtractor
from viralflip.features.imu_passive import IMUPassiveFeatureExtractor
from viralflip.features.rppg import RPPGFeatureExtractor
from viralflip.features.gps import GPSFeatureExtractor
from viralflip.features.light import LightFeatureExtractor
from viralflip.features.baro import BaroFeatureExtractor
from viralflip.features.screen import ScreenFeatureExtractor
from viralflip.features.quality import QualityAssessor

__all__ = [
    "VoiceFeatureExtractor",
    "CoughFeatureExtractor",
    "TappingFeatureExtractor",
    "GaitActiveFeatureExtractor",
    "IMUPassiveFeatureExtractor",
    "RPPGFeatureExtractor",
    "GPSFeatureExtractor",
    "LightFeatureExtractor",
    "BaroFeatureExtractor",
    "ScreenFeatureExtractor",
    "QualityAssessor",
]

# Feature dimensions for each modality
FEATURE_DIMS = {
    "voice": 24,      # f0 (4) + jitter/shimmer/HNR (3) + spectral (4) + mfcc (12) + rate (1)
    "cough": 6,       # count, night, day, burstiness, max_hourly, conf_mean
    "tap": 5,         # count, iti_mean, iti_std, iti_cv, outlier_rate, fatigue_slope
    "gait_active": 7, # cadence, interval stats (3), jerk (2), sway, regularity
    "imu_passive": 6, # activity levels (3), fragmentation, restlessness, tremor
    "rppg": 5,        # hr_mean, ibi_std, hrv_rmssd, pulse_amp_mean, pulse_amp_cv
    "gps": 5,         # radius, entropy, unique_places, time_home, routine_sim
    "light": 4,       # lux_mean, lux_std, lux_evening, circadian_score
    "baro": 4,        # pressure_mean, pressure_std, pressure_slope, jump_count
    "screen": 5,      # on_count, on_minutes, first_unlock, last_lock, longest_off
}

# Quality dimension for each modality (typically 1-3)
QUALITY_DIMS = {
    "voice": 3,       # snr_est, voiced_fraction, clipping_rate
    "cough": 1,       # mic_uptime_fraction
    "tap": 1,         # completion_flag
    "gait_active": 2, # steps_detected_ok, placement_flag
    "imu_passive": 1, # imu_uptime_fraction
    "rppg": 2,        # sqi, motion_artifact_score
    "gps": 1,         # gps_uptime_fraction
    "light": 1,       # light_uptime_fraction
    "baro": 1,        # baro_uptime_fraction
    "screen": 1,      # screen_event_uptime_fraction
}

