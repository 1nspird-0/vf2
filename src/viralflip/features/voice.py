"""Voice feature extraction from audio recordings.

Features extracted:
- Fundamental frequency (F0): mean, std, range, slope
- Voice quality: jitter, shimmer, HNR
- Spectral features: centroid, bandwidth, rolloff, slope
- MFCCs: coefficients 1-6 (mean and std)
- Speaking rate: energy envelope peaks, pause ratio

Quality metrics:
- SNR estimate
- Voiced fraction
- Clipping rate
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from scipy.stats import linregress


@dataclass
class VoiceFeatures:
    """Container for voice features."""
    
    # F0 features
    f0_mean: float
    f0_std: float
    f0_range: float
    f0_slope: float
    
    # Voice quality
    jitter_local: float
    shimmer_local: float
    hnr_mean: float
    
    # Spectral features
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    spectral_slope: float
    
    # MFCC features (mean and std for coeffs 1-6)
    mfcc1_mean: float
    mfcc1_std: float
    mfcc2_mean: float
    mfcc2_std: float
    mfcc3_mean: float
    mfcc3_std: float
    mfcc4_mean: float
    mfcc4_std: float
    mfcc5_mean: float
    mfcc5_std: float
    mfcc6_mean: float
    mfcc6_std: float
    
    # Speaking rate
    speaking_rate_proxy: float
    pause_ratio: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.f0_mean, self.f0_std, self.f0_range, self.f0_slope,
            self.jitter_local, self.shimmer_local, self.hnr_mean,
            self.spectral_centroid_mean, self.spectral_bandwidth_mean,
            self.spectral_rolloff_mean, self.spectral_slope,
            self.mfcc1_mean, self.mfcc1_std,
            self.mfcc2_mean, self.mfcc2_std,
            self.mfcc3_mean, self.mfcc3_std,
            self.mfcc4_mean, self.mfcc4_std,
            self.mfcc5_mean, self.mfcc5_std,
            self.mfcc6_mean, self.mfcc6_std,
            self.speaking_rate_proxy, self.pause_ratio,
        ], dtype=np.float32)


@dataclass
class VoiceQuality:
    """Container for voice quality metrics."""
    
    snr_est: float
    voiced_fraction: float
    clipping_rate: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.snr_est, self.voiced_fraction, self.clipping_rate
        ], dtype=np.float32)


class VoiceFeatureExtractor:
    """Extract voice features from audio recordings."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 6,
        min_duration_sec: float = 5.0,
        f0_min: float = 75.0,
        f0_max: float = 500.0,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
    ):
        """Initialize voice feature extractor.
        
        Args:
            sample_rate: Target sample rate in Hz.
            n_mfcc: Number of MFCC coefficients.
            min_duration_sec: Minimum audio duration in seconds.
            f0_min: Minimum F0 frequency in Hz.
            f0_max: Maximum F0 frequency in Hz.
            frame_length_ms: Frame length in milliseconds.
            hop_length_ms: Hop length in milliseconds.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.min_duration_sec = min_duration_sec
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.n_fft = 2 ** int(np.ceil(np.log2(self.frame_length)))
    
    def extract(
        self, 
        audio: np.ndarray,
        sr: Optional[int] = None,
    ) -> tuple[VoiceFeatures, VoiceQuality]:
        """Extract voice features from audio.
        
        Args:
            audio: Audio waveform (mono).
            sr: Sample rate of audio (will resample if different from target).
            
        Returns:
            Tuple of (VoiceFeatures, VoiceQuality).
        """
        # Resample if needed
        if sr is not None and sr != self.sample_rate:
            audio = self._resample(audio, sr)
        
        # Normalize audio
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Check duration
        duration = len(audio) / self.sample_rate
        if duration < self.min_duration_sec:
            return self._empty_features(), self._empty_quality()
        
        # Compute quality metrics
        quality = self._compute_quality(audio)
        
        # Apply simple spectral gating for noise reduction
        audio = self._spectral_gate(audio)
        
        # Voice activity detection
        vad_mask = self._vad(audio)
        voiced_audio = audio[vad_mask] if np.any(vad_mask) else audio
        
        if len(voiced_audio) < self.frame_length:
            return self._empty_features(), quality
        
        # Extract F0 features
        f0_feats = self._extract_f0_features(voiced_audio)
        
        # Extract voice quality features
        jitter, shimmer, hnr = self._extract_voice_quality(voiced_audio)
        
        # Extract spectral features
        spectral_feats = self._extract_spectral_features(voiced_audio)
        
        # Extract MFCC features
        mfcc_feats = self._extract_mfcc_features(voiced_audio)
        
        # Extract speaking rate features
        speaking_rate, pause_ratio = self._extract_speaking_rate(audio, vad_mask)
        
        features = VoiceFeatures(
            f0_mean=f0_feats[0],
            f0_std=f0_feats[1],
            f0_range=f0_feats[2],
            f0_slope=f0_feats[3],
            jitter_local=jitter,
            shimmer_local=shimmer,
            hnr_mean=hnr,
            spectral_centroid_mean=spectral_feats[0],
            spectral_bandwidth_mean=spectral_feats[1],
            spectral_rolloff_mean=spectral_feats[2],
            spectral_slope=spectral_feats[3],
            mfcc1_mean=mfcc_feats[0],
            mfcc1_std=mfcc_feats[1],
            mfcc2_mean=mfcc_feats[2],
            mfcc2_std=mfcc_feats[3],
            mfcc3_mean=mfcc_feats[4],
            mfcc3_std=mfcc_feats[5],
            mfcc4_mean=mfcc_feats[6],
            mfcc4_std=mfcc_feats[7],
            mfcc5_mean=mfcc_feats[8],
            mfcc5_std=mfcc_feats[9],
            mfcc6_mean=mfcc_feats[10],
            mfcc6_std=mfcc_feats[11],
            speaking_rate_proxy=speaking_rate,
            pause_ratio=pause_ratio,
        )
        
        return features, quality
    
    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if sr == self.sample_rate:
            return audio
        num_samples = int(len(audio) * self.sample_rate / sr)
        return signal.resample(audio, num_samples)
    
    def _spectral_gate(self, audio: np.ndarray, noise_factor: float = 1.5) -> np.ndarray:
        """Simple spectral gating for noise reduction."""
        # Compute STFT
        f, t, Zxx = signal.stft(
            audio, fs=self.sample_rate, 
            nperseg=self.frame_length, 
            noverlap=self.frame_length - self.hop_length
        )
        
        # Estimate noise floor from quietest 10% of frames
        frame_power = np.mean(np.abs(Zxx) ** 2, axis=0)
        noise_frames = frame_power < np.percentile(frame_power, 10)
        
        if np.sum(noise_frames) > 0:
            noise_spectrum = np.mean(np.abs(Zxx[:, noise_frames]), axis=1, keepdims=True)
        else:
            noise_spectrum = np.min(np.abs(Zxx), axis=1, keepdims=True)
        
        # Apply gate
        mask = np.abs(Zxx) > noise_factor * noise_spectrum
        Zxx_gated = Zxx * mask
        
        # Inverse STFT
        _, audio_denoised = signal.istft(
            Zxx_gated, fs=self.sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length
        )
        
        return audio_denoised[:len(audio)]
    
    def _vad(self, audio: np.ndarray, energy_threshold: float = 0.02) -> np.ndarray:
        """Simple energy-based voice activity detection."""
        # Compute frame energies
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        energies = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            energies[i] = np.sqrt(np.mean(frame ** 2))
        
        # Threshold
        threshold = energy_threshold * np.max(energies)
        voiced_frames = energies > threshold
        
        # Convert frame mask to sample mask
        sample_mask = np.zeros(len(audio), dtype=bool)
        for i, is_voiced in enumerate(voiced_frames):
            if is_voiced:
                start = i * self.hop_length
                end = min(start + self.frame_length, len(audio))
                sample_mask[start:end] = True
        
        return sample_mask
    
    def _extract_f0_features(self, audio: np.ndarray) -> tuple[float, float, float, float]:
        """Extract F0 (fundamental frequency) features using autocorrelation."""
        # Compute F0 for each frame
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        f0_values = []
        
        min_lag = int(self.sample_rate / self.f0_max)
        max_lag = int(self.sample_rate / self.f0_min)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            
            # Autocorrelation
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            
            # Find peak in valid range
            if max_lag < len(acf):
                search_range = acf[min_lag:max_lag]
                if len(search_range) > 0 and np.max(search_range) > 0.3 * acf[0]:
                    peak_idx = np.argmax(search_range) + min_lag
                    f0 = self.sample_rate / peak_idx
                    f0_values.append(f0)
        
        if len(f0_values) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        f0_values = np.array(f0_values)
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)
        f0_range = np.max(f0_values) - np.min(f0_values)
        
        # F0 slope (linear regression over time)
        if len(f0_values) > 1:
            x = np.arange(len(f0_values))
            slope, _, _, _, _ = linregress(x, f0_values)
            f0_slope = slope
        else:
            f0_slope = 0.0
        
        return f0_mean, f0_std, f0_range, f0_slope
    
    def _extract_voice_quality(self, audio: np.ndarray) -> tuple[float, float, float]:
        """Extract jitter, shimmer, and HNR."""
        # Simplified extraction using frame-based analysis
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        
        if n_frames < 2:
            return 0.0, 0.0, 0.0
        
        # Compute frame properties
        frame_periods = []
        frame_amplitudes = []
        frame_hnr = []
        
        min_lag = int(self.sample_rate / self.f0_max)
        max_lag = int(self.sample_rate / self.f0_min)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            
            # Period estimation via autocorrelation
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            
            if max_lag < len(acf):
                search_range = acf[min_lag:max_lag]
                if len(search_range) > 0:
                    peak_val = np.max(search_range)
                    if peak_val > 0.3 * acf[0]:
                        peak_idx = np.argmax(search_range) + min_lag
                        frame_periods.append(peak_idx / self.sample_rate)
                        frame_amplitudes.append(np.max(np.abs(frame)))
                        
                        # HNR: ratio of harmonic to noise
                        hnr_val = 10 * np.log10(peak_val / (acf[0] - peak_val + 1e-10) + 1e-10)
                        frame_hnr.append(hnr_val)
        
        if len(frame_periods) < 2:
            return 0.0, 0.0, 0.0
        
        frame_periods = np.array(frame_periods)
        frame_amplitudes = np.array(frame_amplitudes)
        
        # Jitter: period-to-period variation
        period_diffs = np.abs(np.diff(frame_periods))
        jitter_local = np.mean(period_diffs) / (np.mean(frame_periods) + 1e-10)
        
        # Shimmer: amplitude-to-amplitude variation
        amp_diffs = np.abs(np.diff(frame_amplitudes))
        shimmer_local = np.mean(amp_diffs) / (np.mean(frame_amplitudes) + 1e-10)
        
        # HNR mean
        hnr_mean = np.mean(frame_hnr) if len(frame_hnr) > 0 else 0.0
        
        return float(jitter_local), float(shimmer_local), float(hnr_mean)
    
    def _extract_spectral_features(self, audio: np.ndarray) -> tuple[float, float, float, float]:
        """Extract spectral features."""
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            audio, fs=self.sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length,
        )
        
        # Power spectrum
        Sxx = np.abs(Sxx) ** 2 + 1e-10
        
        # Spectral centroid
        centroids = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        spectral_centroid_mean = np.mean(centroids)
        
        # Spectral bandwidth
        bandwidths = np.sqrt(
            np.sum(((f[:, np.newaxis] - centroids) ** 2) * Sxx, axis=0) / np.sum(Sxx, axis=0)
        )
        spectral_bandwidth_mean = np.mean(bandwidths)
        
        # Spectral rolloff (85% of energy)
        cumsum = np.cumsum(Sxx, axis=0)
        total = np.sum(Sxx, axis=0, keepdims=True)
        rolloff_idx = np.argmax(cumsum >= 0.85 * total, axis=0)
        rolloffs = f[rolloff_idx]
        spectral_rolloff_mean = np.mean(rolloffs)
        
        # Spectral slope (linear regression of log power vs frequency)
        log_Sxx = np.log(Sxx + 1e-10)
        slopes = []
        for frame_idx in range(log_Sxx.shape[1]):
            if np.std(log_Sxx[:, frame_idx]) > 0:
                slope, _, _, _, _ = linregress(f, log_Sxx[:, frame_idx])
                slopes.append(slope)
        spectral_slope = np.mean(slopes) if slopes else 0.0
        
        return (
            float(spectral_centroid_mean),
            float(spectral_bandwidth_mean),
            float(spectral_rolloff_mean),
            float(spectral_slope),
        )
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> list[float]:
        """Extract MFCC features (mean and std for each coefficient)."""
        # Compute MFCCs using simplified implementation
        # In production, use librosa.feature.mfcc
        
        # Compute power spectrogram
        f, t, Sxx = signal.spectrogram(
            audio, fs=self.sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length,
        )
        power_spec = np.abs(Sxx) ** 2
        
        # Mel filterbank
        n_mels = 40
        mel_filters = self._mel_filterbank(n_mels, len(f), self.sample_rate)
        
        # Apply filterbank
        mel_spec = np.dot(mel_filters, power_spec)
        log_mel_spec = np.log(mel_spec + 1e-10)
        
        # DCT to get MFCCs
        from scipy.fftpack import dct
        mfccs = dct(log_mel_spec, type=2, axis=0, norm='ortho')[:self.n_mfcc + 1]
        
        # Skip c0 (energy), use c1-c6
        mfccs = mfccs[1:self.n_mfcc + 1]
        
        # Compute mean and std for each coefficient
        result = []
        for i in range(self.n_mfcc):
            result.append(float(np.mean(mfccs[i])))
            result.append(float(np.std(mfccs[i])))
        
        return result
    
    def _mel_filterbank(self, n_mels: int, n_fft: int, sr: int) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Mel points
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Bin indices
        bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_fft - 1)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft))
        for i in range(n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank
    
    def _extract_speaking_rate(
        self, audio: np.ndarray, vad_mask: np.ndarray
    ) -> tuple[float, float]:
        """Extract speaking rate features."""
        # Compute energy envelope
        envelope = np.abs(audio)
        
        # Smooth envelope
        window_size = int(0.02 * self.sample_rate)  # 20ms
        kernel = np.ones(window_size) / window_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
        
        # Find peaks in envelope (syllable-like events)
        peaks, _ = signal.find_peaks(
            envelope_smooth,
            height=0.1 * np.max(envelope_smooth),
            distance=int(0.1 * self.sample_rate)  # min 100ms between peaks
        )
        
        # Speaking rate (peaks per second)
        duration = len(audio) / self.sample_rate
        speaking_rate = len(peaks) / duration if duration > 0 else 0.0
        
        # Pause ratio
        voiced_samples = np.sum(vad_mask)
        pause_ratio = 1.0 - (voiced_samples / len(audio)) if len(audio) > 0 else 0.0
        
        return float(speaking_rate), float(pause_ratio)
    
    def _compute_quality(self, audio: np.ndarray) -> VoiceQuality:
        """Compute quality metrics for audio."""
        # SNR estimation (simple: signal power / noise floor power)
        frame_energies = []
        for i in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[i:i + self.frame_length]
            frame_energies.append(np.mean(frame ** 2))
        
        frame_energies = np.array(frame_energies)
        if len(frame_energies) > 0:
            signal_power = np.percentile(frame_energies, 90)
            noise_power = np.percentile(frame_energies, 10) + 1e-10
            snr_est = 10 * np.log10(signal_power / noise_power)
        else:
            snr_est = 0.0
        
        # Voiced fraction from VAD
        vad_mask = self._vad(audio)
        voiced_fraction = np.mean(vad_mask)
        
        # Clipping rate
        clipping_threshold = 0.99
        clipping_rate = np.mean(np.abs(audio) > clipping_threshold)
        
        return VoiceQuality(
            snr_est=float(snr_est),
            voiced_fraction=float(voiced_fraction),
            clipping_rate=float(clipping_rate),
        )
    
    def _empty_features(self) -> VoiceFeatures:
        """Return empty features (all zeros)."""
        return VoiceFeatures(
            f0_mean=0.0, f0_std=0.0, f0_range=0.0, f0_slope=0.0,
            jitter_local=0.0, shimmer_local=0.0, hnr_mean=0.0,
            spectral_centroid_mean=0.0, spectral_bandwidth_mean=0.0,
            spectral_rolloff_mean=0.0, spectral_slope=0.0,
            mfcc1_mean=0.0, mfcc1_std=0.0,
            mfcc2_mean=0.0, mfcc2_std=0.0,
            mfcc3_mean=0.0, mfcc3_std=0.0,
            mfcc4_mean=0.0, mfcc4_std=0.0,
            mfcc5_mean=0.0, mfcc5_std=0.0,
            mfcc6_mean=0.0, mfcc6_std=0.0,
            speaking_rate_proxy=0.0, pause_ratio=0.0,
        )
    
    def _empty_quality(self) -> VoiceQuality:
        """Return empty quality metrics."""
        return VoiceQuality(snr_est=0.0, voiced_fraction=0.0, clipping_rate=0.0)

