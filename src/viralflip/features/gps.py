"""GPS mobility feature extraction (privacy-preserving).

Features extracted:
- Radius of gyration
- Location entropy (Shannon entropy over geohash bins)
- Unique places count
- Time at home fraction
- Routine similarity (cosine similarity vs baseline)

Quality metrics:
- GPS uptime fraction

PRIVACY NOTE: Only derived features are stored. Raw lat/lon traces
are NOT persisted.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GPSFeatures:
    """Container for GPS mobility features."""
    
    radius_of_gyration: float
    location_entropy: float
    unique_places_count: float
    time_at_home_fraction: float
    routine_similarity: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.radius_of_gyration,
            self.location_entropy,
            self.unique_places_count,
            self.time_at_home_fraction,
            self.routine_similarity,
        ], dtype=np.float32)


@dataclass
class GPSQuality:
    """Container for GPS quality metrics."""
    
    gps_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.gps_uptime_fraction], dtype=np.float32)


class GPSFeatureExtractor:
    """Extract privacy-preserving GPS mobility features."""
    
    def __init__(
        self,
        geohash_precision: int = 6,
        home_inference_hours: tuple[int, int] = (0, 6),
        earth_radius_km: float = 6371.0,
    ):
        """Initialize GPS feature extractor.
        
        Args:
            geohash_precision: Geohash precision for location binning.
            home_inference_hours: Hours used to infer home location (night).
            earth_radius_km: Earth radius for distance calculations.
        """
        self.geohash_precision = geohash_precision
        self.home_inference_hours = home_inference_hours
        self.earth_radius_km = earth_radius_km
        self._home_geohash: Optional[str] = None
        self._baseline_histogram: Optional[np.ndarray] = None
    
    def set_home(self, home_geohash: str) -> None:
        """Set home location geohash.
        
        Args:
            home_geohash: Geohash of home location.
        """
        self._home_geohash = home_geohash
    
    def set_baseline_histogram(self, histogram: np.ndarray) -> None:
        """Set baseline hourly location histogram for routine comparison.
        
        Args:
            histogram: Normalized histogram of location patterns.
        """
        self._baseline_histogram = histogram
    
    def extract(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        timestamps: np.ndarray,
        bin_duration_minutes: float = 360.0,
    ) -> tuple[GPSFeatures, GPSQuality]:
        """Extract GPS features from location pings.
        
        Args:
            latitudes: Array of latitude values.
            longitudes: Array of longitude values.
            timestamps: Array of timestamps (minutes from bin start).
            bin_duration_minutes: Total bin duration in minutes.
            
        Returns:
            Tuple of (GPSFeatures, GPSQuality).
        """
        if len(latitudes) == 0:
            return self._empty_features(), GPSQuality(gps_uptime_fraction=0.0)
        
        # Compute uptime (based on ping frequency)
        expected_pings = bin_duration_minutes / 5  # Assume ~5 min ping interval
        uptime = min(len(latitudes) / expected_pings, 1.0)
        
        # Geohash each location
        geohashes = [
            self._encode_geohash(lat, lon) 
            for lat, lon in zip(latitudes, longitudes)
        ]
        
        # Radius of gyration
        radius = self._compute_radius_of_gyration(latitudes, longitudes)
        
        # Location entropy
        entropy = self._compute_location_entropy(geohashes)
        
        # Unique places
        unique_places = len(set(geohashes))
        
        # Time at home
        if self._home_geohash is not None:
            time_at_home = np.mean([gh == self._home_geohash for gh in geohashes])
        else:
            # Infer home as most frequent location
            time_at_home = self._infer_time_at_home(geohashes)
        
        # Routine similarity
        routine_sim = self._compute_routine_similarity(geohashes, timestamps)
        
        features = GPSFeatures(
            radius_of_gyration=float(radius),
            location_entropy=float(entropy),
            unique_places_count=float(unique_places),
            time_at_home_fraction=float(time_at_home),
            routine_similarity=float(routine_sim),
        )
        
        quality = GPSQuality(gps_uptime_fraction=float(uptime))
        
        return features, quality
    
    def extract_from_summary(
        self,
        radius_of_gyration: float,
        location_entropy: float,
        unique_places: int,
        time_at_home: float,
        routine_similarity: float,
        uptime_fraction: float,
    ) -> tuple[GPSFeatures, GPSQuality]:
        """Create features from pre-computed summary statistics.
        
        Args:
            radius_of_gyration: Radius of gyration in km.
            location_entropy: Shannon entropy of locations.
            unique_places: Number of unique locations.
            time_at_home: Fraction of time at home.
            routine_similarity: Cosine similarity to baseline.
            uptime_fraction: GPS uptime fraction.
            
        Returns:
            Tuple of (GPSFeatures, GPSQuality).
        """
        features = GPSFeatures(
            radius_of_gyration=float(radius_of_gyration),
            location_entropy=float(location_entropy),
            unique_places_count=float(unique_places),
            time_at_home_fraction=float(time_at_home),
            routine_similarity=float(routine_similarity),
        )
        
        quality = GPSQuality(gps_uptime_fraction=float(uptime_fraction))
        
        return features, quality
    
    def _encode_geohash(self, lat: float, lon: float) -> str:
        """Encode lat/lon to geohash string.
        
        Simplified implementation for this use case.
        """
        # Base32 alphabet
        alphabet = '0123456789bcdefghjkmnpqrstuvwxyz'
        
        lat_range = (-90.0, 90.0)
        lon_range = (-180.0, 180.0)
        
        geohash = []
        is_lon = True
        bit = 0
        ch = 0
        
        while len(geohash) < self.geohash_precision:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if lon >= mid:
                    ch |= (1 << (4 - bit))
                    lon_range = (mid, lon_range[1])
                else:
                    lon_range = (lon_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if lat >= mid:
                    ch |= (1 << (4 - bit))
                    lat_range = (mid, lat_range[1])
                else:
                    lat_range = (lat_range[0], mid)
            
            is_lon = not is_lon
            bit += 1
            
            if bit == 5:
                geohash.append(alphabet[ch])
                bit = 0
                ch = 0
        
        return ''.join(geohash)
    
    def _compute_radius_of_gyration(
        self, 
        latitudes: np.ndarray, 
        longitudes: np.ndarray,
    ) -> float:
        """Compute radius of gyration in kilometers."""
        if len(latitudes) < 2:
            return 0.0
        
        # Center of mass
        lat_center = np.mean(latitudes)
        lon_center = np.mean(longitudes)
        
        # Compute distances to center
        distances = [
            self._haversine_distance(lat, lon, lat_center, lon_center)
            for lat, lon in zip(latitudes, longitudes)
        ]
        
        # Radius of gyration
        rg = np.sqrt(np.mean(np.array(distances) ** 2))
        
        return float(rg)
    
    def _haversine_distance(
        self, 
        lat1: float, lon1: float, 
        lat2: float, lon2: float,
    ) -> float:
        """Compute Haversine distance between two points in km."""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return self.earth_radius_km * c
    
    def _compute_location_entropy(self, geohashes: list[str]) -> float:
        """Compute Shannon entropy of location distribution."""
        if len(geohashes) == 0:
            return 0.0
        
        # Count occurrences
        counts: dict[str, int] = {}
        for gh in geohashes:
            counts[gh] = counts.get(gh, 0) + 1
        
        # Compute probabilities
        total = len(geohashes)
        probs = np.array([count / total for count in counts.values()])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)
    
    def _infer_time_at_home(self, geohashes: list[str]) -> float:
        """Infer time at home as time at most frequent location."""
        if len(geohashes) == 0:
            return 0.0
        
        counts: dict[str, int] = {}
        for gh in geohashes:
            counts[gh] = counts.get(gh, 0) + 1
        
        max_count = max(counts.values())
        return float(max_count / len(geohashes))
    
    def _compute_routine_similarity(
        self, 
        geohashes: list[str], 
        timestamps: np.ndarray,
    ) -> float:
        """Compute cosine similarity to baseline routine.
        
        If no baseline, returns 1.0 (perfect similarity with self).
        """
        if self._baseline_histogram is None:
            return 1.0
        
        if len(geohashes) == 0:
            return 0.0
        
        # Build current hourly location histogram
        # Simplified: use unique geohash per hour
        n_hours = 6  # For 6-hour bin
        current_hist = np.zeros(n_hours * 10)  # 10 possible locations per hour
        
        for gh, ts in zip(geohashes, timestamps):
            hour_idx = int(ts / 60) % n_hours
            # Hash geohash to location bin
            loc_bin = hash(gh) % 10
            current_hist[hour_idx * 10 + loc_bin] += 1
        
        # Normalize
        current_hist = current_hist / (np.linalg.norm(current_hist) + 1e-10)
        
        # Cosine similarity
        if len(self._baseline_histogram) != len(current_hist):
            return 1.0
        
        similarity = np.dot(current_hist, self._baseline_histogram)
        
        return float(np.clip(similarity, 0, 1))
    
    def _empty_features(self) -> GPSFeatures:
        """Return empty features."""
        return GPSFeatures(
            radius_of_gyration=0.0,
            location_entropy=0.0,
            unique_places_count=0.0,
            time_at_home_fraction=0.0,
            routine_similarity=1.0,  # No change from baseline
        )

