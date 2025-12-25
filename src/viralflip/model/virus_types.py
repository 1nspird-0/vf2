"""Virus type definitions for multi-class classification.

ViralFlip can classify detected illnesses into specific viral categories.
"""

from enum import IntEnum
from typing import Dict, List


class VirusType(IntEnum):
    """Enumeration of virus types that ViralFlip can detect.
    
    Each virus has characteristic physiological signatures:
    - COVID: Elevated HR, voice changes, loss of smell indicators, specific cough patterns
    - FLU: High fever pattern, muscle fatigue (gait changes), rapid onset
    - COLD: Mild symptoms, gradual onset, nasal voice changes
    - RSV: Respiratory distress, wheezing patterns, distinct cough
    - PNEUMONIA: Severe respiratory changes, significant HR elevation
    - GENERAL: Catch-all for unclassified respiratory illness
    """
    
    HEALTHY = 0      # No illness detected
    COVID = 1        # COVID-19 / SARS-CoV-2
    FLU = 2          # Influenza A/B
    COLD = 3         # Common cold (rhinovirus, etc.)
    RSV = 4          # Respiratory Syncytial Virus
    PNEUMONIA = 5    # Bacterial/viral pneumonia
    GENERAL = 6      # General respiratory illness (unclassified)
    

# Number of virus classes (including HEALTHY)
NUM_VIRUS_CLASSES = len(VirusType)

# Number of illness types (excluding HEALTHY)
NUM_ILLNESS_TYPES = NUM_VIRUS_CLASSES - 1


# Human-readable names
VIRUS_NAMES: Dict[VirusType, str] = {
    VirusType.HEALTHY: "Healthy",
    VirusType.COVID: "COVID-19",
    VirusType.FLU: "Influenza",
    VirusType.COLD: "Common Cold",
    VirusType.RSV: "RSV",
    VirusType.PNEUMONIA: "Pneumonia",
    VirusType.GENERAL: "General Respiratory",
}


# Short labels for display
VIRUS_SHORT_NAMES: Dict[VirusType, str] = {
    VirusType.HEALTHY: "HEALTHY",
    VirusType.COVID: "COVID",
    VirusType.FLU: "FLU",
    VirusType.COLD: "COLD",
    VirusType.RSV: "RSV",
    VirusType.PNEUMONIA: "PNEUMO",
    VirusType.GENERAL: "RESP",
}


# Characteristic drift patterns per virus (relative weights)
# These are used in synthetic data generation
VIRUS_DRIFT_PROFILES: Dict[VirusType, Dict[str, float]] = {
    VirusType.COVID: {
        "voice": 2.5,      # Strong voice changes
        "cough": 2.0,      # Dry persistent cough
        "rppg": 2.2,       # HR elevation
        "gait_active": 1.5,  # Fatigue
        "tap": 1.8,        # Motor impairment
        "light": 1.2,      # Sleep disruption
        "baro": 0.8,       # Mild
    },
    VirusType.FLU: {
        "voice": 1.5,      # Moderate voice change
        "cough": 1.8,      # Productive cough
        "rppg": 2.8,       # High fever = high HR
        "gait_active": 2.5,  # Significant fatigue
        "tap": 2.0,        # Muscle aches affect motor
        "light": 2.0,      # High sleep disruption
        "baro": 1.0,
    },
    VirusType.COLD: {
        "voice": 2.0,      # Nasal voice
        "cough": 1.2,      # Mild cough
        "rppg": 0.8,       # Minimal HR change
        "gait_active": 0.6,  # Low fatigue
        "tap": 0.5,        # Minimal motor impact
        "light": 0.8,      # Mild sleep disruption
        "baro": 0.5,
    },
    VirusType.RSV: {
        "voice": 1.2,      # Some voice change
        "cough": 2.5,      # Wheezing cough
        "rppg": 2.0,       # Respiratory distress
        "gait_active": 1.8,  # Breathing affects activity
        "tap": 1.0,
        "light": 1.5,
        "baro": 1.8,       # Respiratory pressure changes
    },
    VirusType.PNEUMONIA: {
        "voice": 1.8,      # Weak voice
        "cough": 3.0,      # Severe cough
        "rppg": 3.0,       # High HR, fever
        "gait_active": 3.0,  # Severe fatigue
        "tap": 2.5,        # Weakness
        "light": 2.5,      # Poor sleep
        "baro": 2.5,       # Respiratory changes
    },
    VirusType.GENERAL: {
        "voice": 1.5,
        "cough": 1.5,
        "rppg": 1.5,
        "gait_active": 1.5,
        "tap": 1.0,
        "light": 1.2,
        "baro": 1.0,
    },
}


# Illness duration distribution (mean days, std days)
VIRUS_DURATION_DAYS: Dict[VirusType, tuple] = {
    VirusType.COVID: (10, 4),      # ~7-14 days
    VirusType.FLU: (7, 2),         # ~5-9 days
    VirusType.COLD: (8, 3),        # ~5-11 days
    VirusType.RSV: (8, 3),         # ~5-11 days
    VirusType.PNEUMONIA: (14, 5),  # ~9-19 days
    VirusType.GENERAL: (7, 3),     # ~4-10 days
}


# Onset speed (hours until peak symptoms)
VIRUS_ONSET_HOURS: Dict[VirusType, tuple] = {
    VirusType.COVID: (48, 24),     # Gradual onset
    VirusType.FLU: (12, 6),        # Rapid onset
    VirusType.COLD: (36, 18),      # Gradual onset
    VirusType.RSV: (24, 12),       # Moderate onset
    VirusType.PNEUMONIA: (48, 24), # Gradual onset
    VirusType.GENERAL: (36, 18),   # Variable
}


# Relative prevalence (for synthetic data sampling)
VIRUS_PREVALENCE: Dict[VirusType, float] = {
    VirusType.COVID: 0.15,
    VirusType.FLU: 0.25,
    VirusType.COLD: 0.30,
    VirusType.RSV: 0.10,
    VirusType.PNEUMONIA: 0.05,
    VirusType.GENERAL: 0.15,
}


def get_virus_name(virus_type: int) -> str:
    """Get human-readable name for virus type."""
    return VIRUS_NAMES.get(VirusType(virus_type), "Unknown")


def get_virus_from_name(name: str) -> VirusType:
    """Get virus type from name (case-insensitive)."""
    name_lower = name.lower()
    for vt, vname in VIRUS_NAMES.items():
        if vname.lower() == name_lower or vt.name.lower() == name_lower:
            return vt
    return VirusType.GENERAL


def list_virus_types() -> List[str]:
    """List all virus type names."""
    return [VIRUS_NAMES[vt] for vt in VirusType if vt != VirusType.HEALTHY]

