"""Rule-based classifier using micro-Doppler CV + kinematic features."""

from dataclasses import dataclass
from feature_extractor import Features


@dataclass
class Classification:
    label: str
    confidence: float
    reason: str


# Blade CV thresholds (coefficient of variation of slow-time amplitude)
_CV_DRONE_MIN = 0.25     # drones: strong AM modulation from rotors
_CV_BIRD_MIN = 0.04      # birds: weak wing-beat AM
_CV_RIGID_MAX = 0.03     # cars/persons: rigid body, no modulation

_SPEED_PERSON_MAX = 3.5  # m/s
_SPEED_VEHICLE_MIN = 8.0 # m/s
_POWER_CAR_MIN = 40.0    # dB (proxy — car RCS >> drone)


def classify(f: Features) -> Classification:
    # Drone: high blade CV (rotor AM modulation)
    if f.blade_cv >= _CV_DRONE_MIN:
        conf = min(1.0, 0.6 + (f.blade_cv - _CV_DRONE_MIN) / 0.4)
        return Classification("drone", conf, f"blade CV={f.blade_cv:.3f} ≥ {_CV_DRONE_MIN}")

    # Car / motorcycle: fast + high peak power + no blade CV
    if f.velocity_mps >= _SPEED_VEHICLE_MIN and f.peak_power_db >= _POWER_CAR_MIN:
        label = "car" if f.peak_power_db >= 55.0 else "motorcycle"
        conf = min(1.0, 0.65 + (f.peak_power_db - _POWER_CAR_MIN) / 40.0)
        return Classification(label, conf, f"vehicle speed + high RCS proxy")

    # Person: slow + low blade CV
    if f.velocity_mps <= _SPEED_PERSON_MAX and f.blade_cv < _CV_RIGID_MAX:
        return Classification("person", 0.80, f"slow speed + rigid body")

    # Bird: some wing-beat AM, moderate speed
    if _CV_BIRD_MIN <= f.blade_cv < _CV_DRONE_MIN:
        return Classification("bird", 0.70, f"blade CV={f.blade_cv:.3f} matches wing-beat")

    return Classification("unknown", 0.40, "ambiguous")
