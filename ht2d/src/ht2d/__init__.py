"""ht2d — Hypothesis Test Drift Detection.

Compact package version of the river.ipynb example.
"""

from .distributions import Distribution, GaussianMixture, DriftingMixtureStream
from .detectors import (
    DriftDetector,
    NoDriftDetector,
    ThresholdDriftDetector,
    ZTestDriftDetector,
)
from .models import (
    to_dict,
    to_array,
    window_accuracy,
    BaseModelAdapter,
    HoeffdingTreeClassifier,
)

__all__ = [
    "Distribution",
    "GaussianMixture",
    "DriftingMixtureStream",
    "DriftDetector",
    "NoDriftDetector",
    "ThresholdDriftDetector",
    "ZTestDriftDetector",
    "to_dict",
    "to_array",
    "window_accuracy",
    "BaseModelAdapter",
    "HoeffdingTreeClassifier",
]
