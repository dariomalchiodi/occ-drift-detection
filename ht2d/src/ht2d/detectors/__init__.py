from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Iterable

import numpy as np
import scipy.stats as st


class DriftDetector(ABC):
    """Base class for drift detectors working on a window of outlier flags."""

    @abstractmethod
    def reset(self, outliers: Iterable[bool]) -> None:
        """Initialize detector from a reference window."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, outliers: Iterable[bool]) -> bool:
        """Return True if drift is detected on the given window."""
        raise NotImplementedError

    @abstractmethod
    def message(self) -> str:
        raise NotImplementedError


class NoDriftDetector(DriftDetector):
    """Detector that never signals drift (baseline)."""

    def reset(self, outliers: Iterable[bool]) -> None:  # pragma: no cover - trivial
        return None

    def detect(self, outliers: Iterable[bool]) -> bool:
        return False

    def message(self) -> str:
        return "No drift detection"


class ThresholdDriftDetector(DriftDetector):
    """
    Simple detector: signal when the empirical outlier fraction exceeds f.
    """

    def __init__(self, f: float = 0.1) -> None:
        self.f = float(f)

    def reset(self, outliers: Iterable[bool]) -> None:  # notebook: no baseline stored
        return None

    def detect(self, outliers: Iterable[bool]) -> bool:
        outs = np.fromiter((bool(o) for o in outliers), dtype=bool)
        if len(outs) == 0:
            return False
        outlier_fraction = outs.mean()
        return bool(outlier_fraction > self.f)

    def message(self) -> str:
        return f"Threshold Drift Detection (f={self.f})"


class ZTestDriftDetector(DriftDetector):
    """
    Two-sided z-test detector on outlier proportion.

    - reset(outliers): store baseline proportion p_orig.
    - detect(outliers): compare new p_outliers to p_orig.
      Signals drift if |p_outliers - p_orig| > z_{1-alpha/2} / sqrt(2n).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.is_built = False
        self.tot_orig: int = 0
        self.n_orig: int = 0

    def reset(self, outliers: Iterable[bool]) -> None:
        outs = np.fromiter((bool(o) for o in outliers), dtype=bool)
        self.tot_orig = int(outs.sum())
        self.n_orig = int(len(outs))
        if self.n_orig == 0:
            raise ValueError("reset requires a non-empty baseline window")
        self.is_built = True

    def detect(self, outliers: Iterable[bool]) -> bool:
        if not self.is_built:
            raise RuntimeError("ZTestDriftDetector has not been reset.")

        outs = np.fromiter((bool(o) for o in outliers), dtype=bool)
        n = int(len(outs))
        if n == 0:
            return False

        tot = int(outs.sum())
        p_outliers = tot / n
        p_orig = self.tot_orig / self.n_orig

        z = st.norm()
        c = abs(p_outliers - p_orig)
        threshold = z.ppf(1 - self.alpha / 2) / math.sqrt(2 * n)
        return bool(c > threshold)

    def message(self) -> str:
        return rf"Z-test Drift Detection ($\alpha={self.alpha}$)"


__all__ = [
    "DriftDetector",
    "NoDriftDetector",
    "ThresholdDriftDetector",
    "ZTestDriftDetector",
]
