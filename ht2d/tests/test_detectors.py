import numpy as np
from ht2d.detectors import NoDriftDetector, ThresholdDriftDetector, ZTestDriftDetector


def test_no_drift_detector_never_signals():
    det = NoDriftDetector()
    window = [False] * 100
    det.reset(window)
    assert det.detect(window) is False


def test_threshold_drift_detector_triggers_on_high_fraction():
    det = ThresholdDriftDetector(f=0.2)
    # below threshold
    w1 = [False] * 90 + [True] * 10  # 10%
    assert det.detect(w1) is False
    # above threshold
    w2 = [False] * 70 + [True] * 30  # 30%
    assert det.detect(w2) is True


def test_ztest_drift_detector_detects_strong_shift():
    rng = np.random.default_rng(0)
    det = ZTestDriftDetector(alpha=0.05)

    # baseline: very low outlier rate
    base = rng.random(500) < 0.05
    det.reset(base)

    # new window: much higher outlier rate
    new = rng.random(500) < 0.30
    assert det.detect(new) is True
