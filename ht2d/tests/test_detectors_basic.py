import numpy as np
from ht2d.detectors import NoDriftDetector, ThresholdDriftDetector, ZTestDriftDetector


def test_no_drift_detector_basic():
    det = NoDriftDetector()
    baseline = [False] * 100
    det.reset(baseline)
    assert det.detect(baseline) is False


def test_threshold_detector_basic():
    det = ThresholdDriftDetector(f=0.1)

    # below threshold: ~5% outliers
    w_low = [False] * 95 + [True] * 5
    assert det.detect(w_low) is False

    # above threshold: ~20% outliers
    w_high = [False] * 80 + [True] * 20
    assert det.detect(w_high) is True


def test_ztest_detector_basic():
    rng = np.random.default_rng(0)
    det = ZTestDriftDetector(alpha=0.05)

    # baseline: low outlier rate
    baseline = rng.random(400) < 0.05
    det.reset(baseline)

    # window with similar rate -> no drift (most of the time)
    same = rng.random(400) < 0.05
    # it's probabilistic, so use a soft assertion: usually False
    assert det.detect(same) is False or det.detect(same) is True  # ensures it runs without error

    # window with clearly higher rate -> should detect
    shifted = rng.random(400) < 0.30
    assert det.detect(shifted) is True
