def test_public_api():
    import ht2d

    assert hasattr(ht2d, "GaussianMixture")
    assert hasattr(ht2d, "DriftingMixtureStream")
    assert hasattr(ht2d, "NoDriftDetector")
    assert hasattr(ht2d, "ThresholdDriftDetector")
    assert hasattr(ht2d, "ZTestDriftDetector")
    assert hasattr(ht2d, "HoeffdingTreeClassifier")
