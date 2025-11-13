import itertools as it
import numpy as np
from ht2d.distributions import GaussianMixture, DriftingMixtureStream


def test_gaussian_mixture_shapes():
    g = GaussianMixture(
        mu=[[-2, -2], [2, 2]],
        sigma=[[0.1, 0.2], [0.3, 0.4]],
        class_=[0, 1],
        p=[0.5, 0.5],
        n=100,
    )
    xs, ys = zip(*it.islice(g.sample(), 50))
    assert len(xs) == 50
    assert all(set(x.keys()) == {"x_0", "x_1"} for x in xs)
    assert set(ys).issubset({0, 1})


def test_drifting_mixture_stream_runs():
    rng = np.random.default_rng(0)
    g1 = GaussianMixture(
        mu=[[0, 0], [1, 1]],
        sigma=[[0.2, 0.2], [0.2, 0.2]],
        class_=[0, 1],
        p=[0.5, 0.5],
        n=10000,
        rng=rng,
    )
    g2 = GaussianMixture(
        mu=[[0, 1], [0, 0]],
        sigma=[[0.4, 0.4], [0.4, 0.4]],
        class_=[0, 1],
        p=[0.5, 0.5],
        n=10000,
        rng=rng,
    )
    stream = DriftingMixtureStream(mixtures=[g1, g2], drift_points=[2000], rng=rng)
    it_stream = it.islice(stream.sample(), 3000)
    count = sum(1 for _ in it_stream)
    assert count == 3000
