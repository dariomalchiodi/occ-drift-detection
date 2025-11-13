from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Iterable, List, Sequence, Tuple, Union, Optional

import numpy as np


class Distribution(ABC):
    """Abstract base for data-generating distributions."""

    @abstractmethod
    def sample(self) -> Iterable[Tuple[dict, int]]:
        """Yield (x, y) pairs."""
        raise NotImplementedError


class GaussianMixture(Distribution):
    """
    Mixture of multivariate Gaussians producing (x_dict, class) pairs.

    Parameters
    ----------
    mu : list[list[float]]
        Means for each component.
    sigma : list[list[float]]
        Diagonal std deviations for each component.
    class_ : list[int]
        Class label per component.
    p : list[float]
        Mixing probabilities (must sum to 1).
    n : int
        Number of samples to generate.
    """

    def __init__(
        self,
        mu: Sequence[Sequence[float]],
        sigma: Sequence[Sequence[float]],
        class_: Sequence[int],
        p: Sequence[float],
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not (len(mu) == len(sigma) == len(class_) == len(p)):
            raise ValueError("All input lists must have the same length.")
        if not np.isclose(sum(p), 1.0):
            raise ValueError("Probabilities p must sum to 1.")

        self.mu = [list(m) for m in mu]
        self.sigma = [list(s) for s in sigma]
        self.class_ = list(class_)
        self.p = list(p)
        self.n = int(n)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> Generator[Tuple[dict, int], None, None]:
        for _ in range(self.n):
            i = self.rng.choice(len(self.p), p=self.p)
            mu = np.asarray(self.mu[i], dtype=float)
            sigma = np.asarray(self.sigma[i], dtype=float)
            x = self.rng.multivariate_normal(mu, np.diag(sigma))
            y = int(self.class_[i])
            yield {f"x_{j}": float(xj) for j, xj in enumerate(x)}, y


class DriftingMixtureStream:
    """
    Stream that switches between GaussianMixture instances with abrupt or gradual drift.

    mixtures : list[GaussianMixture], len >= 2
    drift_points : list of ints or (start, end) tuples, len == len(mixtures) - 1

    - int k: abrupt drift from mixture i to i+1 at t = k
    - (start, end): gradual drift, interpolating params between start and end
    """

    def __init__(
        self,
        mixtures: Sequence[GaussianMixture],
        drift_points: Sequence[Union[int, Tuple[int, int]]],
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(mixtures) < 2:
            raise ValueError("Need at least two mixtures to simulate drift.")
        if len(drift_points) != len(mixtures) - 1:
            raise ValueError("drift_points must have length len(mixtures) - 1.")

        self.mixtures = list(mixtures)
        self.drift_points = list(drift_points)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0
        self.current_idx = 0

    def _interpolate_mixtures(
        self,
        mix_a: GaussianMixture,
        mix_b: GaussianMixture,
        alpha: float,
    ) -> GaussianMixture:
        """Return a GaussianMixture interpolating parameters between A and B."""
        mu = [
            (1 - alpha) * np.array(ma) + alpha * np.array(mb)
            for ma, mb in zip(mix_a.mu, mix_b.mu)
        ]
        sigma = [
            (1 - alpha) * np.array(sa) + alpha * np.array(sb)
            for sa, sb in zip(mix_a.sigma, mix_b.sigma)
        ]
        p = [(1 - alpha) * pa + alpha * pb for pa, pb in zip(mix_a.p, mix_b.p)]
        total_p = sum(p)
        p = [pi / total_p for pi in p]
        class_ = mix_a.class_
        return GaussianMixture(mu, sigma, class_, p, n=1_000_000, rng=self.rng)

    def sample(self) -> Generator[Tuple[dict, int], None, None]:
        while True:
            if self.current_idx >= len(self.mixtures) - 1:
                # After the last configured drift: use final mixture
                yield next(self.mixtures[-1].sample())
                self.t += 1
                continue

            drift = self.drift_points[self.current_idx]

            if isinstance(drift, int):
                # Abrupt drift
                if self.t < drift:
                    yield next(self.mixtures[self.current_idx].sample())
                else:
                    self.current_idx += 1
                    yield next(self.mixtures[self.current_idx].sample())
            else:
                # Gradual drift: (start, end)
                start, end = drift
                if self.t < start:
                    yield next(self.mixtures[self.current_idx].sample())
                elif self.t > end:
                    self.current_idx += 1
                    yield next(self.mixtures[self.current_idx].sample())
                else:
                    alpha = (self.t - start) / (end - start)
                    mix = self._interpolate_mixtures(
                        self.mixtures[self.current_idx],
                        self.mixtures[self.current_idx + 1],
                        alpha,
                    )
                    yield next(mix.sample())

            self.t += 1


__all__ = ["Distribution", "GaussianMixture", "DriftingMixtureStream"]
