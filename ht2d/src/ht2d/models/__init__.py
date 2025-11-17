from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from river import tree
except Exception:  # pragma: no cover - optional dep
    tree = None  # type: ignore

try:
    from river import naive_bayes
except Exception:  # pragma: no cover - optional dep
    naive_bayes = None  # type: ignore


def to_dict(x_array: Sequence[float]) -> Dict[str, float]:
    return {f"x_{i}": float(x_array[i]) for i in range(len(x_array))}


def to_array(x_dict: Mapping[str, float]) -> np.ndarray:
    # assumes keys are x_0, x_1, ..., x_{d-1}
    return np.array([x_dict[f"x_{i}"] for i in range(len(x_dict))], dtype=float)


def window_accuracy(model: Any, x_window: Iterable[np.ndarray], y_window: Iterable[int]) -> float:
    correct = 0
    total = 0
    for x, y in zip(x_window, y_window):
        y_hat = model.predict_one(to_dict(x))
        correct += int(y_hat == y)
        total += 1
    return correct / total if total > 0 else 0.0


class BaseModelAdapter(ABC):
    """Abstract adapter exposing a River-style online API."""

    @abstractmethod
    def learn_one(self, x: Mapping[str, float], y: Any) -> "BaseModelAdapter":
        ...

    @abstractmethod
    def predict_one(self, x: Mapping[str, float]) -> Any:
        ...

    # @abstractmethod
    # def learn_many(self, X: Iterable[Sequence[float]], y: Iterable[Any]) -> "BaseModelAdapter":
    #     ...



class HoeffdingTreeClassifier(BaseModelAdapter):
    """
    Adapter to make a River HoeffdingTreeClassifier look like an sklearn-style estimator.

    This mirrors the notebook's HoeffdingTreeClassifier wrapper.
    """

    def __init__(self, river_model: Any | None = None) -> None:
        if river_model is not None:
            self.model = river_model
        else:
            if tree is None:
                raise ImportError(
                    "river is not installed. Install `river` to use HoeffdingTreeClassifier."
                )
            self.model = tree.HoeffdingTreeClassifier()

        self._estimator_type = "classifier"
        self.classes_: List[int] = []
        self.is_fitted_: bool = False

    # Online API
    def learn_one(self, x: Mapping[str, float], y: Any) -> "HoeffdingTreeClassifier":
        self.model.learn_one(dict(x), y)
        if y not in self.classes_:
            self.classes_.append(y)
        self.is_fitted_ = True
        return self

    def predict_one(self, x: Mapping[str, float]) -> Any:
        return self.model.predict_one(dict(x))

    # Batch-style helpers mirroring the notebook

    def fit(self, X, y=None) -> "HoeffdingTreeClassifier":
        self.learn_many(X, y)
        return self

    def learn_many(self, X, y) -> "HoeffdingTreeClassifier":
        for xi, yi in zip(X, y):
            self.learn_one(to_dict(xi), yi)
        return self

    def predict(self, X):
        return np.array([self.model.predict_one(to_dict(xi)) for xi in X])

    def predict_proba(self, X):
        # Only works if underlying model supports predict_proba_one
        probs = []
        for xi in X:
            p = self.model.predict_proba_one(to_dict(xi))
            classes = list(self.model.classes)
            probs.append([p.get(c, 0.0) for c in classes])
        return np.array(probs)

class GaussianNBClassifier(BaseModelAdapter):
    """
    Adapter to make river.naive_bayes.GaussianNB look like our standard online model.

    It exposes the same API:
    - learn_one(x_dict, y)
    - predict_one(x_dict)
    plus batch helpers: fit, learn_many, predict, predict_proba.
    """

    def __init__(self, river_model: Any | None = None) -> None:
        if river_model is not None:
            self.model = river_model
        else:
            if naive_bayes is None:
                raise ImportError(
                    "river is not installed. Install `river` to use GaussianNBClassifier."
                )
            self.model = naive_bayes.GaussianNB()

        self._estimator_type = "classifier"
        self.classes_: List[int] = []
        self.is_fitted_: bool = False

    # --- Online API ---

    def learn_one(self, x: Mapping[str, float], y: Any) -> "GaussianNBClassifier":
        self.model.learn_one(dict(x), y)
        if y not in self.classes_:
            self.classes_.append(y)
        self.is_fitted_ = True
        return self

    def predict_one(self, x: Mapping[str, float]) -> Any:
        return self.model.predict_one(dict(x))

    # --- Batch-style helpers (for our experiment code) ---

    def fit(self, X, y=None) -> "GaussianNBClassifier":
        self.learn_many(X, y)
        return self

    def learn_many(self, X, y) -> "GaussianNBClassifier":
        for xi, yi in zip(X, y):
            self.learn_one(to_dict(xi), yi)
        return self

    def predict(self, X):
        return np.array([self.model.predict_one(to_dict(xi)) for xi in X])

    def predict_proba(self, X):
        # Only works if underlying model supports predict_proba_one
        if not hasattr(self.model, "predict_proba_one"):
            raise AttributeError(
                "Underlying river model does not support predict_proba_one."
            )
        probs = []
        for xi in X:
            p = self.model.predict_proba_one(to_dict(xi))
            classes = list(self.model.classes)
            probs.append([p.get(c, 0.0) for c in classes])
        return np.array(probs)



__all__ = [
    "to_dict",
    "to_array",
    "window_accuracy",
    "BaseModelAdapter",
    "HoeffdingTreeClassifier",
    "GaussianNBClassifier",
]
