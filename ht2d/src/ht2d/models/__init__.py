from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from river import tree
except Exception:  
    tree = None  # type: ignore

try:
    from river import naive_bayes
except Exception:  
    naive_bayes = None  # type: ignore

try:
    from river import linear_model
except Exception:  
    linear_model = None  # type: ignore

try:
    from river.forest import ARFClassifier #RandomForestClassifier
except Exception:  
    ARFClassifier = None  # type: ignore

try:
    from river import linear_model
except Exception:  # 
    linear_model = None  # type: ignore

try:
    from river import neural_net
except Exception:  #
    neural_net = None  # type: ignore

try:
    from river import optim
except Exception:  # 
    optim = None  # type: ignore

try:
    from sklearn.svm import LinearSVC
except Exception:  
    LinearSVC = None  # type: ignore

try:
    from sklearn.svm import SVC
except Exception:
    SVC = None  # type: ignore

try:
    from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
except Exception:  # 
    SklearnMLPClassifier = None  # type: ignore

def to_dict(x_array: Sequence[float]) -> Dict[str, float]:
    return {f"x_{i}": float(x_array[i]) for i in range(len(x_array))}


def to_array(x_dict: Mapping[str, float]) -> np.ndarray:
    # assumes keys are x_0, x_1, ..., x_{d-1}
    return np.array([x_dict[f"x_{i}"] for i in range(len(x_dict))], dtype=float)

class BaseModel(ABC):
    """Abstract model exposing a River-style online API."""

    @abstractmethod
    def learn_one(self, x: Mapping[str, float], y: Any) -> BaseModel:
        ...

    @abstractmethod
    def predict_one(self, x: Mapping[str, float]) -> Any:
        ...

    @abstractmethod
    def learn_many(self, X: Iterable[Sequence[float]], y: Iterable[Any]) -> BaseModel:
        """Stream many examples into an online model (dict-based)."""
        for x, y in zip(X, y):
            x_dict = to_dict(x)
            self.learn_one(x_dict, y)
        return self
    
    def window_accuracy(self, x_window: Iterable[np.ndarray], y_window: Iterable[int]) -> float:
        correct = 0
        total = 0
        for x, y in zip(x_window, y_window):
            y_hat = self.predict_one(to_dict(x))
            correct += int(y_hat == y)
            total += 1
        return correct / total if total > 0 else 0.0


class ModelAdapter(BaseModel):
    """
    Adapter to make a River model look like our standard online model.

    It exposes the same API:
    - learn_one(x_dict, y)
    - predict_one(x_dict)
    plus batch helpers: fit, learn_many, predict, predict_proba.
    """

    def __init__(self) -> None:
        self._estimator_type = "classifier"
        self.classes_: List[int] = []
        self.is_fitted_: bool = False

    # --- Online API ---

    def learn_one(self, x: Mapping[str, float], y: Any) -> ModelAdapter:
        self.model.learn_one(dict(x), y)
        if y not in self.classes_:
            self.classes_.append(y)
        self.is_fitted_ = True
        return self

    def predict_one(self, x: Mapping[str, float]) -> Any:
        return self.model.predict_one(dict(x))

    # --- Batch-style helpers (for our experiment code) ---

    def fit(self, X, y=None) -> ModelAdapter:
        self.learn_many(X, y)
        return self

    def learn_many(self, X, y) -> ModelAdapter:
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

class HoeffdingTreeClassifier(ModelAdapter):

    def __init__(self) -> None:
        self.model = tree.HoeffdingTreeClassifier()
        # call the superclass init
        super().__init__()


class GaussianNBClassifier(ModelAdapter):

    def __init__(self) -> None:
        self.model = naive_bayes.GaussianNB()
        # call the superclass init
        super().__init__()

class PAClassifier(ModelAdapter):
    """
    "Linear SVC-like" online classifier using River's PAClassifier.

    This is a good streaming analogue of a linear max-margin classifier.
    """

    def __init__(self, C: float = 1.0, mode: int = 1, learn_intercept: bool = True) -> None:
        if linear_model is None:
            raise ImportError("river is not installed. Install `river` to use LinearSVCClassifier.")
        self.model = linear_model.PAClassifier(C=C, mode=mode, learn_intercept=learn_intercept)
        super().__init__()


class SklearnBatchAdapter(BaseModel):
    """
    Wraps a scikit-learn batch estimator and exposes the ht2d online-ish API:
      - learn_many(X, y): fits on the full window
      - predict_one(x_dict): predicts for one example

    Note: learn_one is not supported (LinearSVC is batch).
    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.is_fitted_ = False
        self.classes_: List[Any] = []

    def learn_one(self, x: Mapping[str, float], y: Any) -> "SklearnBatchAdapter":
        raise NotImplementedError(
            "This is a batch (sklearn) model. Use learn_many(X, y) to refit on a window."
        )

    def learn_many(self, X: Iterable[Sequence[float]], y: Iterable[Any]) -> "SklearnBatchAdapter":
        X_arr = np.asarray(list(X), dtype=float)
        y_arr = np.asarray(list(y))
        self.estimator.fit(X_arr, y_arr)
        self.is_fitted_ = True
        # sklearn’s DecisionBoundaryDisplay expects numpy array-like classes_
        self.classes_ = list(getattr(self.estimator, "classes_", np.unique(y_arr)))
        return self

    def predict_one(self, x: Mapping[str, float]) -> Any:
        if not self.is_fitted_:
            return None
        x_arr = to_array(x).reshape(1, -1)
        return self.estimator.predict(x_arr)[0]

    def predict(self, X):
        return self.estimator.predict(np.asarray(X, dtype=float))

class SklearnLinearSVCClassifier(SklearnBatchAdapter):
    def __init__(self, C: float = 1.0, loss: str = "squared_hinge", max_iter: int = 5000) -> None:
        if LinearSVC is None:
            raise ImportError("scikit-learn is not installed. Install `scikit-learn` to use this model.")
        super().__init__(LinearSVC(C=C, loss=loss, max_iter=max_iter))

class SklearnRBFSVCClassifier(SklearnBatchAdapter):
    """
    Batch RBF SVM (scikit-learn) wrapped for ht2d experiments.

    Note:
    - Not online
    - Retrained on drift using the current window
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        max_iter: int = -1,
    ) -> None:
        if SVC is None:
            raise ImportError(
                "scikit-learn is not installed. Install `scikit-learn` to use RBF SVC."
            )

        super().__init__(
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                max_iter=max_iter,
            )
        )

class AdaptiveRandomForestClassifier(ModelAdapter):
    """
    River Adaptive Random Forest (online, drift-aware).
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int | None = None,
    ) -> None:
        if ARFClassifier is None:
            raise ImportError(
                "river is not installed. Install `river` to use AdaptiveRandomForestClassifier."
            )

        self.model = ARFClassifier(
            n_models=n_models,
            seed=seed,
        )
        super().__init__()


class LogisticRegressionClassifier(ModelAdapter):
    """
    River Logistic Regression (online, probabilistic).
    """

    def __init__(self) -> None:
        if linear_model is None:
            raise ImportError(
                "river is not installed. Install `river` to use LogisticRegression."
            )

        self.model = linear_model.LogisticRegression()
        super().__init__()

class SklearnMLPClassifier(ModelAdapter):
    """
    scikit-learn MLPClassifier
    """

    def __init__(
        self,
        hidden_layer_sizes=(50, 50),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=300,
        random_state=0,
    ) -> None:
        if SklearnMLPClassifier is None:
            raise ImportError("scikit-learn is not installed. Install `scikit-learn` to use SklearnMLPClassifier.")

        # sklearn model expects arrays, not dicts
        self.model = SklearnMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        )
        super().__init__()

    # override: sklearn uses fit/predict on arrays
    def learn_many(self, X, y) -> "SklearnMLPClassifier":
        self.model.fit(np.asarray(X), np.asarray(y))
        self.is_fitted_ = True
        # keep classes_ as numpy array for sklearn tools
        self.classes_ = list(getattr(self.model, "classes_", []))
        return self

    def predict_one(self, x: Mapping[str, float]) -> Any:
        xi = to_array(x).reshape(1, -1)
        return self.model.predict(xi)[0]

    def predict(self, X):
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))



__all__ = [
    "to_dict",
    "to_array",
    "BaseModel",
    "ModelAdapter",
    "HoeffdingTreeClassifier",
    "GaussianNBClassifier",
    "PAClassifier",
    "SklearnBatchAdapter",
    "SklearnLinearSVCClassifier",
    "SklearnRBFSVCClassifier",
    "AdaptiveRandomForestClassifier",
    "LogisticRegressionClassifier",
    "sklearnMLPClassifier",
]
