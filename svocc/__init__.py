"""svocc is a module implementing the SV one-class model.
"""

import copy
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from svocc import kernel
from svocc.optimization import GurobiSolver


class SVOCC(BaseEstimator, RegressorMixin):
    """SV one-class model."""

    defaults = {'c': 1,
                'k': kernel.GaussianKernel(),
                'solver': GurobiSolver(),
                'random_state': None}

    # pylint: disable-next=too-many-arguments
    def __init__(self,
                 c=defaults['c'],
                 k=defaults['k'],
                 solver=defaults['solver'],
                 random_state=defaults['random_state']):
        r"""Create an instance of :class:`SVOCC`.

        :param c: Trade-off constant, defaults to 1.
        :type c: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param solver: Solver to be used to obtain the optimization problem
          solution, defaults to `GurobiSolver()`.
        :type solver: :class:`mulearn.optimization.Solver`
        :param random_state: Seed of the pseudorandom generator.
        :type random_state: `int`
        """
        self.c = c
        self.k = k
        self.solver = solver
        self.random_state = random_state
        #pylint:disable-next=invalid-name
        self.X_ = None

    def __repr__(self, N_CHAR_MAX=700):
        args = [f'{p}={v}' for p, v in self.__dict__.items()
                           if p in self.defaults and v != self.defaults[p]]
        return f'{self.__class__.__name__}({", ".join(args)})'

    def generate_membership_(self):
        """Generate r2_05_, x_to_r2_ and estimated_membership_
        from the remaining instance variables."""

        def x_to_r2(x_new):
            return self.k.compute(x_new, x_new) \
                - 2 * np.array([self.k.compute(x_i, x_new)
                                for x_i in self.X_]).dot(self.chis_) \
                + self.fixed_term_

        sv_filter = (0 < self.chis_) & (self.chis_ < self.c )

        chi_sq_radius = np.array([x_to_r2(x) for x in self.X_[sv_filter]])

        if len(chi_sq_radius) == 0:
            raise ValueError('No support vectors found')

        # pylint: disable=attribute-defined-outside-init
        self.r2_05_ = np.mean(chi_sq_radius)

        self.m_ = np.log(2) / self.r2_05_
        self.x_to_r2_ = lambda X: np.array([x_to_r2(x) for x in X])
        self.x_to_mu_ = lambda X: np.exp(-1 * self.m_ * self.x_to_r2_(X))


    # pylint: disable-next=invalid-name
    def fit(self, X, warm_start=False):
        r"""Induce the membership function starting from a labeled sample.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          None.
        :type warm_start: `bool`
        :raises: ValueError if the values in `y` are not between 0 and 1
          (unless `force_rescale` in the constructor has been set to `True`),
          if `X` and have different lengths, or if `X` contains elements of
          different lengths.
        :return: A reference to the trained model.
        :rtype: `FuzzyInductor`
        """

        # pylint: disable=attribute-defined-outside-init
        self.estimated_membership_ = None
        self.x_to_r2_ = None
        self.r2_05_ = None
        self.fixed_term_ = None
        self.chis_ = None
        self.X_ = X

        X = check_array(X)

        if warm_start:
            check_is_fitted(self, ["chis_"])
            if self.chis_ is None:
                raise NotFittedError("chis variable are set to None")
            self.solver.initial_values = self.chis_

        gram = np.array([[self.k.compute(x1, x2) for x1 in X] for x2 in X])

        self.chis_ = self.solver.solve(X, self.c, self.k)

        self.fixed_term_ = np.array(self.chis_).dot(gram.dot(self.chis_))

        self.generate_membership_()

        self.n_features_in_ = X.shape[1]

        return self

    # pylint: disable-next=invalid-name
    def predict(self, X, scored=False):
        r"""Compute predictions for membership to the set.

        Predictions are either computed through the membership function (when
        `alpha` is set to `None`) or obtained via an $\alpha$-cut on
        the same function (when `alpha` is set to a float in [0, 1]).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param scored: flag triggering scored classification.
        :type alpha: bool
        :raises: ValueError if `alpha` is set to a value different from
          `None` and not included in $[0, 1]$.
        :return: Predictions for each value in `X`.
        :rtype: array of `float` (no $\alpha$-cut) or `int` ($\alpha$-cut)
        """

        check_is_fitted(self, ['chis_'])
        X = check_array(X)

        if scored:
            return self.x_to_mu_(X)
        else:
            return (self.x_to_r2_(X) > self.r2_05_).astype(int)

    def score(self, X, y, sample_weight=None):
        r"""Compute the fuzzifier score.

        Score is obtained as the opposite of MSE between predicted
        membership values and labels.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* membership values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- inverse of MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """
        check_X_y(X, y)
        check_is_fitted(self, ['chis_'])

        y_hat = self.predict(X)
        return -1 * mean_squared_error(y, y_hat)

    def get_params(self, deep=True):
        """Returns the parameter values needed to reconstruct the object

        :return: Dictionary having as keys the names of all parameters to
            the class constructor, and as values the corresponding values
            for the object on which the method is called.
        :rtype: `dict`
        """

        return {'c': self.c,
                'k': self.k,
                'solver': self.solver,
                'random_state': self.random_state}

    def set_params(self, **parameters):
        """Sets the parameters of the object and returns the modified object.

        :param parameters: Dictionary having as keys the names of all
            parameters to the class constructor, and as values the
            corresponding values to be set on the object on which the method
            is called.
        :type parameters: `dict`
        :return: A reference to the modified object.
        :rtype: CrispFuzzifier
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __getstate__(self):
        """Return a serializable description of the fuzzifier."""
        d = copy.deepcopy(self.__dict__)
        del d['estimated_membership_']
        del d['x_to_r2_']
        return d

    def __setstate__(self, d):
        """Ensure fuzzifier consistency after deserialization."""
        for parameter, value in d.items():
            setattr(self, parameter, value)

        self.__dict__ = d

        self.generate_membership_()
