"""Implementation of optimization procedures.

This module module contains the implementations of the optimization processes
behind SV one-class training and inference.

Once loaded, the module preliminarily verifies that Gurobi is installed,
emitting a warning otherwise. Note that at this librariy is needed in order to
solve the optimization problems involved in SV one-class training.
"""

import itertools as it

import numpy as np

try:
    from gurobipy import LinExpr, GRB, Model, Env, QuadExpr, GurobiError

    GUROBI_OK = True
except ModuleNotFoundError:
    print('gurobi not available')
    GUROBI_OK = False


class Solver:
    """Abstract solver for optimization problems.

    The base class for solvers is :class:`Solver`: it exposes a method
    `solve` which delegates the numerical optimization process to an abstract
    method `solve_problem` and subsequently clips the results to the boundaries
    of the feasible region.
    """

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def solve_problem(self, xs, c, k):
        """Solve the constrained optimization problem for SV one-class."""
        pass #pylint:disable=unnecessary-pass

    def solve(self, xs, c, k):
        """Solve optimization phase.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure.

        :param xs: Objects in training set.
        :type xs: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: Kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if c is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""
        if c <= 0:
            raise ValueError('c should be positive')

        #pylint:disable-next=assignment-from-no-return
        chis = self.solve_problem(xs, c, k)

        chis_opt = np.clip(chis, 0, c)

        return chis_opt


class GurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    default_values = {"time_limit": 10 * 60,
                      "adjustment": 'auto',
                      "initial_values": None}

    def __init__(self, time_limit=default_values['time_limit'],
                 adjustment=default_values['adjustment'],
                 initial_values=default_values['initial_values']):
        """
        Build an object of type GurobiSolver.

        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param adjustment: Adjustment value to be used with non-PSD matrices,
          defaults to 0. Specifying `'auto'` instead than a numeric value
          will automatically trigger the optimal adjustment if needed.
        :type adjustment: float or `'auto'`
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        """
        self.time_limit = time_limit
        self.adjustment = adjustment
        self.initial_values = initial_values

    def solve_problem(self, xs, c, k):
        """Optimize via gurobi.

        Build and solve the constrained optimization problem at the basis
        of the fuzzy learning procedure using the gurobi API.

        :param xs: objects in training set.
        :type xs: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if optimization fails or if gurobi is not installed
        :returns: list -- optimal values for the independent variables of the
          problem.
        """
        if not GUROBI_OK:
            raise ValueError('gurobi not available')

        m = len(xs)

        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with Model('mulearn', env=env) as model:
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', self.time_limit)

                for i in range(m):
                    if c < np.inf:
                        model.addVar(name=f'chi_{i}', lb=0, ub=c,
                                     vtype=GRB.CONTINUOUS)
                    else:
                        model.addVar(name=f'chi_{i}', lb=0,
                                     vtype=GRB.CONTINUOUS)

                model.update()
                chis = model.getVars()

                if self.initial_values is not None:
                    for c, i in zip(chis, self.initial_values):
                        c.start = i

                obj = QuadExpr()

                for i, j in it.product(range(m), range(m)):
                    obj.add(chis[i] * chis[j], k.compute(xs[i], xs[j]))

                for i in range(m):
                    obj.add(-1 * chis[i] * k.compute(xs[i], xs[i]))

                if self.adjustment and self.adjustment != 'auto':
                    for i in range(m):
                        obj.add(self.adjustment * chis[i] * chis[i])

                model.setObjective(obj, GRB.MINIMIZE)

                const_equal = LinExpr()
                const_equal.add(sum(chis), 1.0)

                model.addConstr(const_equal, GRB.EQUAL, 1)

                try:
                    model.optimize()
                except GurobiError as e:
                    if self.adjustment == 'auto':
                        s = e.message
                        a = float(s[s.find(' of ') + 4:s.find(' would')])
                        print(f'Gram matrix adjustment of {a:f} is applied')

                        # logger.warning('non-diagonal Gram matrix, '
                        #                f'retrying with adjustment {a}')
                        for i in range(m):
                            obj.add(a * chis[i] * chis[i])
                        model.setObjective(obj, GRB.MINIMIZE)

                        model.optimize()
                    else:
                        raise e

                if model.Status != GRB.OPTIMAL:
                    raise ValueError('optimal solution not found!')

                return [ch.x for ch in chis]

    def __repr__(self):
        obj_repr = "GurobiSolver("

        for a in ('time_limit', 'adjustment', 'initial_values'):
            if self.__getattribute__(a) != self.default_values[a]:
                obj_repr += f", {a}={self.default_values[a]}"
        return obj_repr + ")"
