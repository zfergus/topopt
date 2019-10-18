"""
Solvers to solve topology optimization problems.

Todo:
    * Make TopOptSolver an abstract class
    * Rename the current TopOptSolver to MMASolver(TopOptSolver)
    * Create a TopOptSolver using originality criterion
"""
from __future__ import division

import numpy
import nlopt

from topopt.problems import Problem
from topopt.filters import Filter
from topopt.guis import GUI


class TopOptSolver:
    """Solver for topology optimization problems using NLopt's MMA solver."""

    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, maxeval=2000, ftol=1e-3):
        """
        Create a solver to solve the problem.

        Parameters
        ----------
        problem: :obj:`topopt.problems.Problem`
            The topology optimization problem to solve.
        volfrac: float
            The maximum fraction of the volume to use.
        filter: :obj:`topopt.filters.Filter`
            A filter for the solutions to reduce artefacts.
        gui: :obj:`topopt.guis.GUI`
            The graphical user interface to visualize intermediate results.
        maxeval: int
            The maximum number of evaluations to perform.
        ftol: float
            A floating point tolerance for relative change.

        """
        self.problem = problem
        self.filter = filter
        self.gui = gui

        n = problem.nelx * problem.nely
        self.opt = nlopt.opt(nlopt.LD_MMA, n)
        self.xPhys = numpy.ones(n)

        # set bounds
        self.opt.set_upper_bounds(numpy.ones(n))
        self.opt.set_lower_bounds(numpy.zeros(n))

        # set stopping criteria
        self.opt.set_maxeval(maxeval)
        self.opt.set_ftol_rel(ftol)

        # set objective and constraint functions
        self.opt.set_min_objective(self.objective_function)
        self.opt.add_inequality_constraint(self.volume_function, 0)
        self.volfrac = volfrac  # max volume fraction to use

        # setup filter
        self.passive = problem.bc.passive_elements
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        self.active = problem.bc.active_elements
        if self.active.size > 0:
            self.xPhys[self.active] = 1

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the solver."""
        return "{} with {}".format(str(self.problem), str(self))

    def __repr__(self):
        """Create a representation of the solver."""
        return "{}(problem={!r}, volfrac={:g}, filter={!r}, gui={!r}, maxeval={:d}, ftol={:g})".format(
            self.__class__.__name__, self.problem, self.volfrac,
            self.filter, self.gui, self.opt.get_maxeval(),
            self.opt.get_ftol_rel())

    def optimize(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Optimize the problem.

        Parameters
        ----------
            x: numpy.ndarray
                The initial value for the design variables.

        Returns
        -------
            The optimal value of x found.

        """
        self.xPhys = x.copy()
        x = self.opt.optimize(x)
        return x

    def filter_variables(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Filter the variables and impose values on passive/active variables.

        Parameters
        ----------
            x: numpy.ndarray
                The variables to be filtered.

        Returns
        -------
            numpy.ndarray
                The filtered "physical" variables.

        """
        self.filter.filter_variables(x, self.xPhys)
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        if self.active.size > 0:
            self.xPhys[self.active] = 1
        return self.xPhys

    def objective_function(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        """
        Compute the objective value and gradient.

        Parameters
        ----------
            x: numpy.ndarray
                The design variables for which to compute the objective.
            dobj: numpy.ndarray
                The gradient of the objective to compute.

        Returns
        -------
            float
                The objective value.

        """
        # Filter design variables
        self.filter_variables(x)

        # Setup and solve FE problem
        self.problem.update_displacements(self.xPhys)

        # Objective and sensitivity
        obj = self.problem.compute_objective(self.xPhys, dobj)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        self.gui.update(self.xPhys)

        return obj

    def objective_function_fdiff(self, x: numpy.ndarray, dobj: numpy.ndarray,
                                 epsilon=1e-6) -> float:
        """
        Compute the objective value and gradient using finite differences.

        Parameters
        ----------
            x: numpy.ndarray
                The design variables for which to compute the objective.
            dobj: numpy.ndarray
                The gradient of the objective to compute.
            epsilon: float
                Change in the finite difference to compute the gradient.

        Returns
        -------
            float
                The objective value.

        """
        obj = self.objective_function(x, dobj)

        x0 = x.copy()
        dobj0 = dobj.copy()
        dobjf = numpy.zeros(dobj.shape)
        for i, v in enumerate(x):
            x = x0.copy()
            x[i] += epsilon
            o1 = self.objective_function(x, dobj)
            x[i] = x0[i] - epsilon
            o2 = self.objective_function(x, dobj)
            dobjf[i] = (o1 - o2) / (2 * epsilon)
            print("finite differences: {:g}".format(
                numpy.linalg.norm(dobjf - dobj0)))
            dobj[:] = dobj0

        return obj

    def volume_function(self, x: numpy.ndarray, dv: numpy.ndarray) -> float:
        """
        Compute the volume constraint value and gradient.

        Parameters
        ----------
            x: numpy.ndarray
                The design variables for which to compute the volume
                constraint.
            dobj: numpy.ndarray
                The gradient of the volume constraint to compute.

        Returns
        -------
            float
                The volume constraint value.

        """
        # Filter design variables
        self.filter_variables(x)

        # Volume sensitivities
        dv[:] = 1.0

        # Sensitivity filtering
        self.filter.filter_volume_sensitivities(self.xPhys, dv)

        return self.xPhys.sum() - self.volfrac * x.size


# TODO: Seperate optimizer from TopOptSolver
# class MMASolver(TopOptSolver):
#     pass
#
#
# TODO: Port over OC to TopOptSolver
# class OCSolver(TopOptSolver):
#     def oc(self, x, volfrac, dc, dv, g):
#         """ Optimality criterion """
#         l1 = 0
#         l2 = 1e9
#         move = 0.2
#         # reshape to perform vector operations
#         xnew = np.zeros(nelx * nely)
#         while (l2 - l1) / (l1 + l2) > 1e-3:
#             lmid = 0.5 * (l2 + l1)
#             xnew[:] =  np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
#                 np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
#             gt = g + np.sum((dv * (xnew - x)))
#             if gt > 0:
#                 l1 = lmid
#             else:
#                 l2 = lmid
#         return (xnew, gt)
