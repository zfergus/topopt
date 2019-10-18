"""Solve compliant mechanism synthesis problems using topology optimization."""

import numpy

from ..solvers import TopOptSolver


class MechanismSynthesisSolver(TopOptSolver):
    """
    Specialized solver for mechanism synthesis problems.

    This solver is specially designed to create `compliant mechanisms
    <https://en.wikipedia.org/wiki/Compliant_mechanism>`_.
    """

    def __init__(self, problem, volfrac, filter, gui, maxeval=2000, ftol=1e-4):
        """
        Create a mechanism synthesis solver to solve the problem.

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
        super().__init__(problem, volfrac, filter, gui, maxeval, ftol)
        self.init_obj = None
        self.vtot = problem.nelx * problem.nely * volfrac

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

        # Display physical variables
        self.gui.update(self.xPhys)

        # Objective and sensitivity
        obj = self.problem.compute_objective(self.xPhys, dobj)
        if self.init_obj is None:
            self.init_obj = obj
        obj /= self.init_obj

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)
        dobj /= self.init_obj

        # print(obj * self.init_obj)
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
        dv[:] = 1.0 / self.vtot

        # Sensitivity filtering
        self.filter.filter_volume_sensitivities(self.xPhys, dv)

        return self.xPhys.sum() / self.vtot - 1
