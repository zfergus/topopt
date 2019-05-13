#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy

from topopt.boundary_conditions import TopOptBoundaryConditions
from topopt.problems import MechanismSynthesisProblem
import topopt.filters
from topopt.guis import GUI
from topopt.solvers import MechanismSynthesisSolver
from topopt.utils import xy_to_id

from topopt import cmd_helper


class Exercise05BoundaryConditions(TopOptBoundaryConditions):
    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        # topx_to_id = numpy.vectorize(
        #     lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        # topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        id1 = 2 * xy_to_id(0, 0, self.nelx, self.nely)
        id2 = 2 * xy_to_id(0, self.nely, self.nelx, self.nely)
        # fixed = numpy.union1d(topx, numpy.array([id2, id2 + 1]))
        fixed = numpy.array([id1, id1 + 1, id2, id2 + 1])
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 2))
        # f[2 * xy_to_id(0, 0, self.nelx, self.nely), 0] = 1
        f[2 * xy_to_id(0, self.nely // 2, self.nelx, self.nely), 0] = 1
        # f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        f[2 * xy_to_id(
            self.nelx, self.nely // 2, self.nelx, self.nely), 1] = -1
        return f


def main():
    # Default input parameters
    nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args(
        nelx=100, nely=100, volfrac=0.3, penal=10, rmin=1.4)
    bc = Exercise05BoundaryConditions(nelx, nely)
    problem = MechanismSynthesisProblem(nelx, nely, penal, bc)
    title = cmd_helper.title_str(nelx, nely, volfrac, rmin, penal)
    gui = GUI(problem, title)
    filter = [topopt.filters.SensitivityBasedFilter,
        topopt.filters.DensityBasedFilter][ft](nelx, nely, rmin)
    solver = MechanismSynthesisSolver(problem, volfrac, filter, gui)
    cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft, solver=solver)


if __name__ == "__main__":
    main()
