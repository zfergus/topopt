#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy

from topopt.boundary_conditions import TopOptBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.utils import xy_to_id

from topopt import cmd_helper


class Exercise03DistributedBoundaryConditions(TopOptBoundaryConditions):
    @property
    def fixed_nodes(self):
        """ Return a list of fixed nodes for the problem. """
        bottom_left = 2 * xy_to_id(0, self.nely, self.nelx, self.nely)
        bottom_right = 2 * xy_to_id(self.nelx, self.nely, self.nelx, self.nely)
        fixed = numpy.array(
            [bottom_left, bottom_left + 1, bottom_right, bottom_right + 1])
        return fixed

    @property
    def forces(self):
        """ Return the force vector for the problem. """
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        nForces = topx.shape[0]
        cols = numpy.arange(nForces)
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), nForces))
        f[topx, cols] = -1
        return f

    @property
    def nonuniform_forces(self):
        """ Return the force vector for the problem. """
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        nForces = topx.shape[0]
        cols = numpy.arange(nForces)
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), nForces))
        f[topx, cols] = (0.5 * numpy.cos(
            numpy.linspace(0, 2 * numpy.pi, nForces)) - 0.5)
        return f


def main():
    # Default input parameters
    nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args(
        nelx=120, volfrac=0.2, penal=6.0, rmin=1.5)
    bc = Exercise03DistributedBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(nelx, nely, penal, bc)
    cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft, bc=bc,
                    problem=problem)


if __name__ == "__main__":
    main()
