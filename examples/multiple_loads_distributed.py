#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distributed multiple loads example."""
from __future__ import division

import numpy

import context  # noqa

from topopt.boundary_conditions import BoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.utils import xy_to_id

from topopt import cli


class DistributedMultipleLoadsBoundaryConditions(BoundaryConditions):
    """Distributed multiple loads boundary conditions."""

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        bottom_left = 2 * xy_to_id(0, self.nely, self.nelx, self.nely)
        bottom_right = 2 * xy_to_id(self.nelx, self.nely, self.nelx, self.nely)
        fixed = numpy.array(
            [bottom_left, bottom_left + 1, bottom_right, bottom_right + 1])
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
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
        """Return the force vector for the problem."""
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
    """Distributed multiple loads example."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, penalty=6.0, rmin=1.5)
    bc = DistributedMultipleLoadsBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, bc=bc,
             problem=problem)


if __name__ == "__main__":
    main()
