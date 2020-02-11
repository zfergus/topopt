#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multiple loads."""
from __future__ import division

import numpy

import context  # noqa

from topopt.boundary_conditions import BoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.utils import xy_to_id

from topopt import cli


class MultipleLoadsBoundaryConditions(BoundaryConditions):
    """Multiple loads applied to the top boundary."""

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
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 2))
        id1 = 2 * xy_to_id(7 * self.nelx // 20, 0, self.nelx, self.nely) + 1
        id2 = 2 * xy_to_id(13 * self.nelx // 20, 0, self.nelx, self.nely) + 1
        f[id1, 0] = -1
        f[id2, 1] = -1
        return f


def main():
    """Multiple loads."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, rmin=1.5)
    bc = MultipleLoadsBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, bc=bc,
             problem=problem)


if __name__ == "__main__":
    main()
