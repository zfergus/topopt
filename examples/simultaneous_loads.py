#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two simultaneous point loads."""
from __future__ import division

import numpy

import context  # noqa

from topopt.boundary_conditions import BoundaryConditions
from topopt.utils import xy_to_id

from topopt import cli


class SimultaneousLoadsBoundaryConditions(BoundaryConditions):
    """Two simultaneous point loads along the top boundary."""

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
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 1))
        id1 = 2 * xy_to_id(7 * self.nelx // 20, 0, self.nelx, self.nely) + 1
        id2 = 2 * xy_to_id(13 * self.nelx // 20, 0, self.nelx, self.nely) + 1
        f[[id1, id2], 0] = -1
        return f


def main():
    """Two simultaneous point loads."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, rmin=1.5)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft,
             bc=SimultaneousLoadsBoundaryConditions(nelx, nely))


if __name__ == "__main__":
    main()
