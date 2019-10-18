#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distributed load."""
from __future__ import division

import numpy

import context  # noqa

from topopt.boundary_conditions import BoundaryConditions
from topopt.utils import xy_to_id

from topopt import cli


class DistributedLoadBoundaryConditions(BoundaryConditions):
    """Distributed load along the top boundary."""

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        bottom_left = 2 * xy_to_id(0, self.nely, self.nelx, self.nely)
        bottom_right = 2 * xy_to_id(
            self.nelx, self.nely, self.nelx, self.nely)
        fixed = numpy.array([bottom_left, bottom_left + 1,
                             bottom_right, bottom_right + 1])
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 1))
        f[topx, 0] = -1
        return f

    @property
    def nonuniform_forces(self):
        """Return the force vector for the problem."""
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 1))
        f[topx, 0] = (0.5 * numpy.cos(
            numpy.linspace(0, 2 * numpy.pi, topx.shape[0])) - 0.5)
        return f


def main():
    """Distributed load."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, rmin=1.2)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft,
             bc=DistributedLoadBoundaryConditions(nelx, nely))


if __name__ == "__main__":
    main()
