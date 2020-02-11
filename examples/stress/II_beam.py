#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""II-shaped beam domain with stress computation."""
from __future__ import division

import context  # noqa

from topopt.boundary_conditions import IIBeamBoundaryConditions
from topopt.guis import StressGUI
from topopt.problems import VonMisesStressProblem
import topopt.filters
from topopt.solvers import TopOptSolver

from topopt import cli


def main():
    """II-shaped beam domain with stress computation."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, nely=120, volfrac=0.3, penalty=12, rmin=1.2)
    bc = IIBeamBoundaryConditions(nelx, nely)
    problem = VonMisesStressProblem(nelx, nely, penalty, bc)
    gui = StressGUI(problem, title="Stress on II Beam")
    filter = [topopt.filters.SensitivityBasedFilter,
              topopt.filters.DensityBasedFilter][ft](nelx, nely, rmin)
    solver = TopOptSolver(
        problem, volfrac, filter, gui, maxeval=4000, ftol_rel=1e-5)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, solver=solver)


if __name__ == "__main__":
    main()
