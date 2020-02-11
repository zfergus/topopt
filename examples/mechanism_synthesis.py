#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a compliance mechanism that inverts displacments."""

import context  # noqa

from topopt.mechanisms.boundary_conditions import (
    DisplacementInverterBoundaryConditions,
    CrossSensitivityExampleBoundaryConditions,
    GripperBoundaryConditions)
from topopt.mechanisms.problems import MechanismSynthesisProblem
from topopt.mechanisms.solvers import MechanismSynthesisSolver
from topopt.filters import SensitivityBasedFilter, DensityBasedFilter
from topopt.guis import GUI

from topopt import cli


def main():
    """Run the example by constructing the TopOpt objects."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=100, nely=100, volfrac=0.3, penalty=10, rmin=1.4)
    bc = DisplacementInverterBoundaryConditions(nelx, nely)
    # bc = GripperBoundaryConditions(nelx, nely)
    # bc = CrossSensitivityExampleBoundaryConditions(nelx, nely)
    problem = MechanismSynthesisProblem(bc, penalty)
    title = cli.title_str(nelx, nely, volfrac, rmin, penalty)
    gui = GUI(problem, title)
    filter = [SensitivityBasedFilter, DensityBasedFilter][ft](nelx, nely, rmin)
    solver = MechanismSynthesisSolver(problem, volfrac, filter, gui)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, solver=solver)


if __name__ == "__main__":
    main()
