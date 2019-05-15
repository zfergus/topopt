#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a compliance mechanism that inverts displacments."""

from context import topopt  # noqa

from topopt.mechanisms.boundary_conditions import (
    DisplacementInverterBoundaryConditions)
from topopt.mechanisms.problems import MechanismSynthesisProblem
from topopt.mechanisms.solvers import MechanismSynthesisSolver

from topopt.filters import SensitivityBasedFilter, DensityBasedFilter
from topopt.guis import GUI

from topopt import cmd_helper


def main():
    """Run the example by constructing the TopOpt objects."""
    # Default input parameters
    nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args(
        nelx=100, nely=100, volfrac=0.3, penal=10, rmin=1.4)
    bc = DisplacementInverterBoundaryConditions(nelx, nely)
    problem = MechanismSynthesisProblem(nelx, nely, penal, bc)
    title = cmd_helper.title_str(nelx, nely, volfrac, rmin, penal)
    gui = GUI(problem, title)
    filter = [SensitivityBasedFilter, DensityBasedFilter][ft](nelx, nely, rmin)
    solver = MechanismSynthesisSolver(problem, volfrac, filter, gui)
    cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft, solver=solver)


if __name__ == "__main__":
    main()
