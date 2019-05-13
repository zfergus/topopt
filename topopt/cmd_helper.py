# -*- coding: utf-8 -*-
"""Command-line utility."""
from __future__ import division, print_function
import sys
import textwrap

import numpy

import topopt.guis
import topopt.boundary_conditions
import topopt.problems
import topopt.filters
import topopt.solvers

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass


def parse_sys_args(nelx=180, nely=60, volfrac=0.4, penal=3.0, rmin=5.4, ft=1):
    """Parse the system args with the given values as defaults."""
    nelx, nely, volfrac, penal, rmin, ft = (sys.argv[1:] + [
        nelx, nely, volfrac, penal, rmin, ft][len(sys.argv) - 1:])[:6]
    nelx, nely, ft = map(int, [nelx, nely, ft])
    volfrac, penal, rmin = map(float, [volfrac, penal, rmin])
    return (nelx, nely, volfrac, penal, rmin, ft)


def title_str(nelx, nely, volfrac, rmin, penal):
    """Create a title string for the problem."""
    return textwrap.dedent(f"""\
        ndes: {nelx:d} x {nely:d}
        volfrac: {volfrac:g}, penal: {penal:g}, rmin: {rmin:g}""")


def main(nelx, nely, volfrac, penal, rmin, ft, gui=None, bc=None, problem=None,
         filter=None, solver=None, description=None):
    """Run the main application of the command-line tools."""
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * numpy.ones(nely * nelx, dtype=float)

    # Initialize optional arguments
    if solver is None:
        if problem is None:
            if bc is None:
                bc = topopt.boundary_conditions.MBBBeamBoundaryConditions(
                    nelx, nely)
            problem = topopt.problems.ComplianceProblem(
                nelx, nely, penal, bc)
        gui = gui if gui else topopt.guis.GUI(
            problem, title_str(nelx, nely, volfrac, rmin, penal))
        filter = (filter if filter else [
            topopt.filters.SensitivityBasedFilter,
            topopt.filters.DensityBasedFilter][ft](nelx, nely, rmin))
        solver = topopt.solvers.TopOptSolver(problem, volfrac, filter, gui)

    print(textwrap.dedent(f"""\
        {solver:s}
        dims: {nelx:d} x {nely:d}
        volfrac: {volfrac:g}, penal: {penal:g}, rmin: {rmin:g}
        Filter method: {solver.filter:s}"""))

    # Solve the problem
    x_opt = solver.optimize(x)
    x_opt = solver.filter_variables(x_opt)
    if gui is not None:
        gui.update(x_opt)

    # Make sure the plot stays and that the shell remains
    input("Press any key...")
