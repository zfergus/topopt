# -*- coding: utf-8 -*-
"""Command-line utility to run topology optimization."""

import textwrap
import argparse

import numpy

import topopt.guis
import topopt.boundary_conditions
import topopt.problems
import topopt.filters
import topopt.solvers


def create_parser(nelx: int = 180, nely: int = 60, volfrac: float = 0.4,
                  penalty: float = 3.0, rmin: float = 5.4, ft: int = 1
                  ) -> argparse.ArgumentParser:
    """
    Create an argument parser with the given values as defaults.

    Parameters
    ----------
    nelx:
        The default number of elements in the x direction.
    nely:
        The default number of elements in the y direction.
    volfrac:
        The default fraction of the total volume to use.
    penalty:
        The default penalty exponent value in SIMP.
    rmin:
        The default filter radius.
    ft:
        The default filter method to use.
            - ``0``: :obj:`topopt.filters.SensitivityBasedFilter`
            - ``1``: :obj:`topopt.filters.DensityBasedFilter`

    Returns
    -------
        Argument parser with given defaults.

    """
    parser = argparse.ArgumentParser(description="Topology Optimization")
    parser.add_argument("--nelx", type=int, default=nelx,
                        help="number of elements in the x direction")
    parser.add_argument("--nely", type=int, default=nely,
                        help="number of elements in the x direction")
    parser.add_argument("--volfrac", type=float, default=volfrac,
                        help="fraction of the total volume usable")
    parser.add_argument("--penalty", type=float, default=penalty,
                        help="penalty exponent value in SIMP")
    parser.add_argument("--rmin", "--filter-radius", type=float, dest="rmin",
                        default=rmin, help="filter radius")
    parser.add_argument(
        "--ft", "--filter-type", choices=[0, 1], dest="ft", default=ft,
        help="filter type (0: sensitivity based, 1: density based)")
    return parser


def parse_args(nelx: int = 180, nely: int = 60, volfrac: float = 0.4,
               penalty: float = 3.0, rmin: float = 5.4, ft: int = 1
               ) -> tuple:
    """
    Parse the system args with the given values as defaults.

    Parameters
    ----------
    nelx:
        The default number of elements in the x direction.
    nely:
        The default number of elements in the y direction.
    volfrac:
        The default fraction of the total volume to use.
    penalty:
        The default penalty exponent value in SIMP.
    rmin:
        The default filter radius.
    ft:
        The default filter method to use.
            - ``0``: :obj:`topopt.filters.SensitivityBasedFilter`
            - ``1``: :obj:`topopt.filters.DensityBasedFilter`

    Returns
    -------
        Parsed command-line arguments.

    """
    args = create_parser(nelx, nely, volfrac, penalty, rmin, ft).parse_args()
    return args.nelx, args.nely, args.volfrac, args.penalty, args.rmin, args.ft


def title_str(nelx: int, nely: int, volfrac: float, rmin: float,
              penalty: float) -> str:
    """
    Create a title string for the problem.

    Parameters
    ----------
    nelx:
        The number of elements in the x direction.
    nely:
        The number of elements in the y direction.
    volfrac:
        The fraction of the total volume to use.
    rmin:
        The filter radius.
    penalty:
        The penalty exponent value in SIMP.

    Returns
    -------
        Title string for the GUI.

    """
    return textwrap.dedent(f"""\
        dims: {nelx:d} x {nely:d}
        volfrac: {volfrac:g}, penalty: {penalty:g}, rmin: {rmin:g}""")


def main(nelx: int, nely: int, volfrac: float, penalty: float, rmin: float,
         ft: int, gui: topopt.guis.GUI = None,
         bc: topopt.boundary_conditions.BoundaryConditions = None,
         problem: topopt.problems.Problem = None,
         filter: topopt.filters.Filter = None,
         solver: topopt.solvers.TopOptSolver = None) -> None:
    """
    Run the main application of the command-line tools.

    Parameters
    ----------
    nelx:
        The number of elements in the x direction.
    nely:
        The number of elements in the y direction.
    volfrac:
        The fraction of the total volume to use.
    penalty:
        The penalty exponent value in SIMP.
    rmin:
        The filter radius.
    ft:
        The filter method to use.
            - ``0``: :obj:`topopt.filters.SensitivityBasedFilter`
            - ``1``: :obj:`topopt.filters.DensityBasedFilter`
    gui:
        The GUI to use.
    bc:
        The boundary conditions to use.
    problem:
         The problem to use.
    filter:
         The filter to use.
    solver:
         The solver to use.

    """
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * numpy.ones(nely * nelx, dtype=float)

    # Initialize optional arguments
    if solver is None:
        if problem is None:
            if bc is None:
                bc = topopt.boundary_conditions.MBBBeamBoundaryConditions(
                    nelx, nely)
            problem = topopt.problems.ComplianceProblem(bc, penalty)
        gui = gui if gui else topopt.guis.GUI(
            problem, title_str(nelx, nely, volfrac, rmin, penalty))
        filter = (filter if filter else [
            topopt.filters.SensitivityBasedFilter,
            topopt.filters.DensityBasedFilter][ft](nelx, nely, rmin))
        solver = topopt.solvers.TopOptSolver(problem, volfrac, filter, gui)

    print(textwrap.dedent(f"""\
        {solver:s}
        dims: {nelx:d} x {nely:d}
        volfrac: {volfrac:g}, penalty: {penalty:g}, rmin: {rmin:g}
        Filter method: {solver.filter:s}"""))

    # Solve the problem
    x_opt = solver.optimize(x)
    x_opt = solver.filter_variables(x_opt)
    if gui is not None:
        gui.update(x_opt)

    # Make sure the plot stays and that the shell remains
    input("Press any key...")
