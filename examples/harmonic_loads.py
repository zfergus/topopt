#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a compliance mechanism that inverts displacments."""

import textwrap

import numpy
import matplotlib.pyplot as plt

import context  # noqa

from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import HarmonicLoadsProblem
from topopt.solvers import TopOptSolver
from topopt.filters import SensitivityBasedFilter, DensityBasedFilter
from topopt.guis import GUI

from topopt import cli


def main():
    """Run the example by constructing the TopOpt objects."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=300, nely=100, volfrac=0.3, penalty=3, rmin=1.4)
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = HarmonicLoadsProblem(bc, penalty)
    title = cli.title_str(nelx, nely, volfrac, rmin, penalty)
    gui = GUI(problem, title)
    filter = [SensitivityBasedFilter, DensityBasedFilter][ft](nelx, nely, rmin)
    solver = TopOptSolver(problem, volfrac, filter, gui)

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * numpy.ones(nely * nelx, dtype=float)

    print(textwrap.dedent(f"""\
        {solver:s}
        dims: {nelx:d} x {nely:d}
        volfrac: {volfrac:g}, penalty: {penalty:g}, rmin: {rmin:g}
        Filter method: {solver.filter:s}"""))

    frequency_response = []
    for angular_frequency in numpy.linspace(0.0, 0.2, 11):
        # Solve the problem
        print("angular_frequency={}".format(angular_frequency))
        problem.angular_frequency = angular_frequency
        x = solver.optimize(x)
        x = solver.filter_variables(x)
        if gui is not None:
            gui.update(x)
        dobj = numpy.empty(x.shape)
        frequency_response.append(
            [angular_frequency, problem.compute_objective(x, dobj)])

    frequency_response = numpy.array(frequency_response)
    fig, ax = plt.subplots()
    ax.plot(frequency_response[:, 0], frequency_response[:, 1])
    ax.set_title("Frequency Response")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\mathbf{u}^T\mathbf{f}$")

    # Make sure the plot stays and that the shell remains
    input("Press any key...")


if __name__ == "__main__":
    main()
