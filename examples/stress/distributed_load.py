#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multiple simultaneous point loads with stress computation."""
from __future__ import division

import context  # noqa

from topopt.guis import StressGUI
from topopt.problems import VonMisesStressProblem

from topopt import cli

from distributed_load import DistributedLoadBoundaryConditions


def main():
    """Multiple simultaneous point loads with stress computation."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, rmin=1.2)
    bc = DistributedLoadBoundaryConditions(nelx, nely)
    problem = VonMisesStressProblem(nelx, nely, penalty, bc)
    gui = StressGUI(problem, title="Stresses of Distributed Load Example")
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, bc=bc,
             problem=problem, gui=gui)


if __name__ == "__main__":
    main()
