#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multiple loads with stresses."""
from __future__ import division

import context  # noqa

from topopt.problems import VonMisesStressProblem
from topopt.guis import StressGUI

from topopt import cli

from multiple_loads_mma import MultipleLoadsBoundaryConditions


def main():
    """Multiple loads with stresses."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args(
        nelx=120, volfrac=0.2, rmin=1.5)
    bc = MultipleLoadsBoundaryConditions(nelx, nely)
    problem = VonMisesStressProblem(nelx, nely, penalty, bc)
    title = cli.title_str(nelx, nely, volfrac, rmin, penalty)
    gui = StressGUI(problem, title)
    cli.main(nelx, nely, volfrac, penalty, rmin, ft, bc=bc,
             problem=problem, gui=gui)


if __name__ == "__main__":
    main()
