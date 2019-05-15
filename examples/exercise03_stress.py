#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy

from topopt.problems import VonMisesStressProblem
from topopt.guis import StressGUI
from topopt import cmd_helper

from exercise03_mma import Exercise03BoundaryConditions

if __name__ == "__main__":
    def run():
        # Default input parameters
        nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args(
            nelx=120, volfrac=0.2, rmin=1.5)
        bc = Exercise03BoundaryConditions(nelx, nely)
        problem = VonMisesStressProblem(nelx, nely, penal, bc)
        title = cmd_helper.title_str(nelx, nely, volfrac, rmin, penal)
        gui = StressGUI(problem, title)
        cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft, bc=bc,
                        problem=problem, gui=gui)
    run()
