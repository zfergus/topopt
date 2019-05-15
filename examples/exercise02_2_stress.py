#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from topopt.guis import StressGUI
from topopt.problems import VonMisesStressProblem

from topopt import cmd_helper
from exercise02_2 import Exercise02_2BoundaryConditions

if __name__ == "__main__":
    def run():
        # Default input parameters
        nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args(
            nelx=120, volfrac=0.2, rmin=1.2)
        bc = Exercise02_2BoundaryConditions(nelx, nely)
        problem = VonMisesStressProblem(nelx, nely, penal, bc)
        gui = StressGUI(problem, title="Stress on Exercise 02.2")
        cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft, bc=bc,
                        problem=problem, gui=gui)
    run()
