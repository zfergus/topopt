#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from topopt import cmd_helper

if __name__ == "__main__":
    def run():
        # Default input parameters
        nelx, nely, volfrac, penal, rmin, ft = cmd_helper.parse_sys_args()
        # Vary the filter radius
        cmd_helper.main(nelx, nely, volfrac, penal, rmin / 4., ft)
        cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft)
        cmd_helper.main(nelx, nely, volfrac, penal, rmin * 2., ft)
        # Vary the penalization power
        cmd_helper.main(nelx, nely, volfrac, penal / 2., rmin, ft)
        cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft)
        cmd_helper.main(nelx, nely, volfrac, penal * 4., rmin, ft)
        # Vary the discreization
        cmd_helper.main(nelx // 2, nely // 2, volfrac, penal, rmin, ft)
        cmd_helper.main(nelx, nely, volfrac, penal, rmin, ft)
        cmd_helper.main(nelx * 2, nely * 2, volfrac, penal, rmin, ft)
    run()
