#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explore affects of parameters using complicance problem and MBB Beam."""
from __future__ import division

import context  # noqa

from topopt import cli


def main():
    """Explore affects of parameters using complicance problem and MBB Beam."""
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = cli.parse_args()
    cli.main(nelx, nely, volfrac, penalty, rmin, ft)
    # Vary the filter radius
    for scaled_factor in [0.25, 2]:
        cli.main(nelx, nely, volfrac, penalty, scaled_factor * rmin, ft)
    # Vary the penalization power
    for scaled_factor in [0.5, 4]:
        cli.main(nelx, nely, volfrac, scaled_factor * penalty, rmin, ft)
    # Vary the discreization
    for scale_factor in [0.5, 2]:
        cli.main(int(scale_factor * nelx), int(scale_factor * nely),
                 volfrac, penalty, rmin, ft)


if __name__ == "__main__":
    main()
