# -*- coding: utf-8 -*-
"""TopOpt: A Topology Optimization Libarary."""

from . import boundary_conditions
from . import filters
from . import problems
from . import solvers

__all__ = ["boundary_conditions", "filters", "problems", "solvers"]

__version__ = "0.0.1-alpha.0"
__author__ = "Zachary Ferguson"
__email__ = "zfergus@nyu.edu"
