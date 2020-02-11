# -*- coding: utf-8 -*-
"""TopOpt: A Topology Optimization Libarary."""

from . import boundary_conditions
from . import filters
from . import problems
from . import solvers
from . import guis

__all__ = ["boundary_conditions", "filters", "problems", "solvers", "guis"]

__author__ = "Zachary Ferguson"
__copyright__ = "Copyright 2020, Zachary Ferguson"
__license__ = "MIT"
__version__ = "0.0.1-alpha.1"
__maintainer__ = "Zachary Ferguson"
__email__ = "zfergus@nyu.edu"
