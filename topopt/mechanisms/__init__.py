# -*- coding: utf-8 -*-
"""Mechanisms: A Topology Optimization Libarary for Compliant Mechanisms Synthesis."""

from . import boundary_conditions
from . import problems
from . import solvers

__all__ = ["boundary_conditions", "problems", "solvers"]

__version__ = "0.0.1-alpha.0"
__author__ = "Zachary Ferguson"
__email__ = "zfergus@nyu.edu"
