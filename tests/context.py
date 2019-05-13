"""Add topopt to the context of any module that imports this module."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import topopt  # noqa
