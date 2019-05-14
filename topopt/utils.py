"""Utilities for topology optimization."""
from __future__ import division

import numpy
import re


def xy_to_id(x: int, y: int, nelx: int, nely: int, order: str = "F") -> int:
    """
    Map from 2D indices of a node to the flattened 1D index.

    Parameters:
        x: The x-coordinate of the node's positions.
        y: The y-coordinate of the node's positions.
        nelx: The number of elements in the x-direction.
        nely: The number of elements in the y-direction.
        order: The order of indecies. "F" for Fortran/column-major order and
            "C" for C/row-major order.

    Returns:
        The index of the node in the flattened version.
    """
    if order == "C":
        return (y * (nelx + 1)) + x
    else:
        return (x * (nely + 1)) + y


def id_to_xy(index, nelx, nely, order="F"):
    """Map from a 1D index to 2D indices of a matrix."""
    if order == "C":
        y = index // (nelx + 1)
        x = index % (nelx + 1)
    else:
        x = index // (nely + 1)
        y = index % (nely + 1)
    return x, y


def deleterowcol(A, delrow, delcol):
    """
    Delete the specified rows and columns from csc sparse matrix A.

    Assumes that matrix is in symmetric csc form!
    """
    m = A.shape[0]
    keep = numpy.delete(numpy.arange(0, m), delrow)
    A = A[keep, :]
    keep = numpy.delete(numpy.arange(0, m), delcol)
    A = A[:, keep]
    return A


def squared_euclidean(x):
    """Compute the squared euclidean length of x."""
    return x.T.dot(x)


def load_colormap(filename):
    """Load a color map from a file."""
    from matplotlib import colors
    import scipy.io
    import os
    C = scipy.io.loadmat(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), filename))["C"]
    return colors.ListedColormap(C / 255.0)


def camel_case_to_spaces(str):
    """Add a space between camel-case words."""
    return re.sub('([a-z])([A-Z])', r'\1 \2', str)


def camel_case_split(str):
    """Split a camel-case string into a list of strings."""
    return camel_case_to_spaces(str).split()
