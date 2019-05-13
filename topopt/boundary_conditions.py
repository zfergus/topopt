"""Boundary conditions for topology optimization (forces and fixed nodes)."""
from __future__ import division

# Import standard library
import abc

# Import modules
import numpy

# Import TopOpt modules
from .utils import xy_to_id


class TopOptBoundaryConditions(abc.ABC):
    """
    Abstract class for boundary conditions to a topology optimization problem.

    Functionalty for geting fixed nodes, forces, and passive elements.
    """

    __all__ = ["fixed_nodes", "forces", "passive_elements", "active_elements"]

    def __init__(self, nelx, nely):
        """
        Create the boundary conditions with the size of the grid.

        Args:
            nelx (int): number of elements in the x direction
            nely (int): number of elements in the y direction
        """
        self.nelx, self.nely = nelx, nely

    def __str__(self) -> str:
        """Construct a string representation of the boundary conditions."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Construct a formated representation of the boundary conditions."""
        return str(self)

    def __repr__(self) -> str:
        """Construct a representation of the boundary conditions."""
        return "{}(nelx={:d}, nely={:d})".format(
            self.__class__.__name__, self.nelx, self.nely)

    @abc.abstractproperty
    def fixed_nodes(self) -> numpy.ndarray:
        """:obj:`numpy.ndarray`: Fixed nodes of the problem."""
        pass

    @abc.abstractproperty
    def forces(self) -> numpy.ndarray:
        """:obj:`numpy.ndarray`: Force vector for the problem."""
        pass

    @property
    def passive_elements(self) -> numpy.ndarray:
        """:obj:`numpy.ndarray`: Passive elements to be set to zero density."""
        return numpy.array([])

    @property
    def active_elements(self) -> numpy.ndarray:
        """:obj:`numpy.ndarray`: Active elements to be set to full density."""
        return numpy.array([])


class MBBBeamBoundaryConditions(TopOptBoundaryConditions):
    """
    Boundary conditions for the Messerschmitt–Bölkow–Blohm (MBB) beam.

    .. image:: https://bit.ly/2HlzZXL
       :alt: Messerschmitt–Bölkow–Blohm (MBB) beam
    """

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        dofs = numpy.arange(2 * (self.nelx + 1) * (self.nely + 1))
        fixed = numpy.union1d(dofs[0:2 * (self.nely + 1):2], numpy.array(
            [2 * (self.nelx + 1) * (self.nely + 1) - 1]))
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        f = numpy.zeros((ndof, 1))
        f[1, 0] = -1
        return f


class LBracketBoundaryConditions(TopOptBoundaryConditions):
    """
    Boundary conditions for a L-shaped bracket.

    .. image:: https://bit.ly/2JBDQCx
        :alt: L-bracket
    """

    def __init__(self, nelx, nely, minx, maxy):
        """Create the boundary conditions with the size of the grid."""
        TopOptBoundaryConditions.__init__(self, nelx, nely)
        (self.passive_min_x, self.passive_min_y, self.passive_max_x,
            self.passive_max_y) = [minx, 0, nelx - 1, maxy]

    def __repr__(self):
        """Construct a representation of the boundary conditions."""
        return "{}(nelx={:d}, nely={:d}, minx={:d}, maxy={:d})".format(
            self.__class__.__name__, self.nelx, self.nely, self.passive_min_x,
            self.passive_max_y)

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        x = numpy.arange(self.passive_min_x)
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        ids = topx_to_id(x)
        fixed = numpy.union1d(2 * ids, 2 * ids + 1)
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        f = numpy.zeros((ndof, 1))
        fx = self.nelx
        # fy = (self.nely - self.passive_max_y) // 2 + self.passive_max_y
        for i in range(1, 2):
            fy = self.passive_max_y + 2 * i
            id = xy_to_id(fx, fy, self.nelx, self.nely)
            f[2 * id + 1, 0] = -1
        return f

    @property
    def passive_elements(self):
        """Return a list of the passive elements to be set to density 0."""
        X, Y = numpy.mgrid[self.passive_min_x:self.passive_max_x + 1,
                           self.passive_min_y:self.passive_max_y + 1]
        pairs = numpy.vstack([X.ravel(), Y.ravel()]).T
        passive_to_ids = numpy.vectorize(lambda xy: xy_to_id(
            *xy, nelx=self.nelx - 1, nely=self.nely - 1), signature="(m)->()")
        return passive_to_ids(pairs)


class IBeamBoundaryConditions(TopOptBoundaryConditions):
    """Boundary conditions for an I-shaped beam."""

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        x = numpy.arange(self.nelx + 1)
        botx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, self.nely, self.nelx, self.nely))
        ids = 2 * botx_to_id(x)
        fixed = numpy.union1d(ids, ids + 1)
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        x = numpy.arange(self.nelx + 1)
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        ids = 2 * topx_to_id(x) + 1
        f = numpy.zeros((ndof, 1))
        f[ids] = -1
        return f

    @property
    def passive_elements(self):
        """Return a list of the passive elements to be set to density 0."""
        X1, Y1 = numpy.mgrid[
            0:self.nelx // 3, self.nely // 4:3 * self.nely // 4]
        X2, Y2 = numpy.mgrid[
            2 * self.nelx // 3:self.nelx, self.nely // 4:3 * self.nely // 4]
        X = numpy.append(X1.ravel(), X2.ravel())
        Y = numpy.append(Y1.ravel(), Y2.ravel())
        pairs = numpy.vstack([X.ravel(), Y.ravel()]).T
        passive_to_ids = numpy.vectorize(lambda xy: xy_to_id(
            *xy, nelx=self.nelx - 1, nely=self.nely - 1), signature="(m)->()")
        return passive_to_ids(pairs)


class IIBeamBoundaryConditions(IBeamBoundaryConditions):
    """Boundary conditions for an II-shaped beam."""

    @property
    def passive_elements(self):
        """Return a list of the passive elements to be set to density 0."""
        X1, Y1 = numpy.mgrid[
            0:self.nelx // 5, self.nely // 4:3 * self.nely // 4]
        X2, Y2 = numpy.mgrid[2 * self.nelx // 5: 3 * self.nelx // 5,
                             self.nely // 4:3 * self.nely // 4]
        X3, Y3 = numpy.mgrid[
            4 * self.nelx // 5:self.nelx, self.nely // 4:3 * self.nely // 4]
        X = numpy.append(numpy.append(X1.ravel(), X2.ravel()), X3.ravel())
        Y = numpy.append(numpy.append(Y1.ravel(), Y2.ravel()), Y3.ravel())
        pairs = numpy.vstack([X.ravel(), Y.ravel()]).T
        passive_to_ids = numpy.vectorize(lambda xy: xy_to_id(
            *xy, nelx=self.nelx - 1, nely=self.nely - 1), signature="(m)->()")
        return passive_to_ids(pairs)
