"""Boundary conditions for topology optimization (forces and fixed nodes)."""

# Import standard library
import abc

# Import modules
import numpy

# Import TopOpt modules
from .utils import xy_to_id


class BoundaryConditions(abc.ABC):
    """
    Abstract class for boundary conditions to a topology optimization problem.

    Functionalty for geting fixed nodes, forces, and passive elements.

    Attributes
    ----------
    nelx: int
        The number of elements in the x direction.
    nely: int
        The number of elements in the y direction.

    """

    def __init__(self, nelx: int, nely: int):
        """
        Create the boundary conditions with the size of the grid.

        Parameters
        ----------
        nelx:
            The number of elements in the x direction.
        nely:
            The number of elements in the y direction.

        """
        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)

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
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes of the problem."""
        pass

    @abc.abstractproperty
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector for the problem."""
        pass

    @property
    def passive_elements(self):
        """:obj:`numpy.ndarray`: Passive elements to be set to zero density."""
        return numpy.array([])

    @property
    def active_elements(self):
        """:obj:`numpy.ndarray`: Active elements to be set to full density."""
        return numpy.array([])


class MBBBeamBoundaryConditions(BoundaryConditions):
    """Boundary conditions for the Messerschmitt–Bölkow–Blohm (MBB) beam."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes in the bottom corners."""
        dofs = numpy.arange(self.ndof)
        fixed = numpy.union1d(dofs[0:2 * (self.nely + 1):2], numpy.array(
            [2 * (self.nelx + 1) * (self.nely + 1) - 1]))
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector in the top center."""
        f = numpy.zeros((self.ndof, 1))
        f[1, 0] = -1
        return f


class CantileverBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a cantilever."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes on the left."""
        ys = numpy.arange(self.nely + 1)
        lefty_to_id = numpy.vectorize(
            lambda y: xy_to_id(0, y, self.nelx, self.nely))
        ids = lefty_to_id(ys)
        fixed = numpy.union1d(2 * ids, 2 * ids + 1)  # Fix both x and y dof
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector in the middle right."""
        f = numpy.zeros((self.ndof, 1))
        dof_index = 2 * xy_to_id(
            self.nelx, self.nely // 2, self.nelx, self.nely) + 1
        f[dof_index, 0] = -1
        return f


class LBracketBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a L-shaped bracket."""

    def __init__(self, nelx: int, nely: int, minx: int, maxy: int):
        """
        Create L-bracket boundary conditions with the size of the grid.

        Parameters
        ----------
        nelx:
            The number of elements in the x direction.
        nely:
            The number of elements in the y direction.
        minx:
            The minimum x coordinate of the passive upper-right block.
        maxy:
            The maximum y coordinate of the passive upper-right block.

        Raises
        ------
            ValueError: `minx` and `maxy` must be indices in the grid.

        """
        BoundaryConditions.__init__(self, nelx, nely)
        if(minx < 0 or minx >= nelx):
            raise ValueError(
                "minx must be a valid index into the grid [0, nelx)!")
        if(maxy < 0 or maxy >= nely):
            raise ValueError(
                "maxy must be a valid index into the grid [0, nely)!")
        self.passive_min_x = minx
        self.passive_min_y = 0
        self.passive_max_x = nelx - 1
        self.passive_max_y = maxy

    def __repr__(self) -> str:
        """Construct a representation of the boundary conditions."""
        return "{}(nelx={:d}, nely={:d}, minx={:d}, maxy={:d})".format(
            self.__class__.__name__, self.nelx, self.nely, self.passive_min_x,
            self.passive_max_y)

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes in the top row."""
        x = numpy.arange(self.passive_min_x)
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        ids = topx_to_id(x)
        fixed = numpy.union1d(2 * ids, 2 * ids + 1)
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector in the middle right."""
        f = numpy.zeros((self.ndof, 1))
        fx = self.nelx
        # fy = (self.nely - self.passive_max_y) // 2 + self.passive_max_y
        for i in range(1, 2):
            fy = self.passive_max_y + 2 * i
            id = xy_to_id(fx, fy, self.nelx, self.nely)
            f[2 * id + 1, 0] = -1
        return f

    @property
    def passive_elements(self):
        """:obj:`numpy.ndarray`: Passive elements in the upper right corner."""
        X, Y = numpy.mgrid[self.passive_min_x:self.passive_max_x + 1,
                           self.passive_min_y:self.passive_max_y + 1]
        pairs = numpy.vstack([X.ravel(), Y.ravel()]).T
        passive_to_ids = numpy.vectorize(lambda xy: xy_to_id(
            *xy, nelx=self.nelx - 1, nely=self.nely - 1), signature="(m)->()")
        return passive_to_ids(pairs)


class IBeamBoundaryConditions(BoundaryConditions):
    """Boundary conditions for an I-shaped beam."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes in the bottom row."""
        x = numpy.arange(self.nelx + 1)
        botx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, self.nely, self.nelx, self.nely))
        ids = 2 * botx_to_id(x)
        fixed = numpy.union1d(ids, ids + 1)
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector on the top row."""
        x = numpy.arange(self.nelx + 1)
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        ids = 2 * topx_to_id(x) + 1
        f = numpy.zeros((self.ndof, 1))
        f[ids] = -1
        return f

    @property
    def passive_elements(self):
        """:obj:`numpy.ndarray`: Passive elements on the left and right."""
        X1, Y1 = numpy.mgrid[0:(self.nelx // 3),
                             (self.nely // 4):(3 * self.nely // 4)]
        X2, Y2 = numpy.mgrid[(2 * self.nelx // 3):self.nelx,
                             (self.nely // 4):(3 * self.nely // 4)]
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
        """:obj:`numpy.ndarray`: Passives on the left, middle, and right."""
        X1, Y1 = numpy.mgrid[0:(self.nelx // 5),
                             (self.nely // 4):(3 * self.nely // 4)]
        X2, Y2 = numpy.mgrid[(2 * self.nelx // 5):(3 * self.nelx // 5),
                             (self.nely // 4):(3 * self.nely // 4)]
        X3, Y3 = numpy.mgrid[(4 * self.nelx // 5):self.nelx,
                             (self.nely // 4):(3 * self.nely // 4)]
        X = numpy.append(numpy.append(X1.ravel(), X2.ravel()), X3.ravel())
        Y = numpy.append(numpy.append(Y1.ravel(), Y2.ravel()), Y3.ravel())
        pairs = numpy.vstack([X.ravel(), Y.ravel()]).T
        passive_to_ids = numpy.vectorize(lambda xy: xy_to_id(
            *xy, nelx=self.nelx - 1, nely=self.nely - 1), signature="(m)->()")
        return passive_to_ids(pairs)
