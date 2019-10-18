"""Boundary conditions for mechanism synthesis (forces and fixed nodes)."""

# Import standard library
import functools

# Import modules
import numpy

# Import TopOpt modules
from ..boundary_conditions import BoundaryConditions
from ..utils import xy_to_id


class MechanismSynthesisBoundaryConditions(BoundaryConditions):
    """Boundary conditions for compliant mechanism synthesis."""

    @property
    def output_displacement_mask(self):
        """:obj:`numpy.ndarray`:Mask of the output displacement."""
        pass


class DisplacementInverterBoundaryConditions(
        MechanismSynthesisBoundaryConditions):
    """Boundary conditions for a displacment inverter compliant mechanism."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`:Fixed bottom and top left corner nodes."""
        # topx_to_id = numpy.vectorize(
        #     lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        # topx = 2 * topx_to_id(numpy.arange(self.nelx + 1)) + 1
        id1 = 2 * xy_to_id(0, 0, self.nelx, self.nely)
        id2 = 2 * xy_to_id(0, self.nely, self.nelx, self.nely)
        # fixed = numpy.union1d(topx, numpy.array([id2, id2 + 1]))
        fixed = numpy.array([id1, id1 + 1, id2, id2 + 1])
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`:Middle left input force."""
        f = numpy.zeros((self.ndof, 2))
        # f[2 * xy_to_id(0, 0, self.nelx, self.nely), 0] = 1
        f[2 * xy_to_id(0, self.nely // 2, self.nelx, self.nely), 0] = 1
        # f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        f[2 * xy_to_id(
            self.nelx, self.nely // 2, self.nelx, self.nely), 1] = -1
        return f

    @property
    def output_displacement_mask(self):
        """:obj:`numpy.ndarray`:Middle right output displacement mask."""
        l = numpy.zeros(self.ndof, dtype=bool)  # noqa
        l[2 * xy_to_id(self.nelx, self.nely // 2,
                       self.nelx, self.nely), 0] = True
        return l


class GripperBoundaryConditions(MechanismSynthesisBoundaryConditions):
    """Boundary conditions for a gripping mechanism."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`:Fixed bottom and top left corner nodes."""
        y = numpy.arange((self.nely + 1) // 20)
        lefty_to_id = numpy.vectorize(
            lambda y: xy_to_id(0, y, self.nelx, self.nely))
        ids1 = 2 * lefty_to_id(y)

        x = numpy.arange((self.nelx + 1) // 20)
        botx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, self.nely, self.nelx, self.nely))
        ids2 = 2 * botx_to_id(x)
        fixed = functools.reduce(
            numpy.union1d, [ids1, ids1 + 1, ids2, ids2 + 1])
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`:Middle left input force."""
        f = numpy.zeros((self.ndof, 2))
        # f[2 * xy_to_id(0, 0, self.nelx, self.nely), 0] = 1
        f[2 * xy_to_id(0, self.nely // 2, self.nelx, self.nely), 0] = 1
        # f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        f[2 * xy_to_id(
            self.nelx, 4 * self.nely // 10, self.nelx, self.nely) + 1, 1] = -1
        f[2 * xy_to_id(
            self.nelx, 6 * self.nely // 10, self.nelx, self.nely) + 1, 1] = 1
        return f


class CrossSensitivityExampleBoundaryConditions(
        MechanismSynthesisBoundaryConditions):
    """Boundary conditions from Figure 2.19 of *Topology Optimization*."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`:Fixed bottom and top left corner nodes."""
        y = numpy.arange((self.nely + 1) // 10)
        lefty_to_id = numpy.vectorize(
            lambda y: xy_to_id(0, y, self.nelx, self.nely))
        ids1 = 2 * lefty_to_id(y)

        x = numpy.arange((self.nelx + 1) // 10)
        botx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, self.nely, self.nelx, self.nely))
        ids2 = 2 * botx_to_id(x)
        fixed = functools.reduce(
            numpy.union1d, [ids1, ids1 + 1, ids2, ids2 + 1])
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`:Middle left input force."""
        f = numpy.zeros((self.ndof, 2))
        # f[2 * xy_to_id(0, 0, self.nelx, self.nely), 0] = 1
        f[2 * xy_to_id(0, self.nely // 2, self.nelx, self.nely), 0] = 1
        # f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        return f

    @property
    def active_elements(self):
        """:obj:`numpy.ndarray`:Active elements to be set to full density."""
        return numpy.array([
            xy_to_id(0, 0, self.nelx - 1, self.nely - 1),
            xy_to_id(0, 1, self.nelx - 1, self.nely - 1),
            xy_to_id(1, 1, self.nelx - 1, self.nely - 1),
            xy_to_id(1, 0, self.nelx - 1, self.nely - 1),
            xy_to_id(0, self.nely // 2, self.nelx - 1, self.nely - 1),
            xy_to_id(1, self.nely // 2, self.nelx - 1, self.nely - 1),
            xy_to_id(0, self.nely // 2 + 1, self.nelx - 1, self.nely - 1),
            xy_to_id(1, self.nely // 2 + 1, self.nelx - 1, self.nely - 1),
            xy_to_id(self.nely - 1, 0, self.nelx - 1, self.nely - 1)])

    @property
    def output_displacement_mask(self):
        """:obj:`numpy.ndarray`:Middle right output displacement mask."""
        l = numpy.zeros(self.ndof, dtype=bool)  # noqa
        l[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 0] = True
        return l
