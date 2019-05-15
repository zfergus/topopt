"""Boundary conditions for mechanism synthesis (forces and fixed nodes)."""

# Import modules
import numpy

# Import TopOpt modules
from ..boundary_conditions import TopOptBoundaryConditions
from ..utils import xy_to_id


class DisplacementInverterBoundaryConditions(TopOptBoundaryConditions):
    """Boundary conditions for a displacment inverter compliant mechanism."""

    @property
    def fixed_nodes(self):
        """Fixed nodes in the bottom left and top left corners."""
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
        """Middle left input force and middle right opposite output force."""
        f = numpy.zeros((2 * (self.nelx + 1) * (self.nely + 1), 2))
        # f[2 * xy_to_id(0, 0, self.nelx, self.nely), 0] = 1
        f[2 * xy_to_id(0, self.nely // 2, self.nelx, self.nely), 0] = 1
        # f[2 * xy_to_id(self.nelx, 0, self.nelx, self.nely), 1] = -1
        f[2 * xy_to_id(
            self.nelx, self.nely // 2, self.nelx, self.nely), 1] = -1
        return f
