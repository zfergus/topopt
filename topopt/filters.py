"""Filter the solution to topology optimization."""
from __future__ import division

import numpy
import scipy


class Filter(object):
    """Filter solutions to topology optimization to avoid checker boarding."""

    def __init__(self, nelx, nely, rmin):
        """
        Create a filter to filter solutions.

        Build (and assemble) the index+data vectors for the coo matrix format.
        """
        self._repr_string = "{}(nelx={:d}, nely={:d}, rmin={:g})".format(
            self.__class__.__name__, nelx, nely, rmin)
        nfilter = int(nelx * nely * ((2 * (numpy.ceil(rmin) - 1) + 1)**2))
        iH = numpy.zeros(nfilter)
        jH = numpy.zeros(nfilter)
        sH = numpy.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(numpy.maximum(i - (numpy.ceil(rmin) - 1), 0))
                kk2 = int(numpy.minimum(i + numpy.ceil(rmin), nelx))
                ll1 = int(numpy.maximum(j - (numpy.ceil(rmin) - 1), 0))
                ll2 = int(numpy.minimum(j + numpy.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - numpy.sqrt(
                            ((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = numpy.maximum(0.0, fac)
                        cc = cc + 1
        # Finalize assembly and convert to csc format
        self.H = scipy.sparse.coo_matrix(
            (sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        self.Hs = self.H.sum(1)

    def __str__(self):
        """Create a string representation of the filter."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the filter."""
        return str(self)

    def __repr__(self):
        """Create a formated representation of the filter."""
        return self._repr_string

    def filter_variables(self, x, xPhys):
        """Filter the variable of the solution to produce xPhys."""
        raise NotImplementedError(
            "Subclasses must override filter_variables()!")

    def filter_objective_sensitivities(self, xPhys, dobj):
        """Filter derivative of the objective."""
        raise NotImplementedError(
            "Subclasses must override filter_objective_sensitivities()!")

    def filter_volume_sensitivities(self, _xPhys, dv):
        """Filter derivative of the volume."""
        raise NotImplementedError(
            "Subclasses must override filter_volume_sensitivities()!")


class SensitivityBasedFilter(Filter):
    """Sensitivity based filter of solutions."""

    def filter_variables(self, x, xPhys):
        """Filter the variable of the solution to produce xPhys."""
        xPhys[:] = x

    def filter_objective_sensitivities(self, xPhys, dobj):
        """Filter derivative of the objective."""
        dobj[:] = (numpy.asarray(
            (self.H * (xPhys * dobj))[numpy.newaxis].T / self.Hs)[:, 0] /
                numpy.maximum(0.001, xPhys))

    def filter_volume_sensitivities(self, _xPhys, dv):
        """Filter derivative of the volume."""
        pass


class DensityBasedFilter(Filter):
    """Density based filter of solutions."""

    def filter_variables(self, x, xPhys):
        """Filter the variable of the solution to produce xPhys."""
        xPhys[:] = numpy.asarray(self.H * x[numpy.newaxis].T / self.Hs)[:, 0]

    def filter_objective_sensitivities(self, xPhys, dobj):
        """Filter derivative of the objective."""
        dobj[:] = numpy.asarray(
            self.H * (dobj[numpy.newaxis].T / self.Hs))[:, 0]

    def filter_volume_sensitivities(self, _xPhys, dv):
        """Filter derivative of the volume."""
        dv[:] = numpy.asarray(self.H * (dv[numpy.newaxis].T / self.Hs))[:, 0]
