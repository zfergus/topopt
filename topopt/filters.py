"""Filter the solution to topology optimization."""
from __future__ import division

# Import standard library
import abc

# Import modules
import numpy
import scipy


class Filter(abc.ABC):
    """Filter solutions to topology optimization to avoid checker boarding."""

    def __init__(self, nelx: int, nely: int, rmin: float):
        """
        Create a filter to filter solutions.

        Build (and assemble) the index+data vectors for the coo matrix format.

        Parameters
        ----------
        nelx:
            The number of elements in the x direction.
        nely:
            The number of elements in the y direction.
        rmin:
            The filter radius.

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

    def __str__(self) -> str:
        """Create a string representation of the filter."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Create a formated representation of the filter."""
        return str(self)

    def __repr__(self) -> str:
        """Create a formated representation of the filter."""
        return self._repr_string

    @abc.abstractmethod
    def filter_variables(self, x: numpy.ndarray, xPhys: numpy.ndarray) -> None:
        """
        Filter the variable of the solution to produce xPhys.

        Parameters
        ----------
        x:
            The raw density values.
        xPhys:
            The filtered density values to be computed

        """
        pass

    @abc.abstractmethod
    def filter_objective_sensitivities(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> None:
        """
        Filter derivative of the objective.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dobj:
            The filtered objective sensitivities to be computed.

        """
        pass

    @abc.abstractmethod
    def filter_volume_sensitivities(
            self, xPhys: numpy.ndarray, dv: numpy.ndarray) -> None:
        """
        Filter derivative of the volume.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dv:
            The filtered volume sensitivities to be computed.

        """
        pass


class SensitivityBasedFilter(Filter):
    """Sensitivity based filter of solutions."""

    def filter_variables(self, x: numpy.ndarray, xPhys: numpy.ndarray) -> None:
        """
        Filter the variable of the solution to produce xPhys.

        Parameters
        ----------
        x:
            The raw density values.
        xPhys:
            The filtered density values to be computed

        """
        xPhys[:] = x

    def filter_objective_sensitivities(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> None:
        """
        Filter derivative of the objective.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dobj:
            The filtered objective sensitivities to be computed.

        """
        dobj[:] = (numpy.asarray(
            (self.H * (xPhys * dobj))[numpy.newaxis].T / self.Hs)[:, 0] /
            numpy.maximum(0.001, xPhys))

    def filter_volume_sensitivities(
            self, xPhys: numpy.ndarray, dv: numpy.ndarray) -> None:
        """
        Filter derivative of the volume.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dv:
            The filtered volume sensitivities to be computed.

        """
        return


class DensityBasedFilter(Filter):
    """Density based filter of solutions."""

    def filter_variables(self, x: numpy.ndarray, xPhys: numpy.ndarray) -> None:
        """
        Filter the variable of the solution to produce xPhys.

        Parameters
        ----------
        x:
            The raw density values.
        xPhys:
            The filtered density values to be computed

        """
        xPhys[:] = numpy.asarray(self.H * x[numpy.newaxis].T / self.Hs)[:, 0]

    def filter_objective_sensitivities(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> None:
        """
        Filter derivative of the objective.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dobj:
            The filtered objective sensitivities to be computed.

        """
        dobj[:] = numpy.asarray(
            self.H * (dobj[numpy.newaxis].T / self.Hs))[:, 0]

    def filter_volume_sensitivities(
            self, xPhys: numpy.ndarray, dv: numpy.ndarray) -> None:
        """
        Filter derivative of the volume.

        Parameters
        ----------
        xPhys:
            The filtered density values.
        dv:
            The filtered volume sensitivities to be computed.

        """
        dv[:] = numpy.asarray(self.H * (dv[numpy.newaxis].T / self.Hs))[:, 0]
