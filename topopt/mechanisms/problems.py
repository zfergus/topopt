"""Compliant mechanism synthesis problems using topology optimization."""

import numpy
import scipy.sparse

from ..problems import TopOptProblem
from ..utils import deleterowcol


class MechanismSynthesisProblem(TopOptProblem):
    r"""
    Topology optimization problem to generate compliant mechanisms.

    :math:`\begin{aligned}
    \max_{\boldsymbol{\rho}} \quad & u_{\text{out}}=\mathbf{l}^{T} \mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{K}\mathbf{u} =
    \mathbf{f}_\text{in}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where
    """

    @staticmethod
    def lk(E: float = 1.0, nu: float = 0.3):
        """
        Build the element stiffness matrix.

        Parameters:
            E: Young's modulus of the material.
            nu: Poisson's ratio of the material.

        Returns:
            The element stiffness matrix for the material.
        """
        return TopOptProblem.lk(1e0, nu)

    def __init__(self, nelx, nely, penal, bc):
        """Create the topology optimization problem."""
        TopOptProblem.__init__(self, nelx, nely, penal, bc)
        # Max and min stiffness
        self.Emin = 1e-6
        self.Emax = 1e2

    def build_K(self, xPhys, remove_constrained=True):
        """Build the stiffness matrix for the problem."""
        sK = ((self.KE.flatten()[numpy.newaxis]).T *
              self.penalized_densities(xPhys)).flatten(order='F')
        # Add spring stiffnesses
        spring_ids = numpy.nonzero(self.f)[0]
        sK = numpy.append(sK, numpy.array([1e0, 1e0]))
        iK = numpy.append(self.iK, spring_ids)
        jK = numpy.append(self.jK, spring_ids)
        # Build stiffness matrix
        K = scipy.sparse.coo_matrix(
            (sK, (iK, jK)), shape=(self.ndof, self.ndof))
        K = (K.T + K) / 2.
        # Remove constrained dofs from matrix and convert to coo
        if remove_constrained:
            K = deleterowcol(K.tocsc(), self.fixed, self.fixed).tocoo()
        return K

    def compute_objective(self, xPhys, dobj):
        """Compute objective and its gradient."""
        u = self.u[:, 0][self.edofMat].reshape(-1, 8)
        v = self.u[:, 1][self.edofMat].reshape(-1, 8)
        obj = self.u[:, 0].T.dot(self.f[:, 1])
        self.obje[:] = (v.dot(self.KE) * u).sum(1)
        dobj[:] = -self.diff_penalized_densities(xPhys) * self.obje
        return obj
