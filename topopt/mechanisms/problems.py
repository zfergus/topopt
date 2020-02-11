"""Compliant mechanism synthesis problems using topology optimization."""

import numpy
import scipy.sparse

from ..problems import ElasticityProblem
from .boundary_conditions import MechanismSynthesisBoundaryConditions
from ..utils import deleterowcol


class MechanismSynthesisProblem(ElasticityProblem):
    r"""
    Topology optimization problem to generate compliant mechanisms.

    :math:`\begin{aligned}
    \max_{\boldsymbol{\rho}} \quad &
    \{u_{\text{out}}=\mathbf{l}^{T} \mathbf{u}\}\\
    \textrm{subject to}: \quad & \mathbf{K}\mathbf{u} =
    \mathbf{f}_\text{in}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1,
    \quad e=1, \dots, N.\\
    \end{aligned}`

    where :math:`\mathbf{l}` is a vector with the value 1 at the degree(s) of
    freedom corresponding to the output point and with zeros at all other
    places.

    Attributes
    ----------
    spring_stiffnesses: numpy.ndarray
        The spring stiffnesses of the
        actuator and output displacement.
    Emin: float
        The minimum stiffness of elements.
    Emax: float
        The maximum stiffness of elements.

    """

    @staticmethod
    def lk(E: float = 1.0, nu: float = 0.3) -> numpy.ndarray:
        """
        Build the element stiffness matrix.

        Parameters
        ----------
        E:
            Young's modulus of the material.
        nu:
            Poisson's ratio of the material.

        Returns
        -------
            The element stiffness matrix for the material.

        """
        return ElasticityProblem.lk(1e0, nu)

    def __init__(
            self, bc: MechanismSynthesisBoundaryConditions, penalty: float):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        nelx:
            Number of elements in the x direction.
        nely:
            Number of elements in the x direction.
        penalty:
            Penalty value used to penalize fractional densities in SIMP.
        bc:
            Boundary conditions of the problem.

        """
        super().__init__(bc, penalty)
        self.Emin = 1e-6  # Minimum stiffness of elements
        self.Emax = 1e2  # Maximum stiffness of elements
        # Spring stiffnesses for the actuator and output displacement
        self.spring_stiffnesses = numpy.full(
            numpy.nonzero(self.f)[0].shape, 10.0)

    def build_K(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo.coo_matrix:
        """
        Build the stiffness matrix for the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.
        remove_constrained:
            Should the constrained nodes be removed?

        Returns
        -------
            The stiffness matrix for the mesh.

        """
        # Build the stiffness matrix using inheritance
        K = super().build_K(xPhys, remove_constrained=False).tocsc()
        # Add spring stiffnesses
        spring_ids = numpy.nonzero(self.f)[0]
        K[spring_ids, spring_ids] += self.spring_stiffnesses
        # K = (K.T + K) / 2.  # Make sure the stiffness matrix is symmetric
        # Remove constrained dofs from matrix and convert to coo
        if remove_constrained:
            K = deleterowcol(K, self.fixed, self.fixed)
        return K.tocoo()

    def compute_objective(self, xPhys: numpy.ndarray, dobj: numpy.ndarray
                          ) -> float:
        r"""
        Compute the objective and gradient of the mechanism synthesis problem.

        The objective is :math:`u_{\text{out}}=\mathbf{l}^{T} \mathbf{u}`
        where :math:`\mathbf{l}` is a vector with the value 1 at
        the degree(s) of freedom corresponding to the output point and with
        zeros at all other places. The gradient of the objective is

        :math:`\begin{align}
        u_\text{out} &= \mathbf{l}^T\mathbf{u} = \mathbf{l}^T\mathbf{u} +
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial u_\text{out}}{\partial \rho_e} &=
        (\mathbf{K}\boldsymbol{\lambda} + \mathbf{l})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        \end{align}`

        where :math:`\mathbf{K}\boldsymbol{\lambda} = -\mathbf{l}`.

        Parameters
        ----------
        xPhys:
            The density design variables.
        dobj:
            The gradient of the objective to compute.

        Returns
        -------
            The objective of the compliant mechanism synthesis problem.

        """
        # Setup and solve FE problem
        self.update_displacements(xPhys)

        u = self.u[:, 0][self.edofMat].reshape(-1, 8)  # Displacement
        λ = self.u[:, 1][self.edofMat].reshape(-1, 8)  # Fixed vector (Kλ = -l)
        obj = self.f[:, 1].T @ self.u[:, 0]
        self.obje[:] = (λ @ self.KE * u).sum(1)
        self.compute_young_moduli(xPhys, dobj)  # Stores the derivative in dobj
        dobj *= -self.obje
        return obj
