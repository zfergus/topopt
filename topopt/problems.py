"""Topology optimization problem to solve."""

import abc

import numpy
import scipy.sparse
import scipy.sparse.linalg
import cvxopt
import cvxopt.cholmod

from .boundary_conditions import BoundaryConditions
from .utils import deleterowcol


class Problem(abc.ABC):
    """
    Abstract topology optimization problem.

    Attributes
    ----------
    bc: BoundaryConditions
        The boundary conditions for the problem.
    penalty: float
        The SIMP penalty value.
    f: numpy.ndarray
        The right-hand side of the FEM equation (forces).
    u: numpy.ndarray
        The variables of the FEM equation.
    obje: numpy.ndarray
        The per element objective values.

    """

    def __init__(self, bc: BoundaryConditions, penalty: float):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        # Problem size
        self.nelx = bc.nelx
        self.nely = bc.nely
        self.nel = self.nelx * self.nely

        # Count degrees of fredom
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # SIMP penalty
        self.penalty = penalty

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # RHS and Solution vectors
        self.f = bc.forces
        self.u = numpy.zeros(self.f.shape)

        # Per element objective
        self.obje = numpy.zeros(self.nely * self.nelx)

    def __str__(self) -> str:
        """Create a string representation of the problem."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Create a formated representation of the problem."""
        return str(self)

    def __repr__(self) -> str:
        """Create a representation of the problem."""
        return "{}(bc={!r}, penalty={:g})".format(
            self.__class__.__name__, self.penalty, self.bc)

    def penalize_densities(self, x: numpy.ndarray, drho: numpy.ndarray = None
                           ) -> numpy.ndarray:
        """
        Compute the penalized densties (and optionally its derivative).

        Parameters
        ----------
        x:
            The density variables to penalize.
        drho:
            The derivative of the penealized densities to compute. Only set if
            drho is not None.

        Returns
        -------
        numpy.ndarray
            The penalized densities used for SIMP.

        """
        rho = x**self.penalty
        if drho is not None:
            assert(drho.shape == x.shape)
            drho[:] = rho
            valid = x != 0  # valid values for division
            drho[valid] *= self.penalty / x[valid]
        return rho

    @abc.abstractmethod
    def compute_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        """
        Compute objective and its gradient.

        Parameters
        ----------
        xPhys:
            The design variables.
        dobj:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        pass


class ElasticityProblem(Problem):
    """
    Abstract elasticity topology optimization problem.

    Attributes
    ----------
    Emin: float
        The Young's modulus use for the void regions.
    Emax: float
        The Young's modulus use for the solid regions.
    nu: float
        Poisson's ratio of the material.
    f: numpy.ndarray
        The right-hand side of the FEM equation (forces).
    u: numpy.ndarray
        The variables of the FEM equation (displacments).
    nloads: int
        The number of loads applied to the material.

    """

    @staticmethod
    def lk(E: float = 1.0, nu: float = 0.3) -> numpy.ndarray:
        """
        Build the element stiffness matrix.

        Parameters
        ----------
        E:
            The Young's modulus of the material.
        nu:
            The Poisson's ratio of the material.

        Returns
        -------
        numpy.ndarray
            The element stiffness matrix for the material.

        """
        k = numpy.array([
            0.5 - nu / 6., 0.125 + nu / 8., -0.25 - nu / 12.,
            -0.125 + 0.375 * nu, -0.25 + nu / 12., -0.125 - nu / 8., nu / 6.,
            0.125 - 0.375 * nu])
        KE = E / (1 - nu**2) * numpy.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return KE

    def __init__(self, bc: BoundaryConditions, penalty: float):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        super().__init__(bc, penalty)
        # Max and min stiffness
        self.Emin = 1e-9
        self.Emax = 1.0

        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.3
        self.build_indices()

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # Number of loads
        self.nloads = self.f.shape[1]

    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        self.KE = self.lk(E=self.Emax, nu=self.nu)
        self.edofMat = numpy.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = numpy.array([
                    2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2,
                    2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((8, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 8))).flatten()

    def compute_young_moduli(self, x: numpy.ndarray, dE: numpy.ndarray = None
                             ) -> numpy.ndarray:
        """
        Compute the Young's modulus of each element from the densties.

        Optionally compute the derivative of the Young's modulus.

        Parameters
        ----------
        x:
            The density variable of each element.
        dE:
            The derivative of Young's moduli to compute. Only set if dE is not
            None.

        Returns
        -------
        numpy.ndarray
            The elements' Young's modulus.

        """
        drho = None if dE is None else numpy.empty(x.shape)
        rho = self.penalize_densities(x, drho)
        if drho is not None and dE is not None:
            assert(dE.shape == x.shape)
            dE[:] = (self.Emax - self.Emin) * drho
        return (self.Emax - self.Emin) * rho + self.Emin

    def build_K(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
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
        scipy.sparse.coo_matrix
            The stiffness matrix for the mesh.

        """
        sK = ((self.KE.flatten()[numpy.newaxis]).T *
              self.compute_young_moduli(xPhys)).flatten(order='F')
        K = scipy.sparse.coo_matrix(
            (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            K = deleterowcol(K.tocsc(), self.fixed, self.fixed).tocoo()
        return K

    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the displacements given the densities.

        Compute the displacment, :math:`u`, using linear elastic finite
        element analysis (solving :math:`Ku = f` where :math:`K` is the
        stiffness matrix and :math:`f` is the force vector).

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.

        Returns
        -------
        numpy.ndarray
            The distplacements solve using linear elastic finite element
            analysis.

        """
        # Setup and solve FE problem
        K = self.build_K(xPhys)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(numpy.int), K.col.astype(numpy.int))
        # Solve system
        F = cvxopt.matrix(self.f[self.free, :])
        cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
        new_u = self.u.copy()
        new_u[self.free, :] = numpy.array(F)[:, :]
        return new_u

    def update_displacements(self, xPhys: numpy.ndarray) -> None:
        """
        Update the displacements of the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to compute the displacements.

        """
        self.u[:, :] = self.compute_displacements(xPhys)


class ComplianceProblem(ElasticityProblem):
    r"""
    Topology optimization problem to minimize compliance.

    :math:`\begin{aligned}
    \min_{\boldsymbol{\rho}} \quad & \mathbf{f}^T\mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{K}\mathbf{u} = \mathbf{f}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where :math:`\mathbf{f}` are the forces, :math:`\mathbf{u}` are the \
    displacements, :math:`\mathbf{K}` is the striffness matrix, and :math:`V`
    is the volume.
    """

    def compute_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # Setup and solve FE problem
        self.update_displacements(xPhys)

        obj = 0.0
        dobj[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            self.obje[:] = (ui @ self.KE * ui).sum(1)
            obj += (E * self.obje).sum()
            dobj[:] += -dE * self.obje
        dobj /= float(self.nloads)
        return obj / float(self.nloads)


class HarmonicLoadsProblem(ElasticityProblem):
    r"""
    Topology optimization problem to minimize dynamic compliance.

    Replaces standard forces with undamped forced vibrations.

    :math:`\begin{aligned}
    \min_{\boldsymbol{\rho}} \quad & \mathbf{f}^T\mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{S}\mathbf{u} = \mathbf{f}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where :math:`\mathbf{f}` is the amplitude of the load, :math:`\mathbf{u}`
    is the amplitude of vibration, and :math:`\mathbf{S}` is the system matrix
    (or "dynamic striffness" matrix) defined as

    :math:`\begin{aligned}
    \mathbf{S} = \mathbf{K} - \omega^2\mathbf{M}
    \end{aligned}`

    where :math:`\omega` is the angular frequency of the load, and
    :math:`\mathbf{M}` is the global mass matrix.
    """

    @staticmethod
    def lm(nel: int) -> numpy.ndarray:
        r"""
        Build the element mass matrix.

        :math:`M = \frac{1}{9 \times 4n}\begin{bmatrix}
        4 & 0 & 2 & 0 & 1 & 0 & 2 & 0 \\
        0 & 4 & 0 & 2 & 0 & 1 & 0 & 2 \\
        2 & 0 & 4 & 0 & 2 & 0 & 1 & 0 \\
        0 & 2 & 0 & 4 & 0 & 2 & 0 & 1 \\
        1 & 0 & 2 & 0 & 4 & 0 & 2 & 0 \\
        0 & 1 & 0 & 2 & 0 & 4 & 0 & 2 \\
        2 & 0 & 1 & 0 & 2 & 0 & 4 & 0 \\
        0 & 2 & 0 & 1 & 0 & 2 & 0 & 4
        \end{bmatrix}`

        Where :math:`n` is the total number of elements. The total mass is
        equal to unity.

        Parameters
        ----------
        nel:
            The total number of elements.

        Returns
        -------
        numpy.ndarray
            The element mass matrix for the material.

        """
        return numpy.array([
            [4, 0, 2, 0, 1, 0, 2, 0],
            [0, 4, 0, 2, 0, 1, 0, 2],
            [2, 0, 4, 0, 2, 0, 1, 0],
            [0, 2, 0, 4, 0, 2, 0, 1],
            [1, 0, 2, 0, 4, 0, 2, 0],
            [0, 1, 0, 2, 0, 4, 0, 2],
            [2, 0, 1, 0, 2, 0, 4, 0],
            [0, 2, 0, 1, 0, 2, 0, 4]], dtype=float) / (36 * nel)

    def __init__(self, bc: BoundaryConditions, penalty: float):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        super().__init__(bc, penalty)
        self.angular_frequency = 0e-2

    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        super().build_indices()
        self.ME = self.lm(self.nel)

    def build_M(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
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
        scipy.sparse.coo_matrix
            The stiffness matrix for the mesh.

        """
        # vals = numpy.tile(self.ME.flatten(), xPhys.size)
        vals = (self.ME.reshape(-1, 1) *
                self.penalize_densities(xPhys)).flatten(order='F')
        M = scipy.sparse.coo_matrix((vals, (self.iK, self.jK)),
                                    shape=(self.ndof, self.ndof))
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            M = deleterowcol(M.tocsc(), self.fixed, self.fixed).tocoo()
        return M

    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        r"""
        Compute the amplitude of vibration given the densities.

        Compute the amplitude of vibration, :math:`\mathbf{u}`, using linear
        elastic finite element analysis (solving
        :math:`\mathbf{S}\mathbf{u} = \mathbf{f}` where :math:`\mathbf{S} =
        \mathbf{K} - \omega^2\mathbf{M}` is the system matrix and
        :math:`\mathbf{f}` is the force vector).

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.

        Returns
        -------
        numpy.ndarray
            The displacements solve using linear elastic finite element
            analysis.

        """
        # Setup and solve FE problem
        K = self.build_K(xPhys)
        M = self.build_M(xPhys)
        S = (K - self.angular_frequency**2 * M).tocoo()
        cvxopt_S = cvxopt.spmatrix(
            S.data, S.row.astype(numpy.int), S.col.astype(numpy.int))
        # Solve system
        F = cvxopt.matrix(self.f[self.free, :])
        try:
            # F stores solution after solve
            cvxopt.cholmod.linsolve(cvxopt_S, F)
        except Exception:
            F = scipy.sparse.linalg.spsolve(S.tocsc(), self.f[self.free, :])
            F = F.reshape(-1, self.nloads)
        new_u = self.u.copy()
        new_u[self.free, :] = numpy.array(F)[:, :]
        return new_u

    def compute_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # Setup and solve FE problem
        self.update_displacements(xPhys)

        obj = 0.0
        dobj[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        drho = numpy.empty(xPhys.shape)
        penalty = self.penalty
        self.penalty = 2
        rho = self.penalize_densities(xPhys, drho)
        self.penalty = penalty
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            obje1 = (ui @ self.KE * ui).sum(1)
            obje2 = (ui @ (-self.angular_frequency**2 * self.ME) * ui).sum(1)
            self.obje[:] = obje1 + obje2
            obj += (E * obje1 + rho * obje2).sum()
            dobj[:] += -(dE * obje1 + drho * obje2)
        dobj /= float(self.nloads)
        return obj / float(self.nloads)


class VonMisesStressProblem(ElasticityProblem):
    """
    Topology optimization problem to minimize stress.

    Todo:
        * Currently this problem minimizes compliance and computes stress on
          the side. This needs to be replaced to match the promise of
          minimizing stress.
    """

    @staticmethod
    def B(side: float) -> numpy.ndarray:
        r"""
        Construct a strain-displacement matrix for a 2D regular grid.

        :math:`B = \frac{1}{2s}\begin{bmatrix}
        1 &  0 & -1 &  0 & -1 &  0 &  1 &  0 \\
        0 &  1 &  0 &  1 &  0 & -1 &  0 & -1 \\
        1 &  1 &  1 & -1 & -1 & -1 & -1 &  1
        \end{bmatrix}`

        where :math:`s` is the side length of the square elements.

        Todo:
            * Check that this is not -B

        Parameters
        ----------
        side:
            The side length of the square elements.

        Returns
        -------
        numpy.ndarray
            The strain-displacement matrix for a 2D regular grid.

        """
        n = -0.5 / side
        p = 0.5 / side
        return numpy.array([[p, 0, n, 0, n, 0, p, 0],
                            [0, p, 0, p, 0, n, 0, n],
                            [p, p, p, n, n, n, n, p]])

    @staticmethod
    def E(nu):
        r"""
        Construct a constitutive matrix for a 2D regular grid.

        :math:`E = \frac{1}{1 - \nu^2}\begin{bmatrix}
        1 & \nu & 0 \\
        \nu & 1 & 0 \\
        0 & 0 & \frac{1 - \nu}{2}
        \end{bmatrix}`

        Parameters
        ----------
        nu:
            The Poisson's ratio of the material.

        Returns
        -------
        numpy.ndarray
            The constitutive matrix for a 2D regular grid.

        """
        return numpy.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1 - nu) / 2.]]) / (1 - nu**2)

    def __init__(self, nelx, nely, penalty, bc, side=1):
        super().__init__(bc, penalty)
        self.EB = self.E(self.nu) @ self.B(side)
        self.du = numpy.zeros((self.ndof, self.nel * self.nloads))
        self.stress = numpy.zeros(self.nel)
        self.dstress = numpy.zeros(self.nel)

    def build_dK0(self, drho_xi, i, remove_constrained=True):
        sK = ((self.KE.flatten()[numpy.newaxis]).T * drho_xi).flatten(
            order='F')
        iK = self.iK[64 * i: 64 * i + 64]
        jK = self.jK[64 * i: 64 * i + 64]
        dK = scipy.sparse.coo_matrix(
            (sK, (iK, jK)), shape=(self.ndof, self.ndof))
        # Remove constrained dofs from matrix and convert to coo
        if remove_constrained:
            dK = deleterowcol(dK.tocsc(), self.fixed, self.fixed).tocoo()
        return dK

    def build_dK(self, xPhys, remove_constrained=True):
        drho = numpy.empty(xPhys.shape)
        self.compute_young_moduli(xPhys, drho)
        blocks = [self.build_dK0(drho[i], i, remove_constrained)
                  for i in range(drho.shape[0])]
        dK = scipy.sparse.block_diag(blocks, format="coo")
        return dK

    @staticmethod
    def sigma_pow(s11: numpy.ndarray, s22: numpy.ndarray, s12: numpy.ndarray,
                  p: float) -> numpy.ndarray:
        r"""
        Compute the von Mises stress raised to the :math:`p^{\text{th}}` power.

        :math:`\sigma^p = \left(\sqrt{\sigma_{11}^2 - \sigma_{11}\sigma_{22} +
        \sigma_{22}^2 + 3\sigma_{12}^2}\right)^p`

        Todo:
            * Properly document what the sigma variables represent.
            * Rename the sigma variables to something more readable.

        Parameters
        ----------
        s11:
            :math:`\sigma_{11}`
        s22:
            :math:`\sigma_{22}`
        s12:
            :math:`\sigma_{12}`
        p:
            The power (:math:`p`) to raise the von Mises stress.

        Returns
        -------
        numpy.ndarray
            The von Mises stress to the :math:`p^{\text{th}}` power.

        """
        return numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)**p

    @staticmethod
    def dsigma_pow(s11: numpy.ndarray, s22: numpy.ndarray, s12: numpy.ndarray,
                   ds11: numpy.ndarray, ds22: numpy.ndarray,
                   ds12: numpy.ndarray, p: float) -> numpy.ndarray:
        r"""
        Compute the gradient of the stress to the :math:`p^{\text{th}}` power.

        :math:`\nabla\sigma^p = \frac{p\sigma^{p-1}}{2\sigma}\nabla(\sigma^2)`

        Todo:
            * Properly document what the sigma variables represent.
            * Rename the sigma variables to something more readable.

        Parameters
        ----------
        s11:
            :math:`\sigma_{11}`
        s22:
            :math:`\sigma_{22}`
        s12:
            :math:`\sigma_{12}`
        ds11:
            :math:`\nabla\sigma_{11}`
        ds22:
            :math:`\nabla\sigma_{22}`
        ds12:
            :math:`\nabla\sigma_{12}`
        p:
            The power (:math:`p`) to raise the von Mises stress.

        Returns
        -------
        numpy.ndarray
            The gradient of the von Mises stress to the :math:`p^{\text{th}}`
            power.

        """
        sigma = numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
        dinside = (2 * s11 * ds11 - s11 * ds22 - ds11 * s22 + 2 * s22 *
                   ds22 + 6 * s12 * ds12)
        return p * (sigma)**(p - 1) / (2.0 * sigma) * dinside

    def compute_stress_objective(self, xPhys, dobj, p=4):
        """Compute stress objective and its gradient."""
        # Setup and solve FE problem
        # self.update_displacements(xPhys)

        rho = self.compute_young_moduli(xPhys)
        EBu = sum([self.EB @ self.u[:, i][self.edofMat.T]
                   for i in range(self.nloads)])
        s11, s22, s12 = numpy.hsplit((EBu * rho / float(self.nloads)).T, 3)
        # Update the stress for plotting
        self.stress[:] = numpy.sqrt(
            s11**2 - s11 * s22 + s22**2 + 3 * s12**2).squeeze()

        obj = self.sigma_pow(s11, s22, s12, p).sum()

        # Setup and solve FE problem
        K = self.build_K(xPhys)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(numpy.int), K.col.astype(numpy.int))

        # Setup dK @ u
        dK = self.build_dK(xPhys).tocsc()
        U = numpy.tile(self.u[self.free, :], (self.nel, 1))
        dKu = (dK @ U).reshape((-1, self.nel * self.nloads), order="F")

        # Solve system and solve for du: K @ du = dK @ u
        rhs = cvxopt.matrix(dKu)
        cvxopt.cholmod.linsolve(K, rhs)  # rhs stores solution after solve
        self.du[self.free, :] = -numpy.array(rhs)

        du = self.du.reshape((self.ndof * self.nel, self.nloads), order="F")
        rep_edofMat = (numpy.tile(self.edofMat.T, self.nel) + numpy.tile(
            numpy.repeat(numpy.arange(self.nel) * self.ndof, self.nel),
            (8, 1)))
        dEBu = sum([self.EB @ du[:, j][rep_edofMat]
                    for j in range(self.nloads)])
        rhodEBu = numpy.tile(rho, self.nel) * dEBu
        drho = numpy.empty(xPhys.shape)
        self.compute_young_moduli(xPhys, drho)
        drhoEBu = numpy.diag(drho).flatten() * numpy.tile(EBu, self.nel)
        ds11, ds22, ds12 = map(
            lambda x: x.reshape(self.nel, self.nel).T,
            numpy.hsplit(((drhoEBu + rhodEBu) / float(self.nloads)).T, 3))
        dobj[:] = self.dstress[:] = self.dsigma_pow(
            s11, s22, s12, ds11, ds22, ds12, p).sum(0)

        return obj

    def test_calculate_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray, p: float = 4,
            dx: float = 1e-6) -> float:
        """
        Calculate the gradient of the stresses using finite differences.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of the stresses to compute.
        p:
            The exponent for computing the softmax of the stresses.
        dx:
            The difference in x values used for finite differences.

        Returns
        -------
        float
            The analytic objective value.

        """
        dobja = dobj.copy()  # Analytic gradient
        obja = self.compute_stress_objective(
            xPhys, dobja, p)  # Analytic objective
        dobjf = dobj.copy()  # Finite difference of the stress
        delta = numpy.zeros(xPhys.shape)
        for i in range(xPhys.shape[0]):
            delta[[i - 1, i]] = 0, dx
            self.update_displacements(xPhys + delta)
            s1 = self.compute_stress_objective(xPhys + delta, dobj.copy(), p)
            self.update_displacements(xPhys - delta)
            s2 = self.compute_stress_objective(xPhys - delta, dobj.copy(), p)
            dobjf[i] = ((s1 - s2) / (2. * dx))

        print("Differences: {:g}".format(numpy.linalg.norm(dobjf - dobja)))
        # print("Analytic Norm: {:g}".format(numpy.linalg.norm(ds)))
        # print("Numeric Norm:  {:g}".format(numpy.linalg.norm(dsf)))
        # print("Analytic:\n{:s}".format(ds))
        # print("Numeric:\n{:s}".format(dsf))
        return obja

    def compute_objective(self, xPhys, dobj):
        """Compute compliance and its gradient."""
        obj = ComplianceProblem.compute_objective(self, xPhys, dobj)
        self.compute_stress_objective(xPhys, numpy.zeros(dobj.shape))
        return obj
