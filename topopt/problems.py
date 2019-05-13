"""Topology optimization problem to solve."""
from __future__ import division

import numpy

import scipy.sparse
import cvxopt
import cvxopt.cholmod

from topopt.utils import deleterowcol


class TopOptProblem:
    """Abstract topology optimization problem."""

    @staticmethod
    def lk(E=1.0, nu=0.3):
        """
        Build the element stiffness matrix.

        Parameters
        ----------
            E : float
                Young's modulus of the material
            nu : float
                Poisson's ratio of the material

        Returns
        -------
        numpy.ndarray
            The element stiffness matrix for the material

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

    def __init__(self, nelx, nely, penal, bc):
        """
        Create the topology optimization problem.

        Parameters
        ----------
            nelx : int
                Number of elements in the x direction
            nely : int
                Number of elements in the x direction
            penal : float
                Penalty value used to penalize fractional densities in SIMP
            bc : topopt.boundary_conditions.TopOptBoundaryConditions
                Boundary conditions of the problem

        """
        # Problem size
        self.nelx = nelx
        self.nely = nely
        self.nel = nelx * nely

        # Count degrees of fredom
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # Max and min stiffness
        self.Emin = 1e-9
        self.Emax = 1.0

        # SIMP penalty
        self.penal = penal

        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.3
        self.build_indices(nelx, nely)

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.get_fixed_nodes()
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # RHS and Solution vectors
        self.f = bc.get_forces()
        self.u = numpy.zeros(self.f.shape)
        self.nloads = self.f.shape[1]

        # Per element objective
        self.obje = numpy.zeros(nely * nelx)

    def __str__(self):
        """Create a string representation of the problem."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the problem."""
        return str(self)

    def __repr__(self):
        """Create a representation of the problem."""
        return "{:s}(nelx={:d}, nely={:d}, penal={:g}, bc={:s})".format(
            self.__class__.__name__, self.nelx, self.nely, self.penal,
            repr(self.bc))

    def build_indices(self, nelx, nely):
        """Build the index vectors for the finite element coo matrix format."""
        self.KE = self.lk(E=self.Emax, nu=self.nu)
        self.edofMat = numpy.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = numpy.array([
                    2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2,
                    2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((8, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 8))).flatten()

    def penalized_densities(self, x):
        """Compute the penalized densties."""
        return self.Emin + (self.Emax - self.Emin) * x**self.penal

    def diff_penalized_densities(self, x):
        """Compute the gradient of penalized densties."""
        return (self.Emax - self.Emin) * self.penal * x**(self.penal - 1)

    def build_K(self, xPhys, remove_constrained=True):
        """Build the stiffness matrix for the problem."""
        sK = ((self.KE.flatten()[numpy.newaxis]).T *
              self.penalized_densities(xPhys)).flatten(order='F')
        K = scipy.sparse.coo_matrix(
            (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        # Remove constrained dofs from matrix and convert to coo
        if remove_constrained:
            K = deleterowcol(K.tocsc(), self.fixed, self.fixed).tocoo()
        return K

    def compute_displacements(self, xPhys):
        """Compute the displacments given the densities."""
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

    def update_displacements(self, xPhys):
        """Update the displacments of the problem."""
        self.u[:, :] = self.compute_displacements(xPhys)

    def compute_objective(self, xPhys, dobj):
        """Compute objective and its gradient."""
        raise NotImplementedError(
            "Subclasses of {} must override compute_objective()!".format(
                self.__class__.__name__))


class ComplianceProblem(TopOptProblem):
    """Topology optimization problem to minimize compliance."""

    def compute_objective(self, xPhys, dobj):
        """Compute compliance and its gradient."""
        obj = 0.0
        dobj[:] = 0.0
        rho = self.penalized_densities(xPhys)
        drho = self.diff_penalized_densities(xPhys)
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            self.obje[:] = (ui.dot(self.KE) * ui).sum(1)
            obj += (rho * self.obje).sum()
            dobj[:] += -drho * self.obje
        dobj /= float(self.nloads)
        return obj / float(self.nloads)


class MechanismSynthesisProblem(TopOptProblem):
    """Topology optimization problem to generate compliance mechanisms."""

    @staticmethod
    def lk(E=1.0, nu=0.3):
        """
        Build the element stiffness matrix.

        Parameters
        ----------
            E (float): Young's modulus of the material (Default: 1.0)
            nu (float): Poisson's ratio of the material (Default: 0.3)
        ----------

        Returns
        -------
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


class VonMisesStressProblem(TopOptProblem):
    """
    Topology optimization problem to minimize stress.

    TODO: Currently this problem minimizes compliance and computes stress on
    the side. This needs to be replaced to match the promise of minimizing
    stress.
    """

    @staticmethod
    def B(side):
        """Precomputed strain-displacement matrix."""
        # TODO: Check that this is not -B
        n = -0.5 / side
        p = 0.5 / side
        return numpy.array([[p, 0, n, 0, n, 0, p, 0],
                            [0, p, 0, p, 0, n, 0, n],
                            [p, p, p, n, n, n, n, p]])

    @staticmethod
    def E(nu):
        """Precomputed constitutive matrix."""
        return numpy.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1 - nu) / 2.]]) / (1 - nu**2)

    def __init__(self, nelx, nely, penal, bc, side=1):
        TopOptProblem.__init__(self, nelx, nely, penal, bc)
        self.EB = self.E(self.nu).dot(self.B(side))
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
        drho = self.diff_penalized_densities(xPhys)
        blocks = [self.build_dK0(drho[i], i, remove_constrained)
                  for i in range(drho.shape[0])]
        dK = scipy.sparse.block_diag(blocks, format="coo")
        return dK

    @staticmethod
    def sigma_pow(s11, s22, s12, p):
        return numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)**p

    @staticmethod
    def dsigma_pow(s11, s22, s12, ds11, ds22, ds12, p):
        sigma = numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
        dinside = (2 * s11 * ds11 - s11 * ds22 - ds11 * s22 + 2 * s22 *
                   ds22 + 6 * s12 * ds12)
        return p * (sigma)**(p - 1) / (2.0 * sigma) * dinside

    def compute_objective2(self, xPhys, dobj, p=4):
        """Compute stress objective and its gradient."""
        rho = self.penalized_densities(xPhys)
        EBu = sum([self.EB.dot(self.u[:, i][self.edofMat.T])
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
        dKu = dK.dot(U).reshape((-1, self.nel * self.nloads), order="F")

        # Solve system and solve for du: K @ du = dK @ u
        rhs = cvxopt.matrix(dKu)
        cvxopt.cholmod.linsolve(K, rhs)  # rhs stores solution after solve
        self.du[self.free, :] = -numpy.array(rhs)

        du = self.du.reshape((self.ndof * self.nel, self.nloads), order="F")
        rep_edofMat = (numpy.tile(self.edofMat.T, self.nel) + numpy.tile(
            numpy.repeat(numpy.arange(self.nel) * self.ndof, self.nel), (8, 1)))
        dEBu = sum([self.EB.dot(du[:, j][rep_edofMat])
                    for j in range(self.nloads)])
        rhodEBu = numpy.tile(rho, self.nel) * dEBu
        drho = self.diff_penalized_densities(xPhys)
        drhoEBu = numpy.diag(drho).flatten() * numpy.tile(EBu, self.nel)
        ds11, ds22, ds12 = map(
            lambda x: x.reshape(self.nel, self.nel).T,
            numpy.hsplit(((drhoEBu + rhodEBu) / float(self.nloads)).T, 3))
        dobj[:] = self.dstress[:] = self.dsigma_pow(
            s11, s22, s12, ds11, ds22, ds12, p).sum(0)

        return obj

    def test_calculate_objective(self, xPhys, dobj, p=4, dx=1e-6):
        """
        Calculate the gradient of the von Mises stress using finite
        differences given the densities x. Optionally, provide a delta x
        (default: 1e-6).
        """
        dobja = dobj.copy()  # Analytic gradient
        obja = self.compute_objective2(xPhys, dobja, p)  # Analytic objective
        dobjf = dobj.copy()  # Finite difference of the stress
        delta = numpy.zeros(xPhys.shape)
        for i in range(xPhys.shape[0]):
            delta[[i - 1, i]] = 0, dx
            self.update_displacements(xPhys + delta)
            s1 = self.compute_objective2(xPhys + delta, dobj.copy(), p)
            self.update_displacements(xPhys - delta)
            s2 = self.compute_objective2(xPhys - delta, dobj.copy(), p)
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
        self.compute_objective2(xPhys, numpy.zeros(dobj.shape))
        return obj
