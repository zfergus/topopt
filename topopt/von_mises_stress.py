# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pdb

import numpy
import scipy.sparse
import cvxopt
import cvxopt.cholmod

from topopt.utils import xy_to_id, id_to_xy, squared_euclidean as normsqr


class VonMisesStressCalculator:
    def __init__(self, problem):
        self.problem = problem
        self.edofMat = self.build_indices()

    @staticmethod
    def B(side):
        """ Precomputed strain-displacement matrix. """
        n = -0.5 / side
        p = 0.5 / side
        return numpy.array([[p, 0, n, 0, n, 0, p, 0],
                            [0, p, 0, p, 0, n, 0, n],
                            [p, p, p, n, n, n, n, p]])

    @staticmethod
    def E(nu):
        """ Precomputed constitutive matrix. """
        return numpy.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1 - nu) / 2.]]) / (1 - nu**2)

    def build_indices(self):
        nelx, nely = self.problem.nelx, self.problem.nely
        edofMat = numpy.zeros((8, nelx * nely), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely  # Element index
                n1 = (nely + 1) * elx + ely  # Left nodes
                n2 = (nely + 1) * (elx + 1) + ely  # Right nodes
                edofMat[:, el] = numpy.array([
                    2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2,
                    2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        return edofMat

    def penalized_densities(self, x):
        """ Compute the penalized densties. """
        Emin, Emax, penalty = (
            self.problem.Emin, self.problem.Emax, self.problem.penalty)
        return Emin + (Emax - Emin) * x**penalty

    def diff_penalized_densities(self, x):
        """ Compute the penalized densties. """
        Emin, Emax, penalty = (
            self.problem.Emin, self.problem.Emax, self.problem.penalty)
        return (Emax - Emin) * penalty * x**(penalty - 1)

    def calculate_principle_stresses(self, x, side=1):
        """
        Calculate the principle stresses in the x, y, and shear directions.
        """
        u = self.problem.compute_displacements(x)
        rho = self.penalized_densities(x)
        EB = self.E(self.problem.nu).dot(self.B(side))
        stress = sum([EB.dot(u[:, i][self.edofMat])
                      for i in range(u.shape[1])])
        stress *= rho / float(u.shape[1])
        return numpy.hsplit(stress.T, 3)

    def calculate_stress(self, x, side=1):
        """
        Calculate the Von Mises stress given the densities x.
        """
        s11, s22, s12 = self.calculate_principle_stresses(x, side)
        vm_stress = numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
        return vm_stress.squeeze()

    def calculate_diff_stress(self, x, side=1, p=2):
        """
        Calculate the gradient of the Von Mises stress given the densities x,
        displacements u, and young modulus nu. Optionally, provide the side
        length (default: 1).
        """
        nel = self.problem.nelx * self.problem.nely
        ndof = 2 * (self.problem.nelx + 1) * (self.problem.nely + 1)
        nloads = self.problem.f.shape[1]
        u = self.problem.compute_displacements(x)
        rho = self.penalized_densities(x)
        EB = self.E(self.problem.nu).dot(self.B(side))
        EBu = sum([EB.dot(u[:, i][self.edofMat]) for i in range(nloads)])
        s11, s22, s12 = numpy.hsplit((EBu * rho / float(nloads)).T, 3)

        def sigma_pow(s11, s22, s12):
            return numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)**p

        def dsigma_pow(ds11, ds22, ds12):
            sigma = numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
            dinside = (2 * s11 * ds11 - s11 * ds22 - ds11 * s22 + 2 * s22 *
                       ds22 + 6 * s12 * ds12)
            return p * (sigma)**(p - 1) / (2.0 * sigma) * dinside

        K = self.problem.build_K(x)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(numpy.int), K.col.astype(numpy.int))

        dK = self.problem.build_dK(x).tocsc()
        U = numpy.tile(u[self.problem.free, :], (nel, 1))
        U = cvxopt.matrix(dK.dot(U).reshape(-1, nel * nloads, order="F"))
        cvxopt.cholmod.linsolve(K, U)  # U stores solution after solve
        du = numpy.zeros((ndof, nel * nloads))
        du[self.problem.free, :] = -numpy.array(U)
        du = du.reshape((ndof * nel, nloads), order="F")

        rep_edofMat = (numpy.tile(self.edofMat, nel) + numpy.tile(
            numpy.repeat(numpy.arange(nel) * ndof, nel), (8, 1)))
        dEBu = sum([EB.dot(du[:, j][rep_edofMat]) for j in range(nloads)])
        rhodEBu = (numpy.tile(rho, nel) * dEBu)
        drho = self.diff_penalized_densities(x)
        drhoEBu = (numpy.diag(drho).flatten() * numpy.tile(EBu, nel))
        dstress = ((drhoEBu + rhodEBu) / float(nloads)).T
        ds11, ds22, ds12 = map(
            lambda x: x.reshape(nel, nel).T, numpy.hsplit(dstress, 3))
        ds = dsigma_pow(ds11, ds22, ds12).sum(0)
        return ds

    def calculate_fdiff_stress(self, x, side=1, dx=1e-6):
        """
        Calculate the gradient of the Von Mises stress using finite
        differences given the densities x. Optionally, provide the side length
        (default: 1) and delta x (default: 1e-6).
        """
        # p = 2
        ds = self.calculate_diff_stress(x, side)  # Analytic gradient
        dsf = numpy.zeros(x.shape)  # Finite difference of the stress
        delta = numpy.zeros(x.shape)
        for i in range(x.shape[0]):
            delta[[i - 1, i]] = 0, dx
            s1 = normsqr(self.calculate_stress(x + delta, side))
            s2 = normsqr(self.calculate_stress(x - delta, side))
            dsf[i] = ((s1 - s2) / (2. * dx))

        print("Differences: {:g}".format(numpy.linalg.norm(dsf - ds)))
        # print("Analytic Norm: {:g}".format(numpy.linalg.norm(ds)))
        # print("Numeric Norm:  {:g}".format(numpy.linalg.norm(dsf)))
        # print("Analytic:\n{:s}".format(ds))
        # print("Numeric:\n{:s}".format(dsf))
        return dsf


if __name__ == "__main__":
    from boundary_conditions import BoundaryConditions
    from problems import ComplianceProblem

    nelx = nely = 10

    bc = BoundaryConditions(nelx, nely)
    problem = ComplianceProblem(nelx, nely, 3.0, bc)
    vms = VonMisesStressCalculator(problem)
    x = 0.5 * numpy.ones(nelx * nely)
    vms.calculate_fdiff_stress(x)
