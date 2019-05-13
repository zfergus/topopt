from __future__ import division

import numpy
import nlopt

from topopt.utils import camel_case_to_spaces


class TopOptSolver:

    def __init__(self, problem, volfrac, filter, gui, maxeval=2000, ftol=1e-3):
        self.problem, self.volfrac, self.filter, self.gui = (
            problem, volfrac, filter, gui)

        n = problem.nelx * problem.nely
        self.opt = nlopt.opt(nlopt.LD_MMA, n)
        self.xPhys = numpy.ones(n)

        # set bounds
        self.opt.set_upper_bounds(numpy.ones(n))
        self.opt.set_lower_bounds(numpy.zeros(n))

        # set stopping criteria
        self.opt.set_maxeval(maxeval)
        self.opt.set_ftol_rel(ftol)

        # set objective and constraint functions
        self.opt.set_min_objective(self.objective_function)
        self.opt.add_inequality_constraint(self.volume_function, 0)
        self.volfrac = volfrac # max volume fraction to use

        # setup filter
        self.passive = problem.bc.get_passive_elements()
        if self.passive is not None:
            self.xPhys[self.passive] = 0

    def __str__(self):
        return self.__class__.__name__

    def __format__(self, format_spec):
        return "{} with {}".format(str(self.problem), str(self))

    def optimize(self, x):
        self.xPhys = x.copy()
        x = self.opt.optimize(x)
        return x

    def filter_variables(self, x):
        self.filter.filter_variables(x, self.xPhys)
        if self.passive is not None:
            self.xPhys[self.passive] = 0
        return self.xPhys

    def objective_function(self, x, dobj):
        # Filter design variables
        self.filter_variables(x)

        # Setup and solve FE problem
        self.problem.update_displacements(self.xPhys)

        # Objective and sensitivity
        obj = self.problem.compute_objective(self.xPhys, dobj)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        self.gui.update(self.xPhys)

        return obj

    def objective_function_fdiff(self, x, dobj):
        obj = self.objective_function(x, dobj)

        x0 = x.copy()
        dobj0 = dobj.copy()
        dobjf = numpy.zeros(dobj.shape)
        for i, v in enumerate(x):
            x = x0.copy()
            x[i] += 1e-6
            o1 = self.objective_function(x, dobj)
            x[i] = x0[i] - 1e-6
            o2 = self.objective_function(x, dobj)
            dobjf[i] = (o1 - o2) / (2e-6)
            print("finite differences: {:g}".format(
                numpy.linalg.norm(dobjf - dobj0)))
            dobj[:] = dobj0

        return obj

    def volume_function(self, x, dv):
        # Filter design variables
        self.filter_variables(x)

        # Volume sensitivities
        dv[:] = 1.0

        # Sensitivity filtering
        self.filter.filter_volume_sensitivities(self.xPhys, dv)

        return sum(self.xPhys) - self.volfrac * len(x)


# TODO: Seperate optimizer from TopOptSolver
# class MMASolver(TopOptSolver):
#     pass
#
#
# TODO: Port over OC to TopOptSolver
# class OCSolver(TopOptSolver):
#     def oc(self, x, volfrac, dc, dv, g):
#         """ Optimality criterion """
#         l1 = 0
#         l2 = 1e9
#         move = 0.2
#         # reshape to perform vector operations
#         xnew = np.zeros(nelx * nely)
#         while (l2 - l1) / (l1 + l2) > 1e-3:
#             lmid = 0.5 * (l2 + l1)
#             xnew[:] =  np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
#                 np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
#             gt = g + np.sum((dv * (xnew - x)))
#             if gt > 0:
#                 l1 = lmid
#             else:
#                 l2 = lmid
#         return (xnew, gt)


class MechanismSynthesisSolver(TopOptSolver):

    def __init__(self, problem, volfrac, filter, gui, maxeval=2000, ftol=1e-4):
        TopOptSolver.__init__(
            self, problem, volfrac, filter, gui, maxeval, ftol)
        self.init_obj = None
        self.vtot = problem.nelx * problem.nely * volfrac

    def objective_function(self, x, dobj):
        # Filter design variables
        self.filter_variables(x)

        # Display physical variables
        self.gui.update(self.xPhys)

        # Setup and solve FE problem
        self.problem.update_displacements(self.xPhys)

        # Objective and sensitivity
        obj = self.problem.compute_objective(self.xPhys, dobj)
        if self.init_obj is None:
            self.init_obj = obj
        obj /= self.init_obj

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)
        dobj /= self.init_obj

        # print(obj * self.init_obj)
        return obj

    def volume_function(self, x, dv):
        # Filter design variables
        self.filter_variables(x)

        # Volume sensitivities
        dv[:] = 1.0 / self.vtot

        # Sensitivity filtering
        self.filter.filter_volume_sensitivities(self.xPhys, dv)

        return self.xPhys.sum() / self.vtot - 1
