#!/usr/local/bin/python3
"""Tool to visualize the stresses of a elastic material with holes."""

import numpy

from context import topopt

from topopt.solvers import TopOptSolver
import topopt.von_mises_stress
import topopt.guis
from topopt.problems import VonMisesStressProblem
from topopt.boundary_conditions import BoundaryConditions
from topopt.utils import xy_to_id


class HoleBoundaryConditions(BoundaryConditions):
    """
    Boundary conditions for the hole tool.

    The bottom boundary is fixed and loads are applied to the top boundary.
    """

    @property
    def fixed_nodes(self):
        """Return a list of fixed nodes for the problem."""
        x = numpy.arange(self.nelx + 1)
        botx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, self.nely, self.nelx, self.nely))
        ids = 2 * botx_to_id(x)
        fixed = numpy.union1d(ids, ids + 1)
        return fixed

    @property
    def forces(self):
        """Return the force vector for the problem."""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        x = numpy.arange(0, self.nelx + 1, 10)
        topx_to_id = numpy.vectorize(
            lambda x: xy_to_id(x, 0, self.nelx, self.nely))
        ids = 2 * topx_to_id(x) + 1
        f = numpy.zeros((ndof, 1))
        f[ids] = -1
        return f


class HoleSolver(TopOptSolver):
    """
    Topology optimization solver for the hole tool.

    This solver does not optimze the problem at all. It just computes and
    visualizes the stresses.
    """

    def __init__(
            self, problem, volfrac, filter, gui, maxeval=100000, ftol=0.0):
        """Create a solver for the hole tool."""
        TopOptSolver.__init__(
            self, problem, volfrac, filter, gui, maxeval, ftol)

    def optimize(self, x):
        """
        Compute and visualize the stresses.

        Do not optimize the topology optimization problem.
        """
        nelx, nely = self.problem.nelx, self.problem.nely
        self.xPhys = x.copy()
        while True:
            self.compliance_function(self.xPhys, numpy.zeros(nelx * nely))
        return x

    def compliance_function(self, x, dc):
        """Update the GUI and the displacments of the problem."""
        # Display physical variables
        self.gui.update(self.xPhys)

        # Setup and solve FE problem
        self.problem.update_displacements(self.xPhys)


class HoleGUI(topopt.guis.StressGUI):
    """GUI for drawing the stresses and addind and removing material."""

    def __init__(self, problem, title=""):
        """Initialize the plot and plot the initial design."""
        super(HoleGUI, self).__init__(problem, title)
        self.nelx, self.nely = self.problem.nelx, self.problem.nely
        self.radius = 1

        # Set the callback functions to handle creating holes
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        """Create a hole at the point pressed."""
        # Clamp the point to the domain
        selected_x = int(max(0, min(self.nelx, numpy.round(event.xdata))))
        selected_y = int(max(0, min(self.nely, numpy.round(event.ydata))))
        print("Selected: x = {:d}, y = {:d}".format(selected_x, selected_y))

        selected_point = numpy.array([selected_x, selected_y])

        def dist_from_selected_point(x, y):
            """Compute the distance from the selected_point."""
            return numpy.linalg.norm(selected_point - numpy.array([x, y]))

        # Create a bounding box of the selected point and the selection radius
        min_x = max(0, selected_x - self.radius + 1)
        max_x = min(self.nelx, selected_x + self.radius)
        min_y = max(0, selected_y - self.radius + 1)
        max_y = min(self.nely, selected_y + self.radius)
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if dist_from_selected_point(x, y) <= self.radius:
                    solver.xPhys[
                        xy_to_id(x, y, self.nelx - 1,  self.nely - 1)] = (
                            0.0 if event.button == 1 else 1.0)

        self.problem.compute_objective2(solver.xPhys, self.problem.dstress)

    def on_scroll(self, event):
        """Increase or decrease radius on mouse scroll."""
        if event.button == "up":
            self.radius += 1
        elif event.button == "down":
            self.radius = max(1, self.radius - 1)
        print("radius = {:d}".format(self.radius))


def main():
    """Create the hole tools."""
    nelx, nely = 20, 20
    volfrac, penal = 1.0, 6
    x = volfrac * numpy.ones(nely * nelx, dtype=float)

    bc = HoleBoundaryConditions(nelx, nely)
    problem = VonMisesStressProblem(nelx, nely, penal, bc)
    gui = HoleGUI(problem)
    filter = None
    global solver
    solver = HoleSolver(problem, volfrac, filter, gui)
    solver.optimize(x)

    input("Press enter...")


if __name__ == "__main__":
    solver = None
    main()
