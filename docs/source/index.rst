.. TopOpt documentation master file

TopOpt --- Topology Optimization Library for Python
===================================================

.. image:: https://img.shields.io/github/license/zfergus/topopt.svg?color=blue
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

- **Free Software**: MIT License
- **Github Repository**: https://github.com/zfergus/topopt

`Topology optimization
<https://en.wikipedia.org/wiki/Topology_optimization>`_ is ...

Optimize the classic Messerschmitt–Bölkow–Blohm (MBB) beam in a few lines of
code:

.. code-block:: python

   import topopt
   from topopt.boundary_conditions import MBBBeamBoundaryConditions
   from topopt.problems import ComplianceProblem
   from topopt.solvers import TopOptSolver
   from topopt.filters import DensityBasedFilter
   from topopt.guis import GUI

   nelx, nely = 20, 20  # Number of elements in the x and y
   volfrac = 0.3  # Volume fraction for constraints
   penal = 3.0  # Penalty for SIMP
   rmin = 5.4  # Filter radius

   # Initial solution
   x = volfrac * numpy.ones(nely * nelx, dtype=float)

   # Boundary conditions defining the loads and fixed points
   bc = MBBBeamBoundaryConditions(nelx, nely)

   # Problem to optimize given objective and constraints
   problem = ComplianceProblem(nelx, nely, penal, bc)
   gui = GUI(problem, "Topology Optimization Example")
   topopt_filter = DensityBasedFilter(nelx, nely, rmin)
   solver = TopOptSolver(problem, volfrac, topopt_filter, gui)
   x_opt = solver.optimize(x)

   input("Press enter...")


.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: General

  installation
  concepts
  examples
  changelog

.. toctree::
 :hidden:
 :maxdepth: 2
 :caption: Developers

 contributing

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: API Documentation

  api/topopt.problems
  api/topopt.solvers
  api/topopt.boundary_conditions
  api/topopt.filters
  api/topopt.guis


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
