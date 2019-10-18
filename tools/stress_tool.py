# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy

from matplotlib import colors
import matplotlib.pyplot as plt

import context  # noqa

from topopt.boundary_conditions import BoundaryConditions
from topopt.von_mises_stress import VonMisesStressCalculator
from topopt.utils import xy_to_id, id_to_xy


def set_displacements(u, func, nelx, nely, min_corner=(-1, -1),
                      max_corner=(1, 1)):
    for i in range(nelx + 1):
        for j in range(nely + 1):
            id = xy_to_id(i, j, nelx, nely)
            x = min_corner[0] + (i / nelx) * (max_corner[0] - min_corner[0])
            y = min_corner[1] + (j / nely) * (max_corner[1] - min_corner[1])
            fx, fy = func(x, y)
            u[2 * id] = fx
            u[2 * id + 1] = fy


def set_labels(values):
    x_positions = numpy.linspace(start=0, stop=nelx, num=nelx,
                                 endpoint=False)
    y_positions = numpy.linspace(start=0, stop=nelx, num=nelx,
                                 endpoint=False)

    for text in ax.texts:
        text.set_visible(False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = "{:.2f}".format(values[y_index, x_index])
            text_x = x + 0.5
            text_y = y + 0.5
            ax.text(text_x, text_y, label, color='red', ha='center',
                    va='center')


def onpress(event):
    global selected_x, selected_y
    selected_x = int(max(0, min(nelx, numpy.round(event.xdata))))
    selected_y = int(max(0, min(nely, numpy.round(event.ydata))))
    if event.button == 1:
        print("Selected: x = {:d}, y = {:d}".format(selected_x, selected_y))
    else:
        id = xy_to_id(selected_x, selected_y, nelx, nely)
        u[2 * id] = 0
        u[2 * id + 1] = 0
        stress = vms.calculate_stress(xPhys, u, nu)
        im.set_array(stress.reshape((nelx, nely)).T)
        set_labels(stress.reshape((nelx, nely)).T)
        selected_x = selected_y = None


def onrelease(event):
    if event.button == 1:
        global selected_x, selected_y
        dx = event.xdata - selected_x
        dy = event.ydata - selected_y
        id = xy_to_id(selected_x, selected_y, nelx, nely)
        u[2 * id] += dx
        u[2 * id + 1] += dy
        stress = vms.calculate_stress(xPhys, u, nu)
        im.set_array(stress.reshape((nelx, nely)).T)
        set_labels(stress.reshape((nelx, nely)).T)
        print("Released: dx = {:g}, dy = {:g}".format(dx, dy))
        selected_x = selected_y = None


if __name__ == "__main__":
    E_min, E_max = 1e-9, 1
    nelx, nely = 4, 4
    nu, penal = 0.3, 0.5
    xPhys = 0.5 * numpy.ones(nelx * nely)
    u = numpy.zeros((2 * (nelx + 1) * (nelx + 1), 1))
    # func = eval(input("Enter lambda expresion for u0: "))
    # func = lambda x, y: (x*x - y*y, 2*x*y)
    # func = lambda x, y: (3*y+2, x - 2*y)
    # set_displacements(u, func, nelx, nely)

    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(numpy.zeros((nelx, nely)).T, cmap="viridis",
                   interpolation='none', norm=colors.Normalize(vmin=0, vmax=1),
                   origin="lower", extent=[0, nelx, 0, nely])
    ax.grid(color='r', linestyle='-', linewidth=(4 / nelx))
    plt.xticks(numpy.arange(nelx + 1))
    plt.xlim([0, nelx])
    plt.yticks(numpy.arange(nely + 1))
    plt.ylim([0, nely])

    fig.colorbar(im)
    fig.show()

    from topopt.problems import ComplianceProblem
    problem = ComplianceProblem(nelx, nely, penal,
                                BoundaryConditions(nelx, nely))
    vms = VonMisesStressCalculator(problem)
    stress = vms.calculate_stress(xPhys, u, nu)
    im.set_array(stress.reshape((nelx, nely)).T)
    # im.set_norm(colors.Normalize(vmin=0, vmax=numpy.max(stress)))
    set_labels(stress.reshape((nelx, nely)).T)

    selected_x = selected_y = None
    cid = fig.canvas.mpl_connect('button_press_event', onpress)
    cid = fig.canvas.mpl_connect('button_release_event', onrelease)
    input()
