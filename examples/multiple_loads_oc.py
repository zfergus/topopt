#!/usr/bin/env python
# -*- coding: utf-8 -*-

# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE
# JOHANSEN,  JANUARY 2013
from __future__ import division, print_function

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from matplotlib import colors
import matplotlib.pyplot as plt

import cvxopt
import cvxopt.cholmod

import context  # noqa

from topopt.utils import xy_to_id, id_to_xy

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass


# MAIN DRIVER
def main(nelx, nely, volfrac, penalty, rmin, ft):
    print("Minimum compliance problem with OC")
    print("ndes: %d x %d" % (nelx, nely))
    print("volfrac: %s, rmin: %s, penalty: %s" % (volfrac, rmin, penalty))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)
    # Allocate design variables (as array),  initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    dc = np.zeros((nely, nelx))
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2 * n1 + 2,  2 * n1 + 3,  2 * n2 + 2,
                                       2 * n2 + 3, 2 * n2,  2 * n2 + 1,  2 * n1,  2 * n1 + 1])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index + data vectors for the coo matrix
    # format
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt(
                        ((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)
    # BC's and support
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    bottom_left = 2 * xy_to_id(0, nely, nelx, nely)
    bottom_right = 2 * xy_to_id(nelx, nely, nelx, nely)
    fixed = np.array([bottom_left, bottom_left + 1,
                      bottom_right, bottom_right + 1])
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # Set load
    f = np.zeros((2 * (nelx + 1) * (nely + 1), 2))
    id1 = 2 * xy_to_id(7 * nelx // 20, 0, nelx, nely) + 1
    id2 = 2 * xy_to_id(13 * nelx // 20, 0, nelx, nely) + 1
    f[id1, 0] = -1
    f[id2, 1] = -1
    # f[1, 0] = -1
    # f[2 * (nelx - 1) * (nely + 1), 1] =  -1
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    plt.xlabel(
        "ndes: {:d} x {:d}\nvolfrac: {:g}, rmin: {:g}, penalty: {:g}".format(
            nelx, nely, volfrac, rmin, penalty))
    plot_force_arrows(nelx, nely, f, ax)
    fig.tight_layout()
    fig.show()
    # Set loop counter and gradient vectors
    dv = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)
    for loop in range(2000):  # while change > 0.01 and loop < 100:
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T *
              (Emin + (xPhys)**penalty * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = deleterowcol(K, fixed, fixed).tocoo()
        # Solve system
        # u[free, :] = spsolve(K.tocsc(), f[free, :])
        K = cvxopt.spmatrix(K.data, K.row, K.col)
        B = cvxopt.matrix(f[free, :])
        cvxopt.cholmod.linsolve(K, B)
        u[free, :] = np.array(B)[:, :]
        # Objective and sensitivity
        obj = 0
        dc = np.zeros(nely * nelx)
        # import pdb; pdb.set_trace()
        for i in range(f.shape[1]):
            ui = u[:, i][edofMat]
            ce[:] = (ui.dot(KE) * ui).sum(1)
            obj += ((Emin + xPhys**penalty * (Emax - Emin)) * ce).sum()
            dc[:] += (-penalty * xPhys**(penalty - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = (np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] /
                     np.maximum(0.001, x))
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        # Compute the change by the inf. norm
        change = (np.linalg.norm(x.reshape(nelx * nely, 1) -
                                 xold.reshape(nelx * nely, 1), np.inf))
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.005)
        # Write iteration history to screen
        print("it.: %3d, obj.: %9.3f, Vol.: %.2f, ch.: %.3f" % (loop, obj,
                                                                (g + volfrac * nelx * nely) / (nelx * nely), change))
        if change < 0.01:
            break
    # Make sure the plot stays and that the shell remains
    plt.show()
    input("Press any key...")


# Element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array([1 / 2. - nu / 6., 1 / 8 + nu / 8, -1 / 4 - nu / 12,
                  -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6,
                  1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


# Optimality criterion
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
                                                                  np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)


def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


def plot_force_arrows(nelx, nely, f, ax):
    """Add arrows to the plot for each force."""
    arrowprops = {"arrowstyle": "->", "connectionstyle": "arc3",
                  "lw": "2", "color": 0}
    cmap = plt.cm.get_cmap("hsv", f.shape[1] + 1)
    for load_i in range(f.shape[1]):
        nz = np.nonzero(f[:, load_i])
        arrowprops["color"] = cmap(load_i)
        for i in range(nz[0].shape[0]):
            x, y = id_to_xy(nz[0][i] // 2, nelx, nely)
            x = max(min(x, nelx - 1), 0)
            y = max(min(y, nely - 1), 0)
            z = int(nz[0][i] % 2)
            mag = -50 * f[nz[0][i], load_i]
            ax.annotate("", xy=(x, y), xycoords="data",
                        xytext=(0 if z else mag, mag if z else 0),
                        textcoords="offset points", arrowprops=arrowprops)


# The real main driver
if __name__ == "__main__":
    import sys
    # Default input parameters
    nelx, nely, volfrac, penalty, rmin, ft = (sys.argv[1:] +
                                            [120, 60, 0.2, 3.0, 1.5, 1][len(sys.argv) - 1:])[:6]
    nelx, nely, ft = map(int, [nelx, nely, ft])
    volfrac, penalty, rmin = map(float, [volfrac, penalty, rmin])
    main(nelx, nely, volfrac, penalty, rmin, ft)
