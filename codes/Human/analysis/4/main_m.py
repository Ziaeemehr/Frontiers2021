import os
import sys
import matplotlib
import pylab as pl
import numpy as np
import pandas as pd
from sys import exit
import networkx as nx
from PIL import Image
import pandas as pd
from time import time
from scipy import stats
from os.path import join
from scipy.optimize import bisect
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from scipy.stats.mstats import mquantiles
from mpl_toolkits.axes_grid1 import make_axes_locatable
from visbrain.objects import (BrainObj, SceneObj, SourceObj, ConnectObj)
# ---------------------------------------------------------------------- #


def find_intersection(A, B):
    assert (np.asarray(A).shape == np.asarray(B).shape)
    assert(isinstance(A[0][0].item(), int))

    row, col = A.shape
    C = np.zeros((row, col), dtype=int)

    for i in range(row):
        for j in range(col):

            if (A[i][j] == B[i][j]) and (A[i][j] != 0):
                C[i][j] = 1

    return C


def filter_matrix(A, low, high):

    n = A.shape[0]
    assert (len(A.shape) == 2)
    filt = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i, n):
            if (A[i][j] > low) and (A[i][j] < high):
                filt[i][j] = filt[j][i] = 1

    return filt


def make_grid(nrows, ncols, left, right, hspace, wspace, bottom, top):
    gs = GridSpec(nrows, ncols)
    gs.update(left=left, right=right,
              hspace=hspace, wspace=wspace,
              bottom=bottom, top=top)
    ax = []
    for i in range(nrows):
        for j in range(ncols):
            ax.append(pl.subplot(gs[i, j]))
    return ax
# ---------------------------------------------------------------------- #


def binarize(adj, threshold):
    """
    binarize the given 2d numpy array

    :param data: [2d numpy array] given array.
    :param threshold: [float] threshold value.
    :return: [2d int numpy array] binarized array.
    """

    adj = np.asarray(adj)
    upper, lower = True, False
    adj = np.where(adj >= threshold, upper, lower)
    return adj

# ---------------------------------------------------------------------- #


def plot_singleW_multipleL(S, prob, low, high, ax, label):
    dist_mat = np.loadtxt(join(data_path, distancemat_name))
    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat / np.max(adj_mat) * 100

    filt_weight = filter_matrix(adj_mat, low, high)
    colors = pl.cm.jet(np.linspace(0, 1, len(S) + 1))

    for s in range(len(S)):
        _low, _high = S[s] - 5, S[s] + 5  # s - 1, s + 1
        dist = 0.5 * (_low + _high)
        filt_length = filter_matrix(dist_mat, _low, _high)
        QS = np.zeros((len(mu), len(prob)))
        mean_value = np.zeros(len(mu))

        for j in range(len(nu)):
            subnamefr = str("%.6f-%.1f" % (g, nu[j]))
            cor = np.load(join(data_path, "npz",
                               "{:s}.npz".format(subnamefr)))['cor']
            intersection = find_intersection(filt_length, filt_weight)
            Graph = nx.from_numpy_array(intersection)
            edges = list(Graph.edges())
            tmp=[]
            if len(edges) > 1:
                tmp = []
                for edge in edges:
                    tmp.append(cor[edge[0], edge[1]])

            qs = mquantiles(tmp, prob=prob)
            QS[j, :] = qs
            mean_value[j] = np.mean(tmp)
        ax.fill_between(nu, *QS.T, alpha=0.2, color=colors[s])
        ax.plot(nu,
                mean_value,
                lw=2,
                color=colors[s],
                label="d={}".format(dist))

    ax.legend()
    ax.set_xlabel(r"$\nu_0$ [Hz]", fontsize=16)
    ax.set_ylabel(r"$\langle C\rangle$", fontsize=16)
    ax.text(-0.2, 0.9, label, fontsize=16, transform=ax.transAxes)
    text = "w={:.2f}".format(0.005*(low + high))
    ax.text(0.4, 0.05, text, fontsize=14, transform=ax.transAxes)
    ax.tick_params(labelsize=14)
    ax.set_xlim(np.min(nu), np.max(nu))


def plot_singleL_multipleW(S, prob, low, high, ax, label):

    dist_mat = np.loadtxt(join(data_path, distancemat_name))
    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat / np.max(adj_mat) * 100

    filt_length = filter_matrix(dist_mat, low, high)
    colors = pl.cm.jet(np.linspace(0, 1, len(S)+1))
    for s in range(len(S)):
        _low, _high = S[s], S[s] + 6  # s - 1, s + 1
        weight = 0.5 * (_low + _high)
        filt_weight = filter_matrix(adj_mat, _low, _high)
        QS = np.zeros((len(mu), len(prob)))
        mean_value = np.zeros(len(mu))

        for j in range(len(mu)):
            subnamefr = str("%.6f-%.1f" % (g, nu[j]))
            cor = np.load(join(data_path, "npz",
                               "{:s}.npz".format(subnamefr)))['cor']
            intersection = find_intersection(filt_length, filt_weight)
            Graph = nx.from_numpy_array(intersection)
            edges = list(Graph.edges())

            tmp = []
            if len(edges) > 1:
                tmp = []
                for edge in edges:
                    tmp.append(cor[edge[0], edge[1]])

            qs = mquantiles(tmp, prob=prob)
            QS[j, :] = qs
            mean_value[j] = np.mean(tmp)

        ax.fill_between(nu, *QS.T, alpha=0.2, color=colors[s])
        ax.plot(nu,
                mean_value,
                lw=2,
                color=colors[s],
                label="w={}".format(weight / 100))

    ax.legend()
    ax.set_xlabel(r"$\nu_0$ [Hz]", fontsize=16)
    ax.set_ylabel(r"$\langle C\rangle$", fontsize=16)
    ax.text(-0.2, 0.9, label, fontsize=16, transform=ax.transAxes)
    text = "d={:.2f}".format(0.5*(low + high))
    ax.text(0.4, 0.05, text, fontsize=14, transform=ax.transAxes)
    ax.tick_params(labelsize=14)
    ax.set_xlim(np.min(nu), np.max(nu))

if __name__ == "__main__":

    # N = 65
    nu = np.arange(1, 100, 2)
    mu = [(2.0 * np.pi * i / 1000.0) for i in nu]
    g = float(sys.argv[1])
    
    start = time()
    data_path = "../data"
    max_distance = 160
    network_subname = "Human_65"
    delaymat_name = "{}_delaymat.txt".format(network_subname)
    distancemat_name = "{}_distancemat.txt".format(network_subname)
    connectmat_name = "{}_connectmat.txt".format(network_subname)
    xyz_centers_name = "{}_region_xyz_centers.txt".format(network_subname)



    fig, ax = pl.subplots(2, 2, figsize=(11, 7))

    prob = [0.025, 0.975]

    plot_singleW_multipleL(range(10,100,20), prob, 10, 20, ax[0][0], "(A)")
    plot_singleW_multipleL([20, 30, 40, 50, 70], prob, 20, 30, ax[0][1], "(B)")
    plot_singleL_multipleW(range(0,50,10), prob, 20, 30, ax[1][0], "(C)")
    plot_singleL_multipleW(range(0,61,10), prob, 30, 40, ax[1][1], "(D)")
    
    for i in range(2):
        for j in range(2):
            ax[i][j].set_ylim(-1, 1)
            ax[i][j].set_xticks(range(0,110, 20))
            ax[i][j].set_yticks([-1,0,1])
    pl.tight_layout()
    fig.savefig("fig-{:.3f}.jpg".format(g), dpi=300)
    print("Done in %.3f seconds" % (time() - start))
