import os
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
from config import *
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
    dist_mat = np.loadtxt(join(data_path, "L65.txt"))
    adj_mat = np.loadtxt(join(data_path, "C65.txt"))
    adj_mat = adj_mat / np.max(adj_mat) * 100

    filt_weight = filter_matrix(adj_mat, low, high)
    colors = pl.cm.jet(np.linspace(0, 1, len(S) + 1))

    for s in range(len(S)):
        _low, _high = S[s] - 5, S[s] + 5  # s - 1, s + 1
        dist = 0.5 * (_low + _high)
        filt_length = filter_matrix(dist_mat, _low, _high)
        cor_dist_mean = np.zeros(len(nu))
        cor_dist_std = np.zeros(len(nu))

        for j in range(len(nu)):
            subnamefr = str("%.6f-%.1f" % (G[0], nu[j]))
            cor = np.load(join(data_path, "npz",
                               "{:s}.npz".format(subnamefr)))['cor']
            intersection = find_intersection(filt_length, filt_weight)
            Graph = nx.from_numpy_array(intersection)
            edges = list(Graph.edges())

            if len(edges) > 1:
                tmp = []
                for edge in edges:
                    tmp.append(cor[edge[0], edge[1]])

            cor_dist_mean[j] = np.mean(tmp)
            cor_dist_std[j] = np.std(tmp)
        y1 = cor_dist_mean + 0.5 * cor_dist_std
        y2 = cor_dist_mean - 0.5 * cor_dist_std
        ax.fill_between(nu, y1, y2, alpha=0.2, color=colors[s])
        ax.plot(nu,
                cor_dist_mean,
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

    dist_mat = np.loadtxt(join(data_path, "L65.txt"))
    adj_mat = np.loadtxt(join(data_path, "C65.txt"))
    adj_mat = adj_mat / np.max(adj_mat) * 100

    filt_length = filter_matrix(dist_mat, low, high)
    colors = pl.cm.jet(np.linspace(0, 1, len(S)+1))
    for s in range(len(S)):
        _low, _high = S[s], S[s] + 6  # s - 1, s + 1
        weight = 0.5 * (_low + _high)
        filt_weight = filter_matrix(adj_mat, _low, _high)
        cor_dist_mean = np.zeros(len(nu))
        cor_dist_std = np.zeros(len(nu))

        for j in range(len(mu)):
            subnamefr = str("%.6f-%.1f" % (G[0], nu[j]))
            cor = np.load(join(data_path, "npz",
                               "{:s}.npz".format(subnamefr)))['cor']
            intersection = find_intersection(filt_length, filt_weight)
            Graph = nx.from_numpy_array(intersection)
            edges = list(Graph.edges())

            if len(edges) > 1:
                tmp = []
                for edge in edges:
                    tmp.append(cor[edge[0], edge[1]])

            cor_dist_mean[j] = np.mean(tmp)
            cor_dist_std[j] = np.std(tmp)

        y1 = cor_dist_mean + cor_dist_std
        y2 = cor_dist_mean - cor_dist_std
        ax.fill_between(nu, y1, y2, alpha=0.2, color=colors[s])
        ax.plot(nu,
                cor_dist_mean,
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

    start = time()
    data_path = "../data"
    fig, ax = pl.subplots(2, 2, figsize=(11, 7))

    prob = [0.025, 0.975]

    plot_singleW_multipleL([10, 30, 50, 90], prob, 10, 20, ax[0][0], "(A)")
    plot_singleW_multipleL([20, 30, 50, 70], prob, 20, 30, ax[0][1], "(B)")
    plot_singleL_multipleW([0, 10, 30, 40], prob, 20, 30, ax[1][0], "(C)")
    plot_singleL_multipleW([0, 10, 20, 30, 40, 50, 60],
                        prob, 30, 40, ax[1][1], "(D)")

    for i in range(2):
        for j in range(2):
            ax[i][j].set_ylim(-1, 1)
    
    pl.tight_layout()
    fig.savefig("fig-{:.2f}_std.png".format(prob[0] * 2))
    fig.savefig("fig-{:.2f}_std.pdf".format(prob[0] * 2))
    print("Done in %.3f seconds" % (time() - start))
