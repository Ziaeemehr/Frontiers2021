import os
import sys
import pylab as plt
import numpy as np
import pandas as pd
from os.path import join
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_grid(nrows, ncols, left=0.05, right=0.9,
              bottom=0.05, top=0.95, hspace=0.2,
              wspace=0.2):

    gs = GridSpec(nrows, ncols)
    gs.update(left=left, right=right,
              hspace=hspace, wspace=wspace,
              bottom=bottom, top=top)
    ax = []
    if nrows > 1:
        for i in range(nrows):
            ax_row = []
            for j in range(ncols):
                ax_row.append(plt.subplot(gs[i, j]))
            ax.append(ax_row)
    else:
        for j in range(ncols):
            ax.append(plt.subplot(gs[j]))

    return ax


def plot_scatter(x, y, ax,
                 xlabel=None,
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 color=None,
                 alpha=0.4,
                 label=None,
                 title=None,
                 cmap=None,
                 c=None,
                 colorbar=False,
                 cbar_label=None,
                 vmin=0,
                 vmax=1,
                 ticks=None):

    xl = x.reshape(-1)
    yl = y.reshape(-1)
    im = ax.scatter(xl, yl, s=10, c=c, color=color, alpha=alpha,
                    cmap=cmap, label=label, vmax=vmax, vmin=vmin)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    if (xlim is None) and (ylim is None):
        ax.margins(x=0.02, y=0.02)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cbar = plt.colorbar(im, cax=cax, ticks=ticks)
        if cbar_label:
            cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=14)

    if title:
        ax.set_title(title)
# ---------------------------------------------------------------------- #


def calculate_distance(nu, metric="euclidean", binary=False):

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    if binary:
        adj_mat = binarize(adj_mat)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)

    distance = np.zeros(len(nu))
    for j in range(len(nu)):
        subnamefr = str("%.6f-%.1f" % (g, nu[j]))
        cor = np.load(join(data_path, "npz",
                           "{:s}.npz".format(subnamefr)))['cor']
        cor = cor.reshape(-1)

        # x, y, z = L[L > 1], adj_mat[L > 1], cor[L > 1]
        x, y, z = L.reshape(-1), adj_mat.reshape(-1), cor.reshape(-1)
        X = np.vstack((x, y)).T
        Y = np.vstack((x, z)).T

        distance[j] = pdist(np.vstack((y, z)), metric=metric)
        cmap = plt.cm.get_cmap('plasma_r')  # RdYlBu_r

    np.savez(join(data_path,
                  "npz",
                  "distance-{}".format(metric)),
             nu=nu,
             d=distance)
# ---------------------------------------------------------------------- #


def binarize(data, threshold=1e-8):
    data = np.asarray(data)
    upper, lower = 1, 0
    data = np.where(data >= threshold, upper, lower)
    return data
# ---------------------------------------------------------------------- #


def fit_line(x, y):
    '''
    fit a line to given x and y and return the slope and fitted values
    '''

    df_data = pd.DataFrame({"y": y, "x": x})
    model = smf.ols("y ~ 1 + x", df_data)
    result = model.fit()
    slope = result.params["x"]
    return slope, result.fittedvalues
# ---------------------------------------------------------------------- #


nu = [3, 11, 23, 35, 51]
mu = [(2.0 * np.pi * i / 1000.0) for i in nu]
NU = np.arange(1, 100, 2)
g = float(sys.argv[1])

if __name__ == "__main__":

    data_path = "../data"

    network_subname = "Human_65"
    delaymat_name = "{}_delaymat.txt".format(network_subname)
    distancemat_name = "{}_distancemat.txt".format(network_subname)
    connectmat_name = "{}_connectmat.txt".format(network_subname)
    xyz_centers_name = "{}_region_xyz_centers.txt".format(network_subname)
    max_distance = 55

    calculate_distance(NU)

    fig = plt.figure(figsize=(15, 3))

    ax = make_grid(nrows=1, ncols=5, left=0.1, right=0.96,
                   bottom=0.1, top=0.93,
                   hspace=0.2, wspace=0.35)

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    adj_mat = binarize(adj_mat, 1e-8)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    cmap = plt.cm.get_cmap('plasma_r')
    titles = [r"$\theta$", r"$\alpha$",
              r"$\beta$", r"$\gamma$", "high-$\gamma$"]
    # labels = ["(A)", "(B)", "(C)", "(D)"]
    x, y = L[L > 1], adj_mat[L > 1]
    y_bar = []
    colorbar = True
    slopes = []
    for i in range(len(nu)):

        subnamefr = str("%.6f-%.1f" % (g, nu[i]))
        cor = np.load(join(data_path, "npz",
                           "{:s}.npz".format(subnamefr)))['cor']
        cor = cor.reshape(-1)
        y_bar.append(np.mean(cor))

        z = cor[L > 1]  # !
        plot_scatter(x, z, c=y, ax=ax[i], cmap=cmap,
                     ylim=[-1, 1], alpha=0.7, ticks=[0, 1],
                     label=titles[i], colorbar=colorbar)

        ax[i].set_yticks([-1, 0, 1])
        ax[i].set_xticks([0, max_distance])
        ax[i].set_xlabel("D", fontsize=16, labelpad=-15)
        ax[i].set_ylabel("C", fontsize=16, labelpad=-5)
        ax[i].tick_params(labelsize=16)
        ax[i].legend(fontsize=14, frameon=False, loc="upper right")
    
    

    fig.savefig("figure3_{:.3f}.jpg".format(g), dpi=300)
    plt.close()
