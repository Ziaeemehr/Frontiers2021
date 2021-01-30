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


def calculate_distance(nu, metric="euclidean",):

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
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

    network_subname = "Monkey_29"
    delaymat_name = "{}_delaymat.txt".format(network_subname)
    distancemat_name = "{}_distancemat.txt".format(network_subname)
    connectmat_name = "{}_connectmat.txt".format(network_subname)
    xyz_centers_name = "{}_region_xyz_centers.txt".format(network_subname)
    max_distance = 55

    calculate_distance(NU)

    fig = plt.figure(figsize=(15, 7))

    ax = make_grid(nrows=2, ncols=5, left=0.05, right=0.96,
                   bottom=0.4, top=0.93,
                   hspace=0.2, wspace=0.35)
    ax1 = make_grid(nrows=1, ncols=2, left=0.05, right=0.96,
                    bottom=0.1, top=0.33,
                    hspace=0.2, wspace=0.2)

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    cmap = plt.cm.get_cmap('plasma_r')
    titles = [r"$\theta$", r"$\alpha$",
              r"$\beta$", r"$\gamma$", "high-$\gamma$"]
    labels = ["(A)", "(B)", "(C)", "(D)"]
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
        plot_scatter(x, y, ax=ax[0][i],
                     ylim=[-1, 1], alpha=0.5, label="W", color="k")
        plot_scatter(x, z, c=y, ax=ax[0][i], cmap=cmap,
                     ylim=[-1, 1], alpha=0.7, ticks=[0, 1],
                     label="C", colorbar=colorbar)

        ax[0][i].set_title(titles[i], fontsize=16)
        ax[0][i].set_yticks([-1, 0, 1])
        ax[0][i].set_xticks([0, max_distance])
        ax[0][i].set_xlabel("D", fontsize=16, labelpad=-15)
        ax[0][i].set_ylabel("C, W", fontsize=16, labelpad=-5)
        ax[0][i].tick_params(labelsize=16)
        ax[0][i].legend(fontsize=12, frameon=False, loc="upper right")

        plot_scatter(y, z, c=x, ax=ax[1][i], alpha=0.5,
                     cmap=cmap, colorbar=colorbar, label="D",
                     vmin=0, vmax=max_distance, ticks=[0, max_distance])  # cbar_label="distance [mm]"

        slope, fitted = fit_line(y, z)
        ax[1][i].plot()
        ax[1][i].plot(y, fitted, lw=1,
                      c="gray", ls="--",
                      label="a=%.3f" % slope)
        slopes.append(slope)

        ax[1][i].set_ylabel("C", fontsize=16, labelpad=-5)
        ax[1][i].tick_params(labelsize=16)
        ax[1][i].set_xticks([0, 1])
        ax[1][i].set_ylim([-1, 1])
        ax[1][i].set_yticks([-1, 0, 1])
        # ax[1][i].legend(fontsize=12, frameon=False, loc="upper right")
        ax[1][i].set_xlabel("W", fontsize=16, labelpad=-10)

    f = np.load(join(data_path, "npz", "distance-euclidean.npz"))
    ax1[0].plot(f["nu"], f["d"], marker="o", color="b")
    ax1[0].set_xlabel(r"$\nu_0$", fontsize=16, labelpad=-5)
    ax1[0].set_ylabel("D", fontsize=16)
    ax1[0].tick_params(labelsize=16)
    ax1[0].set_xlim(np.min(NU)-1, np.max(NU)+1)

    # ax1[1].bar(range(len(nu)), y_bar, align='center', color="blue")
    ax1[1].bar(range(len(nu)), slopes, align='center', color="gray")
    ax1[1].tick_params(labelsize=14)
    ax1[1].set_xlabel(r"$\nu_0$", fontsize=16)
    ax1[1].set_ylabel(r"$\langle C \rangle$", fontsize=16)
    ax1[1].set_xticklabels([""] + titles, fontsize=18)
    # ax1[1].set_yticks([0, 0.3])

    for ii in range(2):
        ax[ii][0].text(-0.25, 0.8,
                       labels[ii],
                       fontsize=18,
                       transform=ax[ii][0].transAxes)

    ax1[0].text(-0.08, 0.7, labels[2], fontsize=18,
                transform=ax1[0].transAxes)
    ax1[1].text(-0.12, 0.7, labels[3], fontsize=18,
                transform=ax1[1].transAxes)

    # fig.savefig("figure3.pdf")
    fig.savefig("figure3_{:.3f}.jpg".format(g), dpi=300)
    plt.close()
