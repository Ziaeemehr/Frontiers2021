import os
import sys
import lib
import pylab as plt
import numpy as np
import networkx as nx
from os.path import join
from matplotlib.gridspec import GridSpec
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

    return [ax, gs]


def plot_cor(cor, ax,
             colorbar=True,
             cmap="seismic",
             labelsize=14):

    im = ax.imshow(cor, cmap=cmap, vmax=1, vmin=-1, aspect="auto")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.tick_params(labelsize=labelsize)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_verical_C_freq(g, ax=None, nu=None, cmap="plasma_r"):

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    adj_mat = lib.binarize(adj_mat, 1e-8)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    Dis = L

    for i in range(len(nu)):

        subnamefr = str("%.6f-%.1f" % (g, nu[i]))
        cor = np.load(join(data_path, "npz",
                           "{:s}.npz".format(subnamefr)))['cor']
        cor = cor.reshape(-1)

        # x = np.array([nu[i]] * len(cor[L > 1]))
        x = np.array([nu[i]] * len(cor))

        plot_scatter(x, cor, c=adj_mat, ax=ax[0],
                     s=1, colorbar=True, cmap=cmap,
                     cbar_label="weights",
                     xlabel=r"$\nu_0$",
                     ylabel=r"C",
                     ylim=[-1, 1], alpha=0.5,
                     xlim=[np.min(nu), np.max(nu)],
                     vmax=1, vmin=0,
                     bar_ticks=[0, 1])
        plot_scatter(x, cor, c=Dis, ax=ax[1],
                     s=2, colorbar=True, cmap=cmap,
                     cbar_label="distances",
                     xlabel=r"$\nu_0$",
                     ylabel=r"C",
                     xlim=[np.min(nu), np.max(nu)],
                     ylim=[-1, 1],
                     alpha=0.3,
                     bar_ticks=[0, 25, 50],
                     vmin=0, vmax=max_distance)
        ax[0].set_yticks([-1, 0, 1])
        ax[1].set_yticks([-1, 0, 1])
        # ax[0].text(-0.12, 0.8, "(f)", fontsize=16,
        #            transform=ax[0].transAxes)
        # ax[1].text(-0.12, 0.8, "(g)", fontsize=16,
        #            transform=ax[1].transAxes)


def plot_scatter(x, y, ax,
                 xlabel=None,
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 color=None,
                 alpha=0.4,
                 label=None,
                 title=None,
                 cmap="afmhot",
                 c=None,
                 s=10,
                 colorbar=False,
                 cbar_label=None,
                 vmin=None,
                 vmax=None,
                 bar_ticks=None):

    xl = x.reshape(-1)
    yl = y.reshape(-1)
    im = ax.scatter(xl, yl, s=s, c=c, color=color, alpha=alpha,
                    cmap=cmap, label=label, vmax=vmax, vmin=vmin)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, labelpad=-5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, labelpad=-5)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    if (xlim is None) and (ylim is None):
        ax.margins(x=0.02, y=0.02)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cbar = plt.colorbar(im, cax=cax, ticks=bar_ticks)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=13)

    if title:
        ax.set_title(title)


if __name__ == "__main__":

    network_subname = "Monkey_29"
    delaymat_name = "{}_delaymat.txt".format(network_subname)
    distancemat_name = "{}_distancemat.txt".format(network_subname)
    connectmat_name = "{}_connectmat.txt".format(network_subname)
    xyz_centers_name = "{}_region_xyz_centers.txt".format(network_subname)
    max_distance = 55

    N = 29
    # nu = [11, 21, 31, 41, 51]
    nu = [3, 11, 23, 35, 51]
    mu = [(2.0 * np.pi * i / 1000.0) for i in nu]
    NU = np.arange(1, 100, 1)
    data_path = "../data"
    g = float(sys.argv[1])

    # for i in range(1,100):
    #     fig, ax = plt.subplots(1)
    #     subnamefr = str("%.6f-%.1f" % (g, i))
    #     cor = np.load("{:s}/npz/{:s}.npz".format(data_path,
    #                                              subnamefr))['cor']
    #     plot_cor(cor, ax, colorbar=True)
    #     ax.set_title(str(i), fontsize=16)
    #     plt.savefig(join("fig", str(i)+".png"))
    #     plt.close()
    # exit(0)

    fig = plt.figure(figsize=(10, 4))
    ax1, gs1 = make_grid(nrows=1, ncols=5, left=0.05, right=0.96,
                         bottom=0.5, top=0.93,
                         hspace=0.1, wspace=0.31)
    # ax2, gs2 = make_grid(nrows=1, ncols=5, left=0.05, right=0.96,
    #                      bottom=0.40, top=0.65,
    #                      hspace=0.1, wspace=0.31)
    ax3, gs3 = make_grid(nrows=1, ncols=2, left=0.05, right=0.93,
                         bottom=0.1, top=0.45,
                         hspace=0.1, wspace=0.3)
    ax = [ax1, ax3]
    gs = [gs1, gs3]

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    cmap = plt.cm.get_cmap('plasma_r')
    titles = [r"$\theta$", r"$\alpha$",
              r"$\beta$", r"$\gamma$", "high-$\gamma$"]
    labels = ["(A)", "(B)"]

    colorbar = False
    for i in range(len(nu)):
        if i == 4:
            colorbar = True

        subnamefr = str("%.6f-%.1f" % (g, nu[i]))
        cor = np.load("{:s}/npz/{:s}.npz".format(data_path,
                                                 subnamefr))['cor']
        plot_cor(cor, ax[0][i], colorbar=colorbar)
        ax[0][i].set_title(titles[i], fontsize=16)

    plot_verical_C_freq(g, ax[1], nu=NU)

    for ii in range(2):
        ax[ii][0].text(-0.25, 0.9, labels[ii], fontsize=16,
                       transform=ax[ii][0].transAxes)
    ax[1][0].text(-0.12, 0.8, labels[1], fontsize=16,
                  transform=ax[1][0].transAxes)

    fig.savefig("figur2_{:.3f}.jpg".format(g), dpi=300)
    plt.close()
