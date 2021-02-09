import os
import sys
import lib
import pylab as plt
import numpy as np
import networkx as nx
from os.path import join
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from colormaps import parula

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
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    Dis = L[L>1]

    for i in range(len(NU)):

        subnamefr = str("%.6f-%.1f" % (g, NU[i]))
        cor = np.load(join(data_path, "npz",
                           "{:s}.npz".format(subnamefr)))['cor']
        cor = cor.reshape(-1)

        # x = np.array([nu[i]] * len(cor[L > 1]))
        x = np.array([NU[i]] * len(cor[L>1]))

        plot_scatter(x, cor[L>1], c=adj_mat[L>1], ax=ax[0],
                     s=1, colorbar=True, cmap=cmap,
                     cbar_label="weights",
                     xlabel=r"$\nu_0$",
                     ylabel=r"C",
                     ylim=[-1, 1], alpha=0.5,
                     xlim=[np.min(nu), np.max(nu)],
                     vmax=1, vmin=0,
                     bar_ticks=[0, 1])
        plot_scatter(x, cor[L>1], c=Dis, ax=ax[1],
                     s=2, colorbar=True, cmap=cmap,
                     cbar_label="distances",
                     xlabel=r"$\nu_0$",
                     ylabel=r"C",
                     xlim=[np.min(nu), np.max(nu)],
                     ylim=[-1, 1],
                     alpha=0.3,
                     bar_ticks=[0, 50, 100, 150],
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


def calculate_cor_WD(g, network_subname,
                     d_step=10, w_step=4):

    dist_mat = np.loadtxt(join(data_path, distancemat_name))
    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat / np.max(adj_mat) * 100
    D = range(0, max_distance, d_step)
    W = np.arange(0, 101, w_step)

    for j in range(len(nu)):
        cor_mean = np.zeros((len(W), len(D)))
        cor_std = np.zeros_like(cor_mean)
        n_edges = np.zeros_like(cor_mean)
        subnamefr = str("%.6f-%.1f" % (g, nu[j]))
        cor = np.load("{:s}/npz/{:s}.npz".format(data_path,
                                                 subnamefr))['cor']
        for d in range(len(D)-1):
            low_d, high_d = D[d], D[d+1]
            filt_length = lib.filter_matrix(dist_mat, low_d, high_d)

            for w in range(len(W)-1):
                low_w, high_w = W[w], W[w+1]
                filt_weight = lib.filter_matrix(adj_mat, low_w, high_w)
                intersection = lib.find_intersection(filt_length,
                                                     filt_weight)
                Graph = nx.from_numpy_array(intersection)
                edges = list(Graph.edges())

                if len(edges) > 0:
                    tmp = []
                    for edge in edges:
                        tmp.append(cor[edge[0], edge[1]])

                    cor_mean[w, d] = np.mean(tmp)
                    cor_std[w, d] = np.std(tmp)
                    n_edges[w, d] = len(edges)

        np.savez("{:s}/npz/cor_wd_{:.1f}".format(data_path, nu[j]),
                 W=W,
                 D=D,
                 cor=cor_mean,
                 #  std=cor_std,
                 #  n_edges=n_edges,
                 )


def plot_WD(g, ax, cmap="seismic", colorbar=False):

    def add_colorbar(ax, ticks=[-1, 0, 1], im=None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ax=ax, ticks=ticks)
        cbar.ax.tick_params(labelsize=14)

    for ii in range(len(nu)):
        subnamefr = str("%.6f-%.1f" % (g, nu[ii]))
        data = np.load(join(data_path, "npz",
                            "cor_wd_{:.1f}.npz".format(nu[ii])))
        cor_mean = data["cor"]
        D = data["D"]

        im0 = ax[ii].imshow(cor_mean.T, cmap=cmap, aspect="auto",
                            vmax=1, vmin=-1, origin="lower",
                            #    interpolation="bilinear",
                            extent=[0, 1, 0, np.max(D)])
        if ii == 4:
            add_colorbar(ax[ii], im=im0)

        # titles = ["cor, f={:.1f} Hz".format(nu[j]), "std", "num edges"]
        ax[ii].set_xlabel(r"W", fontsize=14, labelpad=-10)
        ax[ii].set_ylabel(r"D", fontsize=14, labelpad=-10)
        # ax.tick_params(labelsize=14)
        ax[ii].set_xticks([0, 1])
        # ax[ii].set_yticks([0, 20, 40])
        # ax.text(-0.2, 0.9, "(b)", fontsize=16, transform=ax.transAxes)

# ---------------------------------------------------------------------- #


if __name__ == "__main__":

    N = 65
    max_distance = 160
    network_subname = "Human_65"
    delaymat_name = "{}_delaymat.txt".format(network_subname)
    distancemat_name = "{}_distancemat.txt".format(network_subname)
    connectmat_name = "{}_connectmat.txt".format(network_subname)
    xyz_centers_name = "{}_region_xyz_centers.txt".format(network_subname)

    nu = [5, 11, 23, 35, 61]
    mu = [(2.0 * np.pi * i / 1000.0) for i in nu]
    NU = np.arange(1, 100, 2)
    data_path = "../data"
    g = float(sys.argv[1])

    # for i in NU:
    #     fig, ax = plt.subplots(1)
    #     subnamefr = str("%.6f-%.1f" % (g, i))
    #     cor = np.load("{:s}/npz/{:s}.npz".format(data_path,
    #                                              subnamefr))['cor']
    #     plot_cor(cor, ax, colorbar=True)
    #     ax.set_title(str(i), fontsize=16)
    #     plt.savefig(join("fig", str(i)+".png"))
    #     plt.close()
    

    calculate_cor_WD(g, network_subname, d_step=8, w_step=4)

    fig = plt.figure(figsize=(10, 6))
    ax1, gs1 = make_grid(nrows=1, ncols=5, left=0.05, right=0.96,
                         bottom=0.68, top=0.93,
                         hspace=0.1, wspace=0.31)
    ax2, gs2 = make_grid(nrows=1, ncols=5, left=0.05, right=0.96,
                         bottom=0.40, top=0.65,
                         hspace=0.1, wspace=0.31)
    ax3, gs3 = make_grid(nrows=1, ncols=2, left=0.05, right=0.93,
                         bottom=0.1, top=0.35,
                         hspace=0.1, wspace=0.3)
    ax = [ax1, ax2, ax3]
    gs = [gs1, gs2, gs3]

    adj_mat = np.loadtxt(join(data_path, connectmat_name))
    adj_mat = adj_mat.reshape(-1) / np.max(adj_mat)
    L = np.loadtxt(join(data_path, distancemat_name)).reshape(-1)
    cmap = plt.cm.get_cmap('plasma_r')
    titles = [r"$\theta$", r"$\alpha$",
              r"$\beta$", r"$\gamma$", "high-$\gamma$"]
    labels = ["(A)", "(B)", "(C)", "(D)"]

    colorbar = False
    for i in range(len(nu)):
        if i == 4:
            colorbar = True

        subnamefr = str("%.6f-%.1f" % (g, nu[i]))
        cor = np.load("{:s}/npz/{:s}.npz".format(data_path,
                                                 subnamefr))['cor']
        plot_cor(cor, ax[0][i], colorbar=colorbar)
        ax[0][i].axvline(x=N//2, ls='--', color='gray')
        ax[0][i].axhline(y=N//2-1, ls="--", color='gray')
        cor = cor.reshape(-1)
        ax[0][i].set_title(titles[i], fontsize=16)
    plot_WD(g, ax[1], cmap="seismic")
    plot_verical_C_freq(g, ax[2], nu=NU, cmap=parula)

    for ii in range(2):
        ax[ii][0].text(-0.25, 0.8, labels[ii], fontsize=16,
                       transform=ax[ii][0].transAxes)
    ax[2][0].text(-0.12, 0.8, labels[2], fontsize=16,
                  transform=ax[2][0].transAxes)
    ax[2][1].text(-0.12, 0.8, labels[3], fontsize=16,
                  transform=ax[2][1].transAxes)

    fig.savefig("figur2_{:.3f}.jpg".format(g), dpi=300)
    plt.close()
