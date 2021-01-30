import os
import sys
import igraph
import random
import numpy as np
import pylab as pl
import jpype as jp
import networkx as nx
from scipy import signal
from scipy.stats.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import *
# -------------------------------------------------------------------#


def get_pathes(directory="dataset"):

    study_names_p = [f.path for f in os.scandir(directory) if f.is_dir()]
    pathes = {}

    for study_name_p in study_names_p:
        network_names_p = [f.path for f in os.scandir(
            study_name_p) if f.is_dir()]
        network_list = []
        for network_name_p in network_names_p:
            network_list.append(network_name_p.split("/")[2])
        pathes[study_name_p.split("/")[1]] = network_list

    # for i in pathes:
    #     print(i, pathes[i], "\n\n")
    return pathes
# -------------------------------------------------------------------#

# def get_net_directories(study_name=""):

#     study_names_p = [f.path for f in os.scandir(study_name) if f.is_dir()]
#     pathes = {}

#     for study_name_p in study_names_p:
#         network_list = []
#         for network_name_p in network_names_p:
#             network_list.append(network_name_p.split("/")[2])
#         pathes[study_name_p.split("/")[1]] = network_list

#     # for i in pathes:
#     #     print(i, pathes[i], "\n\n")
#     return pathes
# -------------------------------------------------------------------#


def walktrap(adj, steps=5, label="_"):
    conn_indices = np.where(adj)
    weights = adj[conn_indices]
    edges = list(zip(*conn_indices))
    G = igraph.Graph(edges=edges, directed=False)
    comm = G.community_walktrap(weights, steps=steps)
    communities = comm.as_clustering()
    # print comm
    # print("%s number of clusters = %d " % (
    #     label, len(communities)))
    # print "optimal count : ", comm.optimal_count
    return communities
# -------------------------------------------------------------------#


def multilevel(data):
    conn_indices = np.where(data)
    weights = data[conn_indices]
    edges = list(zip(*conn_indices))
    G = igraph.Graph(edges=edges, directed=False)
    G.es['weight'] = weights
    comm = G.community_multilevel(weights=weights, return_levels=False)
    return comm
# -------------------------------------------------------------------#


def communities_to_file(adj, method="walktrap",
                        filename1="File1.txt",
                        filename2="File2.txt",
                        steps=5):

    def to_file(comm, file_name):
        f = open(file_name, "w")
        for c in comm:
            for j in c:
                f.write("%5d" % j)
            f.write("\n")
        f.close()

    if method == "multilevel":
        _comm1 = multilevel(adj)
    else:
        _comm1 = walktrap(adj, steps=steps)

    comm1 = []
    for c in _comm1:
        tmp = []
        if len(c) > 1:
            for j in c:
                # print(j, end=' ')
                tmp.append(j)
            comm1.append(tmp)

    # print comm1.membership
    to_file(comm1, filename1)
    n = adj.shape[0]
    comm2 = [range(n//2), range(n//2, n)]
    to_file(comm2, filename2)

    return len(comm1)
# -------------------------------------------------------------------#


def calculate_NMI(comm1, comm2):
    '''Compares two community structures using normalized
    mutual information as defined by Danon et al (2005)'''

    nmi = igraph.compare_communities(
        comm1, comm2, method='nmi', remove_none=False)
    return nmi

# -------------------------------------------------------------------#


def display_time(time):
    ''' '''

    hour = int(time/3600)
    minute = int((int(time % 3600))/60)
    second = time-(3600.*hour+60.*minute)
    print("Done in %d hours %d minutes %09.6f seconds"
          % (hour, minute, second))
# -------------------------------------------------------------------#


def binarize(data, threshold):
    data = np.asarray(data)
    upper, lower = 1, 0
    data = np.where(data >= threshold, upper, lower)
    return data
# -------------------------------------------------------------------#


def imshow_plot(data, fname='R', cmap='afmhot',
                figsize=(5, 5),
                vmax=None, vmin=None,
                extent=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = pl.figure(100, figsize=figsize)
    pl.clf()
    ax = pl.subplot(111)
    im = ax.imshow(data, interpolation='nearest', cmap=cmap,
                   vmax=vmax, vmin=vmin, extent=extent)
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
    pl.savefig(fname, dpi=150)
    pl.close()

# ---------------------------------------------------------------------- #


def reorder_nodes(C, communities):

    # reordering the nodes:
    N = C.shape[0]
    n_comm = len(communities)

    nc = []
    for i in range(n_comm):
        nc.append(len(communities[i]))

    # new indices of nodes-------------------------------------------

    newindices = []
    for i in range(n_comm):
        newindices += communities[i]
    # --------------------------------------------------------------

    reordered = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            reordered[i, j] = C[newindices[i], newindices[j]]

    return reordered
# ---------------------------------------------------------------------- #


def read_from_file(filename):
    """
    Generate lists using data from file.

    :param filename:
    :return: list of data (list of list)
    """
    DATA = []
    with open(filename, "r") as datafile:
        lines = datafile.readlines()
        for line in lines:
            data = line.split()
            data = [int(i) for i in data]
            DATA.append(data)

    return DATA
# ---------------------------------------------------------------------- #


def determine_frac(A, thr):
    """
    count number of elements in A > thr
    """
    n = len(A[A > thr])
    nodes = A.shape[0]
    n_edges = nodes * (nodes - 1)

    frac = n / float(n_edges)

    return frac

# --------------------------------------------------------------#


def plot_connectome(adj, communities, omega, azim=90, elev=90, fname="C"):

    from mpl_toolkits.mplot3d import Axes3D

    fig = pl.figure(figsize=(6, 5.9))
    ax = fig.gca(projection='3d')

    L = np.genfromtxt("../src/networks/xyz.txt")
    l = np.delete(L, 37, 0)
    N = adj.shape[0]

    # colors = ['red', 'firebrick', 'gold',
    #           'chartreuse', 'darkgreen',
    #           'darkcyan', 'deepskyblue',
    #           'blue', 'm', 'k', 'darkorange']
    colors = pl.cm.brg(np.linspace(0, 1, 20))
    # print "number of communities = ", len(communities)
    # print communities

    nodes_on = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j] > 0:
                nodes_on.extend([i, j])
                xline = [l[i, 0], l[j, 0]]
                yline = [l[i, 1], l[j, 1]]
                zline = [l[i, 2], l[j, 2]]
                ax.plot3D(xline, yline, zline, 'grey', lw=0.5)

    nodes_on = list(set(nodes_on))
    nodes_on.sort()

    ax.plot(l[:, 0], l[:, 1], l[:, 2],
            'o', markersize=15, c="royalblue", alpha=0.4)

    for i in range(len(communities)):
        nodes = common_member(nodes_on, communities[i])
        if nodes:
            nodes = list(nodes)
            ax.plot(l[nodes, 0],
                    l[nodes, 1],
                    l[nodes, 2],
                    'o',
                    # c=colors[i],
                    label=str(i+1),
                    markersize=15,
                    alpha=1.0)
    ax.text(1, 1, 1, str(r"$\nu_0=$%.0f Hz" % to_hz(omega)),
            fontsize=15, transform=ax.transAxes)

    # pl.legend(frameon=False, loc='center right')
    ax.view_init(azim=azim, elev=elev)
    ax.axis('off')
    pl.tight_layout()
    pl.savefig(fname+".png", dpi=150,
               bbox_inches='tight',
               pad_inches=0.01)
    pl.close()
# --------------------------------------------------------------#


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return a_set & b_set
    else:
        return None


def to_hz(x):
    return x/(2.0*np.pi)*1000.

# ------------------------------------------------------------------#


def frequencyAnalysis(g, omega, numNodes, t, xlim=None,
                      ylim1=None,
                      **kwargs):

    pl.style.use('ggplot')
    subname = str("%.6f-%.6f" % (g, omega))

    x = np.fromfile("text/Coor-" + subname + ".bin", dtype=float, count=-1)
    numSteps = int(len(x) / numNodes)
    x = np.reshape(x, (numNodes, numSteps))

    fig, ax = pl.subplots(2, figsize=(8, 5))

    ax[0].plot(t, x[0][:], color="red", alpha=0.7, lw=0.5)
    ax[0].plot(t, x[17][:], color="blue", alpha=0.7, lw=0.5)

    if xlim:
        ax[0].set_xlim(xlim)

    dt = t[1]-t[0]
    fs = 1.0 / dt
    # Sxx = 0.0
    # selectedNodes = range(0, numNodes, 2)
    # for i in selectedNodes:
    #     f, t, sxx = signal.spectrogram(x[i][:], fs,
    #                                    noverlap=512,
    #                                    nperseg=2048)
    #     Sxx += sxx

    # p = ax[1].pcolormesh(t, f, Sxx / len(selectedNodes),
    #                      cmap="afmhot", **kwargs)
    # cbar = fig.colorbar(p, ax=ax[1])
    # ax[1].set_ylabel('Frequency [Hz]')
    # ax[1].set_xlabel('Time [sec]')
    # if ylim1:
    #     ax[1].set_ylim(ylim1)
    # cbar.ax.set_ylabel("amplitude #")

    # last line is frequencies
    c = np.loadtxt("text/F-" + subname + ".txt")
    f = c[-1, :] * 1000
    im = ax[1].imshow(np.log10(c[:-1, :]), origin="lower", cmap="afmhot",
                      extent=[np.min(f), np.max(f), 0, numNodes])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax, ax=ax[1])
    ax[1].grid(False)

    ax[1].set_title(r"$f_0 = %.0f$ [Hz]" %
                    (omega / (2.0 * np.pi) * 1000))
    ax[1].set_xlabel("frequency [Hz]", fontsize=14)
    ax[1].set_ylabel("Nodes #", fontsize=14)
    ax[1].tick_params(labelsize=14)
    pl.tight_layout()
    pl.savefig("fig/f-" + subname + ".png")
    pl.close()
    # pl.show()
# ------------------------------------------------------------------#


def hist_statistic(x, y, nbins=None):

    assert (len(x.shape) == 1)
    assert (len(x) == len(y))

    if nbins is None:
        nbins = int(np.sqrt(len(x)))

    npoints = len(x)
    arg = np.argsort(x)
    xs = x[arg]  # sorted x
    ys = y[arg]  # sorted y with x
    bins = np.linspace(xs[0], xs[-1], nbins)

    temp = []
    count = 0
    bin_index = 1
    std = np.zeros(nbins)
    mean = np.zeros(nbins)
    counts = np.zeros(nbins)

    for j in range(npoints):

        if xs[j] < bins[bin_index]:
            temp.append(ys[j])
            count += 1
        else:
            mean[bin_index - 1] = np.mean(temp)
            std[bin_index - 1] = np.std(temp)
            counts[bin_index - 1] = count
            count = 1
            temp = [ys[j]]
            if bin_index < (nbins - 1):
                bin_index += 1

    bin_width = 0.5*(bins[1]-bins[0])
    bin_centers = bins + bin_width

    return bin_centers, counts/npoints, mean, std
