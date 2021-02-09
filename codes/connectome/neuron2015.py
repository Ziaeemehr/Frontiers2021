import os
import igraph
import numpy as np
import pandas as pd
import pylab as plt
import networkx as nx
from os.path import join
from numpy import min, max
from collections import Counter
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

if not os.path.exists("data"):
    os.makedirs("data")
# -------------------------------------------------------------------


def check_edge_between_clusters(e, nodes):
    "check if given edge is between clusters."
    # print e[0], e[1]
    n_clusters = len(nodes)
    for i in range(n_clusters):
        if ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
                ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
            return True

    return False


def divide_edges_in_between_clusters(
        mat,
        communities):
    """
    divide edges in and between clusters.
    print two matrix.
    """
    n_comm = len(communities)
    N = mat.shape[0]

    len_communities = []
    for i in range(n_comm):
        len_communities.append(len(communities[i]))

    nodes = nodes_of_each_cluster(len_communities)
    G = nx.from_numpy_matrix(mat)
    print(nx.info(G))
    # G.remove_edges_from(G.selfloop_edges())
    G.remove_edges_from(nx.selfloop_edges(G))

    print(nx.info(G))

    edges = G.edges()
    list_in = []
    list_between = []

    for e in edges:
        info = check_edge_between_clusters(e, nodes)
        if info:
            list_between.append(mat[e[0], e[1]])
        else:
            list_in.append(mat[e[0], e[1]])

    return list_in, list_between


def nodes_of_each_cluster(len_communities):
    "return nodes of each cluster as list of list"

    n_comm = len(len_communities)
    nodes = []
    a = [0]
    for i in range(n_comm):
        a.append(a[i]+len_communities[i])
        nodes.append(range(a[i], a[i+1]))
    return nodes


def plot_value_distribution(
        mat,
        clusters,
        ax,
        nbins=100,
        label=None,
        thr=1e-8,
        ** kwargs):

    communities = nodes_of_each_cluster(clusters)
    arrIn, arrOut = divide_edges_in_between_clusters(mat, communities)

    num_of_points = (np.abs(mat) > thr).sum()

    plot_hist(arrIn, num_of_points, ax, bins=nbins, color='red',
              label="intra", **kwargs)
    plot_hist(arrOut, num_of_points, ax, bins=nbins, color='royalblue',
              label="inter", **kwargs)
    if label is not None:
        ax.text(-0.17, 0.93, label,
                fontsize=16, transform=ax.transAxes)


def walktrap(adj, steps=5, directed=True):
    conn_indices = np.where(adj)
    weights = adj[conn_indices]
    edges = list(zip(*conn_indices))
    G = igraph.Graph(edges=edges, directed=directed)
    comm = G.community_walktrap(weights, steps=steps)
    communities = comm.as_clustering()
    print("number of clusters = %d, number of nodes = %d " % (
        len(communities), adj.shape[0]))
    # print "optimal count : ", comm.optimal_count
    return communities


def reordering_nodes(communities, A):
    '''
    reorder the index of given matrix using communities
    '''

    n = max(communities.membership) + 1
    n_nodes = A.shape[0]

    # new indices of nodes
    newindices = []
    for i in range(n):
        newindices += communities[i]

    ordered_A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            ordered_A[i, j] = A[newindices[i], newindices[j]]
    return ordered_A


def plot_rectangles(comm, A, ax=None):

    s = 0
    X = Y = 0
    N = A.shape[0]
    n_comm = len(comm)
    for k in range(n_comm):
        if k > 0:
            s += len(comm[k-1])
            X = s
            Y = s
        ax.add_patch(Rectangle((X-0.5, Y-0.5),
                               len(comm[k]), len(comm[k]),
                               fill=None, lw=1.5, alpha=1,))
# --------------------------------------------------------------#


def plot_matrix(A, ax,
                cmap="afmhot_r",
                title=None,
                labels=None,
                xlabel=None,
                ylabel=None):
    im = ax.imshow(A, origin='lower', cmap=cmap, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(im, ticks=None, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels(labels, rotation='vertical', fontsize=10)
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)
# -------------------------------------------------------------------


def plot_hist(arr, num_of_points, ax, bins='auto', color='red',
              label=None,
              density=False,
              xticks=None,
              yticks=None,
              xlim=None,
              ylim=None,
              xlabel=None,
              ylabel=None,
              xticklabels=None,
              ylogscale=False,
              xlogscale=False,
              bin_range=(-1, 1)):

    hist, bins = np.histogram(arr, bins=bins, range=bin_range)
    if density:
        hist = hist / float(num_of_points)

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align='center',
           width=width,
           alpha=0.7,
           color=color,
           label=label)
    ax.legend(frameon=False, loc='upper left', fontsize=14)
    if ylogscale:
        ax.set_yscale('log')
    if xlogscale:
        ax.set_xscale('log')
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel("Prob", fontsize=12)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=11)
# -------------------------------------------------------------------


def plot_bar(x, y, ax,
             align='edge',
             xticks=None,
             xticklabels=None,
             xlabel=None,
             ylabel=None,
             label=None,
             alpha=0.4,
             width=0.4,
             color="r"):
    ax.bar(x, y,
           align=align,
           width=width,
           alpha=alpha,
           edgecolor="b",
           color=color,
           label=label)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if label is not None:
        ax.legend(loc='upper right')
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=10)
# -------------------------------------------------------------------


def load_distances(filename):

    dfs = pd.read_excel(filename)
    # print(dfs.columns)
    # print(dfs.head())
    source_labels = list(dfs.columns[1:])
    target_labels = list(dfs['Unnamed: 0'])
    del dfs['Unnamed: 0']
    dfs = dfs.fillna(0)
    dfs.index = target_labels

    # could not change label in file
    index = source_labels.index('Tepd')
    source_labels[index] = 'TEpd'
    source_labels = [str(i) for i in source_labels]
    target_labels = [str(i) for i in target_labels]

    # print(dfs.head())
    dfs = dfs.rename(columns=lambda s: str(s), index=lambda s: str(s))
    dfs = dfs.rename(columns={'Tepd': 'TEpd'}, index={'Tepd': 'TEpd'})

    return dfs, source_labels, target_labels
# -------------------------------------------------------------------


if __name__ == "__main__":

    df_dist, source_dist, target_dist = load_distances(
        join("data", 'PNAS_2013_Distance_Matrix.xlsx'))

    data_path = "dataset"
    file_name = "Neuron_2015_Table.xlsx"
    sheetname = "Table S1 DBV23.45 CorrVal"
    dfs = pd.read_excel(join(data_path, file_name))
    dfs = dfs.rename(columns={'TARGET': 'source'})
    dfs = dfs.rename(columns={'SOURCE': 'target'})
    n_links = len(dfs)

    labels = ['source', 'target', 'FLN', 'SLN']
    area_labels = []

    for i in labels[:2]:
        A = dfs[i].to_numpy()
        keys = dict(Counter(A)).keys()
        print(i, len(keys))
        area_labels.append(list(keys))

    source_labels = [str(i) for i in area_labels[0]]
    target_labels = [str(i) for i in area_labels[1]]

    dfs['source'] = [str(i) for i in dfs['source']]
    dfs['target'] = [str(i) for i in dfs['target']]

    if 0:
        np.savetxt("data/TARGETS.txt", target_labels, fmt="%s")
        np.savetxt("data/SOURCES.txt", source_labels, fmt="%s")

    G = nx.DiGraph()
    for i in range(n_links):
        if dfs['target'][i] in source_labels:
            G.add_edge(dfs['source'][i], dfs['target'][i],
                       FLN=dfs['FLN'][i])

    edges = G.edges()
    for e in edges:
        G[e[0]][e[1]]['distance'] = df_dist[e[0]][e[1]]

    print(nx.info(G))

    # edgelist = nx.to_edgelist(G, nodelist=source_labels[0])
    # print(edgelist)

    FLN = nx.to_numpy_array(G, weight="FLN")
    distances = nx.to_numpy_array(G, weight='distance')
    if 0:
        np.savetxt("data/adj29.txt", FLN.T)
        np.savetxt("data/distance29.txt", distances.T)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))

    out_degrees = list(dict(G.out_degree()).values())
    in_degrees = list(dict(G.in_degree()).values())
    indices_in = np.argsort(in_degrees)

    plot_bar(range(len(source_labels)), in_degrees,
             ax=ax[1, 0],
             align='edge',
             xlabel="area",
             ylabel="in/out degree",
             xticklabels=source_labels,
             color='b',
             label="in")
    plot_bar(range(len(source_labels)), out_degrees,
             ax=ax[1, 0],
             align='edge',
             width=-0.4,
             xticks=range(29),
             xlabel="area",
             ylabel="in/out degree",
             xticklabels=source_labels,
             color='r',
             label="out")

    
    hist, bins = np.histogram(
        np.log10(FLN.reshape(-1)), bins=12, range=(-6, 0), density=True)
    center = (bins[:-1] + bins[1:]) / 2
    ax[1, 1].bar(center, hist, width=0.4)

    def gaus(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt, pcov = curve_fit(gaus, center, hist)
    ax[1, 1].plot(center, gaus(center, *popt), 'ro:',
                  label=r'$ \mu, \sigma$=({:.2f},{:.2f})'.format(popt[1], popt[2]))
    ax[1, 1].legend(loc='upper right')
    # print(popt)
    # [ 0.25744595 -3.05082066  1.69455785]

    ax[1, 1].set_xlim(-6, 0)
    ax[1, 1].set_xlabel(r"$log_{10}(FLN)$")
    ax[1, 1].set_ylabel(r"density")

    ax_labels = [["(A)", "(B)"], ["(C)", "(D)"]]
    ax[0][0].text(-0.25,0.85, "(A)",fontsize=14,transform=ax[0][0].transAxes)
    ax[0][1].text(-0.25,0.85, "(B)",fontsize=14,transform=ax[0][1].transAxes)
    ax[1][0].text(-0.25,0.85, "(C)",fontsize=14,transform=ax[1][0].transAxes)
    ax[1][1].text(-0.25,0.85, "(D)",fontsize=14,transform=ax[1][1].transAxes)

    
    communities = walktrap(FLN, steps=4)
    newindices = []
    n = max(communities.membership) + 1
    for i in range(n):
        newindices += communities[i]
    labels_r = np.asarray(source_labels)[newindices]
    adj_r = reordering_nodes(communities, FLN)
    distances_r = reordering_nodes(communities, distances)


    X = np.log10(adj_r.T)
    X[X == -np.inf] = -5.5
    X[X == np.inf] = -5.5
    plot_matrix(X, ax[0,0],
                labels=labels_r,
                xlabel="From area",
                ylabel="To area",
                cmap="jet",
                title="FLN")
    plot_rectangles(communities, adj_r, ax[0,0])

    plot_matrix(distances_r.T, ax[0,1],
                labels=labels_r,
                xlabel="From area",
                ylabel="To area",
                cmap="jet",
                title="Distances")
    plot_rectangles(communities, distances_r, ax[0,1])

    plt.tight_layout()
    plt.savefig("data/adj.jpg", dpi=150)
    plt.close()

    