import os
import igraph
import numpy as np
import pandas as pd
import pylab as plt
import networkx as nx
import scipy.io as sio
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
        ax.text(-0.2, 0.93, label,
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

    ax.set_xticklabels(labels, rotation='vertical', fontsize=11)
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=11)
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
    ax.legend(frameon=False, loc='upper right', fontsize=14)
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
        ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=11)
# -------------------------------------------------------------------


def load_distances(filename):

    dfs = pd.read_excel(filename)
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


def remove_isolated_nodes(adj):
    from copy import deepcopy
    c = deepcopy(adj)
    c = c[~(c == 0).all(1)]
    c = np.transpose(c)
    c = c[~(c == 0).all(1)]
    c = np.transpose(c)

    return c

# -------------------------------------------------------------------


if __name__ == "__main__":

    df_dist, source_dist, target_dist = load_distances(
        join("data", 'PNAS_2013_Distance_Matrix.xlsx'))

    data_path = "data"
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

    G = nx.DiGraph()
    for i in range(n_links):
        if dfs['target'][i] in source_labels:
            G.add_edge(dfs['source'][i], dfs['target'][i],
                       FLN=dfs['FLN'][i])

    edges = G.edges()
    for e in edges:
        G[e[0]][e[1]]['distance'] = df_dist[e[0]][e[1]]

    print(nx.info(G))

    FLN = nx.to_numpy_array(G, weight="FLN")
    distances = nx.to_numpy_array(G, weight='distance')


    communities = walktrap(FLN, steps=4)
    newindices = []
    n = max(communities.membership) + 1
    for i in range(n):
        newindices += communities[i]
    labels_r = np.asarray(source_labels)[newindices]
    distances_r = reordering_nodes(communities, distances)


    clusters = []
    for i in range(len(communities)):
        clusters.append(len(communities[i]))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plot_value_distribution(distances_r, clusters, ax=ax[1],
                            nbins=50, label="(B)", density=True,
                            bin_range=(1, 60))
    ax[1].set_xlabel("Disnatace [mm]")
    


    mat_cont = sio.loadmat("data/Human_66.mat")
    C = mat_cont['C']
    L = mat_cont['L']
    adj = remove_isolated_nodes(C)
    L = np.delete(L, 37, 0)
    L = np.delete(L, 37, 1)
    N = adj.shape[0]
    comm = walktrap(adj, steps=4)
    n_comm = max(comm.membership) + 1
    # reordering the nodes:
    clusters = []
    for i in range(n_comm):
        clusters.append(len(comm[i]))
    # new indices of nodes-------------------------------------------
    newindices = []
    for i in range(n_comm):
        newindices += comm[i]

    L_r = reordering_nodes(comm, L)

    plot_value_distribution(
        L_r,
        clusters,
        ax=ax[0],
        nbins=50,
        xlim=[0, 160],
        xticks=[0, 50, 100, 150],
        bin_range=(1, 160),
        density=True,
        label="(A)")
    ax[0].set_ylabel("density")
    ax[0].set_xlabel("distance [mm]")
    ax[1].set_xlabel("distance [mm]")
    

    plt.tight_layout()
    plt.savefig("data/distances.jpg", dpi=150)
    plt.close()


# print(dfs['TARGET'][1000])
# sources = dfs['TARGET'].to_numpy()
# target = dfs['SOURCE'].to_numpy()
# fln = dfs['FLN'].to_numpy()
# sln = dfs['SLN'].to_numpy()
# target_labels = [
#     'V1', 'V2', 'V4', 'MT', '7m', '7A', '7B', '2', 'ProM', 'STPr', 'STPi', 'STPc', 'PBr',
#     'Tepd', 'TEO', 'F1', '5', 'F2', 'F7', '8B', '10', '46d', '9/46d', '9/46v', '8m',
#     '8l', 'F5', '24c', 'DP', 'V3', 'V3A', 'V4t', 'Pro.St.',
#     '7op', 'LIP', 'VIP', 'MIP', 'PIP', 'AIP', 'V6', 'V6A', 'TPt', 'PGa', 'IPa', 'FST', 'MST',
#     'TEOm', 'PERIRHINAL', 'TEad', 'TEav', 'TEpv', 'Tea/ma', 'Tea/mp', 'ENTORHINAL', 'TH/TF',
#     'SUBICULUM', 'TEMPORAL_POLE', 'CORE', 'MB', 'LB', 'PBc', 'INSULA', 'Gu', 'SII', '1', '3',
#     '23', '24a', '24b', '24d', '29/30', '31', '32', 'F3', 'F6', 'F4', '9', '46v', '8r', '45B',
#     '45A', '44', 'OPRO', 'OPAI', '11', '14', '25', '12', '13', 'PIRIFORM', 'Pi',
# ]
# source_labels = [
#     'V1', 'V2', 'V4', 'MT', '7m', '7A', '7B', '2', 'ProM', 'STPr', 'STPi', 'STPc', 'PBr',
#     'Tepd', 'TEO', 'F1', '5', 'F2', 'F7', '8b', '10', '46d', '9/46d', '9/46v', '8m', '8l',
#     'F5', '24c', 'DP',
# ]
