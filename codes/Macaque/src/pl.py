import os
import lib
import pylab as pl
import numpy as np
from sys import exit
import pandas as pd
from os.path import join
from config import *
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


def calculate_ave_correlation(sub_name, N):
    for g in G:
        for j in range(len(muMean)):
            subname = str("%.6f-%.6f" % (g*N, muMean[j]))
            subnamefr = str("%.6f-%.1f" %
                            (g, muMean[j] / (2.0 * np.pi) * 1000))
            cor = np.zeros((N, N))
            for ens in range(numEnsembles):
                cor_tmp = np.fromfile(
                    join(
                        "../data/text/c-{:s}-{:s}-{:d}.bin".format(sub_name, subname, ens)),
                    dtype=float, count=-1)
                cor_tmp = np.reshape(cor_tmp, (N, N))
                cor += cor_tmp
            cor /= float(numEnsembles)
            np.savez("../data/npz/{:s}".format(subnamefr), cor=cor)


# ---------------------------------------------------------------------- #
if __name__ == "__main__":

    threshold = 0.2

    pathes = lib.get_pathes(directory=directory)
    for study_name in pathes:
        for network_name in pathes[study_name]:
            sub_name = "{0}_{1}".format(study_name, network_name)
            delaymat_name = "{}_delaymat.txt".format(sub_name)
            connectmat_name = "{}_connectmat.txt".format(sub_name)
            xyz_centers_name = "{}_region_xyz_centers.txt".format(sub_name)
            community_name_l1 = "{}_comm_l1.txt".format(sub_name)
            community_name_l2 = "{}_comm_l2.txt".format(sub_name)

            adj = np.loadtxt(os.path.join(directory,
                                          study_name,
                                          network_name,
                                          connectmat_name))
            N = adj.shape[0]
            calculate_ave_correlation(sub_name, N)
