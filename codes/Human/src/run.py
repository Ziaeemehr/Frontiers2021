import os
import lib
import numpy as np
import networks
from os import system
from time import time
from os.path import join
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
from config import *

# ------------------------------------------------------------------#
directories = ["../data", "../data/text",
               "../data/fig", "../data/text/networks"]
for d in directories:
    if not os.path.exists(d):
        os.makedirs(d)
# ------------------------------------------------------------------#


def runCommand(arg):
    command = "{0} {1} {2} {3} {4} {5} {6} {7} {8} \
               {9} {10} {11} {12}".format(*arg)
    system("./prog " + command)
# ------------------------------------------------------------------#


def batch_run():
    arg = []
    pathes = lib.get_pathes(directory=directory)
    for study_name in pathes:
        for network_name in pathes[study_name]:
            sub_name = "{0}_{1}".format(study_name, network_name)
            delaymat_name = "{}_delaymat.txt".format(sub_name)
            distancemat_name = "{}_distancemat.txt".format(sub_name)
            connectmat_name = "{}_connectmat.txt".format(sub_name)
            xyz_centers_name = "{}_region_xyz_centers.txt".format(sub_name)
            community_name_l1 = "{}_comm_l1.txt".format(sub_name)
            community_name_l2 = "{}_comm_l2.txt".format(sub_name)

            xyz = np.loadtxt(os.path.join(directory,
                                          study_name,
                                          network_name,
                                          xyz_centers_name))

            adj = np.loadtxt(os.path.join(directory,
                                          study_name,
                                          network_name,
                                          connectmat_name))
            adj = adj/np.max(adj)
            N = adj.shape[0]

            # delay_mat = euclidean_distances(xyz) / VELOCITY
            if USE_FIXED_DELAY[0]:
                delay_mat = (np.ones((N, N), dtype=int) -
                             np.diag([1] * N)) * USE_FIXED_DELAY[1]
            else:
                delay_mat = np.loadtxt(os.path.join(directory,
                                                    study_name,
                                                    network_name,
                                                    distancemat_name))/VELOCITY
            ncluster = lib.communities_to_file(
                adj, "walktrap",
                filename1=join("../data/text/networks", community_name_l1),
                filename2=join("../data/text/networks", community_name_l2),
                steps=CommunitySteps)
            np.savetxt(join("../data/text/networks", delaymat_name),
                       delay_mat, fmt="%18.9f")
            if USE_BINARIZED:
                adj = lib.binarize(adj, 1e-8)
            np.savetxt(join("../data/text/networks", connectmat_name),
                       adj, fmt="%18.9f")

            for g in G:
                for m in muMean:
                    arg.append([N,
                                t_transition * N,
                                t_simulation * N,
                                g * N,
                                m,
                                muStd,
                                noiseAmplitude,
                                sub_name,
                                ncluster,
                                numEnsembles,
                                PRINT_CORRELATION,
                                PRINT_COORDINATES,
                                dt])
    Parallel(n_jobs=n_jobs)(
        map(delayed(runCommand), arg))
# ------------------------------------------------------------------#


if __name__ == "__main__":

    start = time()
    seed = 136
    batch_run()
    lib.display_time(time() - start)

    # adjMatrix = np.loadtxt("networks/C65.txt")
    # delayMatrix = np.loadtxt("networks/L65.txt")/VELOCITY
    # adjMatrix = adjMatrix / np.max(adjMatrix)

    # nclusters = lib.communities_to_file(adjMatrix,
    #                                     "walktrap",
    #                                     steps=CommunitySteps)
    # np.savetxt("networks/C.txt", adjMatrix, fmt="%15.6f")
    # np.savetxt("networks/D.txt", delayMatrix, fmt="%15.6f")

    # batchRun(nclusters)

    # adjMatrix = lib.binarize(adjMatrix, 1e-8)
    # print(np.sort(list(set((adjMatrix.reshape(-1))))))

    # print(np.sort(list(set((delayMatrix.reshape(-1))))))
    # delayMatrix = lib.binarize(delayMatrix, 0.01) * 10
