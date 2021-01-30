import sys
import igraph
import colorsys
import random
import numpy as np
import pylab as pl
import networkx as nx
from scipy import signal
from scipy.stats.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def filter_matrix(A, low, high):

    n = A.shape[0]
    assert (len(A.shape) == 2)
    filt = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i, n):
            if (A[i][j] > low) and (A[i][j] < high):
                filt[i][j] = filt[j][i] = 1

    return filt

# ------------------------------------------------------------------#


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
