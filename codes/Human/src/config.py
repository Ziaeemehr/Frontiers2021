import numpy as np
import lib

# N = 65
G = [0.2]  # multipy by N
dt = 0.02
muStd = 0.05
nu = np.arange(1, 20, 2)

muMean = [(2.0 * np.pi * i / 1000.0) for i in nu]
noiseAmplitude = 0.05  # ! noise activated
CommunitySteps = 4
numEnsembles = 2
t_transition = 10.0  # multipy by N
t_simulation = t_transition + 10.0  # multipy by N
PRINT_CORRELATION = 1
PRINT_COORDINATES = 0
n_jobs = 4
NUM_THREADS = 1
VELOCITY = 5.0
USE_BINARIZED = False
USE_FIXED_DELAY = [False, 10.]

directory = "networks"

if __name__ == "__main__":

    pathes = lib.get_pathes(directory="networks")
    print(pathes)
    study_name = pathes.keys()

# for study_name in pathes:
# for network_name in pathes[study_name]:
