import numpy as np
import lib

# N = 110
G = [0.05]  # multipy by N
dt = 0.02
muStd = 0.01
# nu = list(range(3, 63, 3))
# nu = [4.0, 9.0, 20.0, 35.0]
nu = np.arange(1, 100, 1)
# nu_0 = [4, 10, 22, 35]  # for plotting the connections
# nu_0 = [11, 21, 31, 41]

muMean = [(2.0 * np.pi * i / 1000.0) for i in nu]
noiseAmplitude = 0.0  # ! noise inactivated
CommunitySteps = 4
numEnsembles = 20
t_transition = 200.0  # multipy by N
t_simulation = t_transition + 200.0  # multipy by N
CONN_NAME = "C"
DELAY_NAME = "D"
PRINT_CORRELATION = 1
PRINT_COORDINATES = 0
n_jobs = 4
NUM_THREADS = 1
VELOCITY = 5.0
USE_BINARIZED = True

directory = "networks"

if __name__ == "__main__":

    pathes = lib.get_pathes(directory="networks")
    print(pathes)
    study_name = pathes.keys()
# for study_name in pathes:
# for network_name in pathes[study_name]:
