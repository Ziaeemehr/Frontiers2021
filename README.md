-  This repository contains the codes for reproducing results and figures in the paper: 
-  Frequency-resolved functional connectivity: Role of delay and the strength of connections [Accepted by Frontiers in neural circuits](https://www.researchgate.net/publication/344217594_Frequency-resolved_functional_connectivity_Role_of_delay_and_the_strength_of_connections) 
-  Abolfazl Ziaeemehr [1], Alireza Valizadeh [1, 2]
   -  [1] Department of Physics, Institute of Advanced Studies in Basic Sciences (IASBS), Zanjan, Iran.
   -  [2] School of Biological Sciences, Institute for Research in Fundamental Sciences (IPM), Tehran, Iran.
-  **Abstract**

*The brain functional network extracted from the BOLD signals reveals the correlated activity of the different brain regions, which is hypothesized to underlie the integration of the information across functionally specialized areas. Functional networks are not static and change over time and in different brain states, enabling the nervous system to engage and disengage different local areas in specific tasks on demand. Due to the low temporal resolution, however, BOLD signals do not allow the exploration of spectral properties of the brain dynamics over different frequency bands which are known to be important in cognitive processes. Recent studies using imaging tools with a high temporal resolution has made it possible to explore the correlation between the regions at multiple frequency bands. These studies introduce the frequency as a new dimension over which the functional networks change, enabling brain networks to transmit multiplex of information at any time. In this computational study, we explore the functional connectivity at different frequency ranges and highlight the role of the distance between the nodes in their correlation. We run the generalized Kuramoto model with delayed interactions on top of the brain's connectome and show that how the transmission delay and the strength of the connections, affect the correlation between the pair of nodes over different frequency bands.*

### How to use:

```sh
cd codes/Human
make clean
make
# open config.py and set the proper parameters
python3 run.py
python3 pl.py
# then copy npz files in codes/Human/data/npz to codes/Human/analysis/data/npz
# and plot figures using folders 2, 3 and 4.
```



