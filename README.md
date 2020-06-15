# epilepsy-msci

This repository contains the modules required to generate a network model of the brain and simulate signal propagation between neurons.

models.py - contains the global class functions for the different network structures and their respective local functions.

analysis.py - contains the functions used to quantitify and visualise the behaviour of a brain network such as generating the degree distribution or plotting the excitement of neurons over time.

network_comparison.py - contains several run functions used to compare different parameters or network structures.

multiprocess_riskcurve.py - is an empty module which executes the module calls from the files above in order to perform parameter variation.

# Packages

The packages required include:
- numpy
- matplotlib
- scipy
- networkx
- itertools
- pickle