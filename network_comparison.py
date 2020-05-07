# MSci Project
# Snigdha Sen and Sophie Martin
 
from models import RandomBrain, SmallWorldBrain, GrowPrefAttachmentBrain, BarabasiAlbertBrain
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import analysis
import models
import pickle
import networkx as nx
from scipy.interpolate import CubicSpline
from logbin230119 import logbin

# Set random seeds using both methods
np.random.seed(100)
random.seed(100)

def run_network_size_comparison():
    Ns = np.logspace(1, 3, 10)

    pathLengthArray = [[]]*4

    for size in Ns:
        N = int(size)
        print('Running size %d' % N)

        print('Random Brain...')
        randomBrain = RandomBrain(numberOfNodes = N, inhibitoryEdgeProbability=0, numberOfIterations=1000,
        averageDegree=8, neuronThreshold=0, refractoryPeriod=2, fractionToFire=0.005, probabilityOfFiring=0.2, directed=False)

        print('Small World...')
        smallworld = SmallWorldBrain(numberOfNodes = N, inhibitoryEdgeProbability=0, numberOfIterations=1000, 
                rewireProbability=0, averageDegree=8, neuronThreshold=0, refractoryPeriod=2, fractionToFire=0.005, probabilityOfFiring=0.2, directed=False)

        print('BA...')
        baBrain = BarabasiAlbertBrain(numberOfNodes = N, inhibitoryEdgeProbability=0, numberOfIterations=1000, 
                rewireProbability=0, averageDegree=8, neuronThreshold=0, refractoryPeriod=2, fractionToFire=0.005, probabilityOfFiring=0.2, directed=False)
        
        print('Growing Brain...')
        prefattach = GrowPrefAttachmentBrain(numberOfNodes = N, mu=0.5, inhibitoryEdgeProbability=0, numberOfIterations=1000, 
                rewireProbability=0, m=8, vertexNumber = 8, neuronThreshold=0, refractoryPeriod=2, fractionToFire=0.005, probabilityOfFiring=0.2, directed=False)

        pathLengthArray[0] = np.append(nx.average_shortest_path_length(randomBrain.network), pathLengthArray[0])
        pathLengthArray[1] = np.append(nx.average_shortest_path_length(smallworld.network), pathLengthArray[1])
        pathLengthArray[2] = np.append(nx.average_shortest_path_length(baBrain.network), pathLengthArray[2])
        pathLengthArray[3] = np.append(nx.average_shortest_path_length(prefattach.network), pathLengthArray[3])

    #np.save('results/path_lengths', np.array(clusterArray))
    plt.figure()
    for data, name in zip(pathLengthArray, ['Erdos-Renyi', 'Small World', 'Barabasi-Albert', 'Clustering Model']):  
        plt.scatter(Ns, data, label=name)

    plt.legend()
    plt.ylabel('Average Shortest Path Length')
    plt.xlabel('$N$')

def run_c_l_p_variation(N, seed=100, save=True, load=False, plot=True, filename=None):
    np.random.seed(seed)
    random.seed(seed)
    print('Running for seed: ', seed)
    ps = np.logspace(-3, -1, 15)
    ps = np.append(ps, np.linspace(0.15, 1, 15))
    pathlength = []
    ccoeff = []
    if not load:
        for p in ps:
            print(p)
            brain = SmallWorldBrain(numberOfNodes=N, numberOfIterations=1000, rewireProbability=p, refractoryPeriod=1, fractionToFire=1/100000, neuronThreshold=1,
        fireRate=0.1, inhibitoryEdgeMultiplier=0, averageDegree=5, directed=False)
            pathlength.append(nx.average_shortest_path_length(brain.network))
            ccoeff.append(nx.average_clustering(brain.network))

    if save:
        if filename is None:
            np.save('results/clustercoeffs_', np.array(ccoeff))
            np.save('results/pathslengths_', np.array(pathlength))
        else:
            np.save(filename[0], np.array(ccoeff))
            np.save(filename[1], np.array(pathlength)) # filename must be an interable with two filenames: ccoeff, then pathlengths

    if load:
        ccoeff = np.load(filename[0])
        pathlength = np.load(filename[1])

    if plot:
        
        _, ax = plt.subplots(1,1)
        ax2 = ax.twinx()

        a = 10
        b = 20
        c = 14
        d = 16
        sub_axes = plt.axes((.4, .5, .35, .35)) 
        sub_axes.plot(np.array(ps)[a:b], ccoeff[a:b], '.', markersize=8, color='black', label='C')
        sub_axes.set_ylim(0, 0.64)
        sub_axes.set_xlabel('$p_r$', size=16)
        sub_axes.set_ylabel('$C$', size=16)
        sub_axes2 = sub_axes.twinx()
        sub_axes2.set_ylabel('$L$', size=16)
        sub_axes2.plot(np.array(ps)[a:b], pathlength[a:b], '+', markersize=8, color='black', label='L')
        sub_axes.axvspan(np.array(ps)[c], np.array(ps)[d],
                        facecolor='red', alpha=0.5)

        ax.plot(ps, ccoeff, '.', markersize=10, color='black', label='C')
        ax2.plot(ps, pathlength, '+', markersize=10, color='black', label='L')
        ax.axvspan(np.array(ps)[c], np.array(ps)[d],
                        facecolor='red', alpha=0.5)
        ax.plot(figsize = (8,6))
        ax.set_xlabel('$p_r$', size=16)
        ax.set_ylabel('$C$', size=16)
        ax2.set_ylabel('$L$', size=16)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=0, prop={'size': 16})
        
        
        # Code for the ratio

        cs = np.array(ccoeff)
        ls = np.array(pathlength)
        ratio = cs/ls

        cubicspline = CubicSpline(ps, cs/ls)
        a=0.095
        ps_smooth = np.linspace(0.0005, 1, 100)
        plt.figure(figsize=(8,6))
        plt.plot(ps, cs/ls, 'o', color='black', label='Data')
        plt.plot(ps_smooth, cubicspline(ps_smooth), '-', color='black', label='Cubic Spline')
        plt.xlabel('$p_r$', size=16)
        plt.legend(prop={'size': 16})

        plt.tick_params(labelsize=16)
        plt.axvline(a, color='red', alpha=0.8)

        plt.ylabel('$C/L$', size=16)

        return max(ratio)

def calculate_omega():
    ps = np.linspace(0.15, 1, 15)
    ps = np.append(ps, np.logspace(-3, -1, 15))
    omegas = []

    for p in ps:
        print('calculating omega for...', p)
        brain = SmallWorldBrain(numberOfNodes=10000, inhibitoryEdgeProbability=0, numberOfIterations=1000, 
            rewireProbability=p, averageDegree=8, neuronThreshold=1, refractoryPeriod=1, fractionToFire=0.01, probabilityOfFiring=0.2, directed=False)
        
        omegas.append(nx.algorithms.smallworld.omega(brain.network))

    np.save('results/omega_small_world', np.array(omegas))

def plot_edge_cases_ri():

    # Plot edge cases on one graph 

    prop = np.load('results/SW_propagate_2.npy')
    die = np.load('results/SW_dies_out.npy')

    numList_prop = []
    numList_die = []

    for _, ye in zip(np.arange(100), [list(i) for i in prop]):
        numList_prop.append(len(ye))

    for _, ye in zip(np.arange(100), [list(i) for i in die]):
        numList_die.append(len(ye))

    plt.figure()
    plt.plot(np.arange(100), np.array(numList_prop)/max(numList_prop), label='Signal propagates', color='black')
    plt.plot(np.arange(100), np.array(numList_die)/max(numList_die), '--', label='Signal dies', color='black')

    plt.axvline(0, color='gray')
    plt.xlabel('Timestep', size=14)
    plt.tick_params(labelsize=14)
    plt.ylabel('$N_{neurons}/N_{max}$', size=16)
    plt.legend()


def run_c_l_p_k_variation(k, seed, plot=True):
    np.random.seed(seed)
    random.seed(seed)
    print('Running for seed=', seed)
    ps = np.logspace(-3, -1, 15)
    ps = np.append(ps, np.linspace(0.15, 1, 15))
    #kList = [5, 7, 10, 15]
    if plot:
        plt.figure(figsize=(8,6))
        #colors = ['black', 'yellow', 'purple', 'green']

    pathlength = []
    ccoeff = []

    for p in ps:
        print(p)
        brain = SmallWorldBrain(numberOfNodes=10000, numberOfIterations=1000, rewireProbability=p, refractoryPeriod=1, fractionToFire=1/10000, neuronThreshold=1,
        fireRate=0.1, inhibitoryEdgeMultiplier=0, averageDegree=k, directed=False)
    
        pathlength.append(nx.average_shortest_path_length(brain.network))
        ccoeff.append(nx.average_clustering(brain.network))

    np.save('results/cpk_repeats/pathslengths_'+str(k)+'_'+str(seed), np.array(pathlength))
    np.save('results/cpk_repeats/clustercoeffs_'+str(k)+'_'+str(seed), np.array(ccoeff))

    if plot:
        frac = np.array(ccoeff)/np.array(pathlength)
        cubicspline = CubicSpline(ps, frac)
        ps_smooth = np.linspace(0.0005, 1, 100)
        plt.plot(ps, frac, '.', color='black')
        plt.plot(ps_smooth, cubicspline(ps_smooth), '-', color='black', label='$<k>$=%d' % k)
        plt.xlabel('$p_r$', size=16)
        plt.legend()
        plt.tick_params(labelsize=16)
        plt.ylabel('$C/L$', size=16)

def plot_topology_EEGs(N, t, save=False, load=False, filepath=None):

    if load:
        res = np.load(filepath)

    elif not load:
        small = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=0, fireRate=0.01, directed=False)
        small.execute()
        lattice = small.get_number_of_excited_neurons()

        small = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=0, fireRate=0.01, directed=False)
        small.execute()
        smallworld = small.get_number_of_excited_neurons()

        small = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.8, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=0, fireRate=0.01, directed=False)
        small.execute()
        random = small.get_number_of_excited_neurons()
        
        if save:
            np.save(filepath, np.array([lattice, smallworld, random]))

        res = np.array([lattice, smallworld, random])

    lattice = []
    for _, ye in zip(np.arange(t), [list(i) for i in res[0]]):
                lattice.append(len(ye))
    small = []
    for _, ye in zip(np.arange(t), [list(i) for i in res[1]]):
                small.append(len(ye))

    rand = []
    for _, ye in zip(np.arange(t), [list(i) for i in res[2]]):
                rand.append(len(ye))

    plt.figure()
    plt.plot(np.arange(t), np.array(lattice)/10000, label='Lattice', color='black')
    plt.plot(np.arange(t), np.array(small)/10000, label='Small-World', color='blue')
    plt.plot(np.arange(t), np.array(rand)/10000, label='Random', color='green')
    plt.xlabel('Timestep', size=16)
    plt.tick_params(labelsize=16)
    plt.ylabel('$N_{excited}/N_{total}$', size=16)
    plt.legend(prop={'size': 16}, loc='best')
