# MSci Project
# Measuring the degree distribution for different inhibitary probabilties
# Snigdha Sen and Sophie Martin
import matplotlib.pyplot as plt
from models import Base, RandomBrain, SmallWorldBrain, GrowPrefAttachmentBrain
from matplotlib import cm
import networkx as nx
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import random
import time
from scipy import sparse
import pickle
import matplotlib.animation as animation

# Get functions
def get_path_length_dist(Brain, degreeProbRange):
    '''
    Returns array of shortest paths lengths for a range of degrees in a given Brain network 
    '''
    np.random.seed(100)
    random.seed(100)
    pathLengths = []
    for deg in degreeProbRange:
        Brain.averageDegree = deg
        Brain.reset_network()
        pathLengths.append(nx.shortest_path_length(Brain.network))
    return np.array(pathLengths)


# Plot functions
def plot_transmission_curve():
    charge = np.linspace(-4, 5, 100)
    temps = np.arange(10, 100, 40)
    #temps = [50]
    colors = ['darkblue', 'darkcyan', 'chocolate', 'crimson', 'darkgreen', 'darkolivegreen']
    lowtemp = 0.001
    k = 8.6e-3
    plt.figure(figsize=(8,6))

    for idx, t in enumerate(temps):
        probTransmit = 1-1/(np.exp((charge-1)/(k*t))+1)
        plt.plot(charge, probTransmit, label='T=%d'%t, color=colors[idx])

    probTransmit = 1-1/(np.exp((charge-1)/(k*lowtemp))+1)
    plt.xlabel('$Q$', size=16)
    plt.ylabel('$P_{transmit}$', size=16)
    plt.tick_params(labelsize=16)
    plt.plot(charge, probTransmit, '--', color='black', label='T=%d'%lowtemp, markersize=16)
    plt.legend(prop={'size': 16})

def plot_transmissions_k():
    charge = np.linspace(-4, 5, 100)
    colors = ['black', 'darkgreen', 'blue', 'red']
    labels = ['$1e^{-4}$', '$1e^{-3}$','$1e^{-2}$', '$1e^{-1}$']
    ks = [1e-4, 1e-3 ,1e-2, 1e-1]
    #goodk = 8.6e-3
    #t = 50

    brain = SmallWorldBrain(numberOfNodes=100000, numberOfIterations=100, rewireProbability=0.1, refractoryPeriod=1, fractionToFire=1/100000, neuronThreshold=1,
    fireRate=0.1, averageDegree=5, directed=False)
    
    for idx, k in enumerate(ks):
        #probTransmit = 1-1/(np.exp((charge-2)/(k*t))+1)
        probTransmit = brain.transmissionProb(charge, k)
        if idx==0:
            plt.plot(charge, probTransmit, '--', label='k='+labels[idx], color=colors[idx])
        else:
            plt.plot(charge, probTransmit, label='k='+labels[idx], color=colors[idx])

    plt.xlabel('Q', size=16)
    plt.tick_params(labelsize=16)
    plt.ylabel('$P_{transmit}$', size=16)
    plt.legend(prop={'size': 16})

def plot_degree_dist(Brain):
    '''
    Plots the degree distribution
    '''
    np.random.seed(100)
    random.seed(100)
    bins= 30
    exDegree, inDegree = Brain.get_degree_distribution()

    plt.figure()
    plt.hist(exDegree, bins=bins, color='blue', label='Excitatory')
    plt.hist(inDegree, bins=bins, color='black', label='Inhibitory')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('N=  %d, Inhibit=%.1f, AvDegree=%d' % (Brain.numberOfNodes, Brain.inhibitoryEdgeProbability, Brain.averageDegree))
    plt.legend()

def plot_effective_average_degree(Brain, numberOfExcitedNeurons):
    '''
    Plots the effective average degree as a function of time
    '''

    raw, _, _ = Brain.get_effective_average_degree()
    #print(np.mean(np.array(effectiveExList)+np.array(effectiveInList)))
    #print('Ex range', min(effectiveExList), max(effectiveExList))
    #print('Inhib range', min(effectiveInList), max(effectiveInList))

    plt.figure()
    plt.plot(np.arange(len(numberOfExcitedNeurons)-1), np.array(raw)[:, 0], color='blue', label='Excitatory')
    plt.plot(np.arange(len(numberOfExcitedNeurons)-1), np.array(raw)[:, 1], color='black', label='Inhibitory')
    plt.ylabel('Effective Average Degree per Node')
    plt.xlabel('Time (timesteps)')
    #plt.title('N=  %d, Inhibit=%.1f, AvDegree=%d' % (Brain.numberOfNodes, Brain.inhibitoryEdgeProbability, Brain.averageDegree))
    plt.legend()

def plot_track_nodes(Brain, nodeList, numberOfIterations):
    '''
    Tracks when nodes in nodeList are excited during the simulation
    '''
    np.random.seed(100)
    random.seed(100)

    Brain.plot_scatter(Brain.get_number_of_excited_neurons(), numberOfIterations, nodeList)

def plot_critical_regimes(N, t, multiplerRegimes=[0, 35, 60]):

    small1 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplerRegimes[0], fireRate=0.01, directed=False)
    small1.execute(movie=False)

    small1Array = []
    for _, ye in zip(np.arange(small1.numberOfIterations), [list(i) for i in small1.numberOfExcitedNeurons]):
        small1Array.append(len(ye))
    ex, inhib, q1 = small1.get_effective_charge()
    #np.save('results/meanQ/subcritical', [ex, inhib, q1])
    print('m=%d, <Q>=%.3f:' %(multiplerRegimes[0], np.mean(np.array(ex)-np.array(inhib))))

    small2 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplerRegimes[1], fireRate=0.01, directed=False)
    small2.execute(movie=False)
    small2Array = []
    for _, ye in zip(np.arange(small2.numberOfIterations), [list(i) for i in small2.numberOfExcitedNeurons]):
        small2Array.append(len(ye))
    ex2, inhib2, q2 = small2.get_effective_charge()
    #np.save('results/meanQ/critical', [ex2, inhib2, q2])
    print('m=%d, <Q>=%.3f:' %(multiplerRegimes[1], np.mean(np.array(ex2)-np.array(inhib2))))

    small3 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplerRegimes[2], fireRate=0.01, directed=False)
    small3.execute(movie=False)
    small3Array = []
    for _, ye in zip(np.arange(small3.numberOfIterations), [list(i) for i in small3.numberOfExcitedNeurons]):
        small3Array.append(len(ye))
    ex3, inhib3, q3 = small3.get_effective_charge()
    #np.save('results/meanQ/supercritical', [ex3, inhib3, q3])
    print('m=%d, <Q>=%.3f:' %(multiplerRegimes[2], np.mean(np.array(ex3)-np.array(inhib3))))

    plt.figure()
    plt.plot(np.arange(t), np.array(small1Array)/small1.numberOfNodes, label='Supercritical', color='blue')
    plt.plot(np.arange(t), np.array(small2Array)/small1.numberOfNodes, label='Critical', color='red')
    plt.plot(np.arange(t), np.array(small3Array)/small1.numberOfNodes, label='Subcritical', color='black')

    #plt.title('Nodes: %d, <k>:%d, Refractory: %d, Threshold: %d' % (small1.numberOfNodes, small1.averageDegree, small1.refractoryPeriod, small1.neuronThreshold))
    plt.xlabel('Timestep', size=16)
    plt.tick_params(labelsize=16)
    plt.ylabel('$N_{excited}/N_{total}$', size=16)
    plt.legend(prop={'size':16})

def plot_coverage_variation(N, t, multiplierRegimes=[0, 35, 60]):
    
    small1 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplierRegimes[0], fireRate=0.01, directed=False)
    small1.execute(movie=False)
    small1Array = [len(np.unique(np.concatenate(small1.get_number_of_excited_neurons()[:i], axis=0)))/1000 for i in np.arange(1,t)]

    small2 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplierRegimes[1], fireRate=0.01, directed=False)
    small2.execute(movie=False)
    small2Array = [len(np.unique(np.concatenate(small2.get_number_of_excited_neurons()[:i], axis=0)))/1000 for i in np.arange(1,t)]

    small3 = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=multiplierRegimes[2], fireRate=0.01, directed=False)
    small3.execute(movie=False)
    small3Array = [len(np.unique(np.concatenate(small3.get_number_of_excited_neurons()[:i], axis=0)))/1000 for i in np.arange(1,t)]

    plt.figure()
    plt.plot(np.arange(1,t), np.array(small1Array), label='Supercritical', color='blue')
    plt.plot(np.arange(1,t), np.array(small2Array), label='Critical', color='red')
    plt.plot(np.arange(1,t), np.array(small3Array), label='Subcritical', color='black')

    #plt.title('Nodes: %d, <k>:%d, Refractory: %d, Threshold: %d' % (small1.numberOfNodes, small1.averageDegree, small1.refractoryPeriod, small1.neuronThreshold))
    plt.xlabel('Timestep', size=16)
    plt.tick_params(labelsize=16)
    plt.ylabel('Coverage', size=16)
    plt.xlim([0, t])
    plt.legend()

def make_activity_movie(numberOfExcitedNeurons, numberOfNodes, numberOfIterations, fps=10, filepath='./simulations/activity.mp4'):

    fig, _ = plt.subplots(1, 1)
    shape = (int(np.rint(np.sqrt(numberOfNodes))), int(np.rint(np.sqrt(numberOfNodes))))
    def init():
        return None

    def animate(i):
        black = np.zeros(shape[0]*shape[1])
        black[numberOfExcitedNeurons[i]] = 1
        black = black.reshape(shape[0], shape[1])
        plt.axis('off')
        plt.imshow(black, cmap='Greys_r')
        return None

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=numberOfIterations, interval=20, blit=False)


    macWriter = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(filepath, writer=macWriter)

def plot_inactive_degree_dist(duration, inactiveDegreeDist, neuronThreshold, sum=False):
    '''
    Plots degree distribution for nodes that are not currently excited at time t
    '''    
    plt.figure()

    if sum:
        plt.plot(np.arange(duration), inactiveDegreeDist[:, 0]+(inactiveDegreeDist[:, 1]*-1), color='black', label='Excitatory + -Inhibitory')
        plt.plot([0, duration], [neuronThreshold, neuronThreshold], '--', color='red', label='Threshold')
        plt.ylabel('Average Charge of Inactive Nodes')
    else:
        plt.plot(np.arange(duration), inactiveDegreeDist[:, 0], '+', color='black', label='Excitatory Degree')
        plt.plot(np.arange(duration), inactiveDegreeDist[:, 1], '+', color='blue', label='Inhibitory Degree')
        plt.ylabel('Average Degree for Inactive Nodes')
    
    plt.xlabel('Time (timesteps)')
    plt.legend()

def plot_multipler_vs_static_prob(multiplierRange, N):
    """
    Calculates the static fraction of inhibitory edges added for a given multiplier
    """
    averagei_e = []
    mean_i = []
    for m in multiplierRange:
        print(m)

        small = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
    numberOfIterations=1000, fractionToFire=0.05, inhibitoryEdgeMultiplier=m, fireRate=0.01, directed=False)

    # Calculate the mean frac of inhibitory per node (static)
        frac = []
        i_list = []
        for n in np.arange(N):
            i = np.argwhere(small.edgeMatrix[n] == -1).shape[0]
            i_list.append(i)
            e = np.argwhere(small.edgeMatrix[n] == 1).shape[0]
            frac.append(i/(e+i))

        averagei_e.append(np.mean(np.array(frac)))
        mean_i.append(np.mean(np.array(i_list)))

    _, ax = plt.subplots(1, 1)
    ax.plot(multiplierRange, averagei_e, '+', color='black', label='Frac')
    ax.set_xlabel('Multiplier')
    ax.set_ylabel('Actual mean inhibitory fraction per node')

    ax2 = ax.twinx()
    ax2.plot(multiplierRange, mean_i, '^', color='black', label='#')
    ax2.set_ylabel('Mean number of inhibitory edges added per node')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left')

def plot_avalanche_dist(numberOfExcitedNeurons, fireTimes):

    avalancheSizes = []
    for idx, _ in enumerate(fireTimes):
        if idx > 0:
            start = fireTimes[idx-1]
            end = fireTimes[idx]

            avalanche = numberOfExcitedNeurons[start:end]
            size = np.sum(avalanche)
            avalancheSizes.append(size)

    plt.figure()
    plt.hist(avalancheSizes)

# Whole analysis run functions
def run_growing_network_comparison():
    '''
    Analysis of the growing pref attachment brain, plotting their cluster co-eff and path length vaturation with mu 
    '''    
    degData = {}
    for mu in np.linspace(0, 1, 3):
        prefAttachment = GrowPrefAttachmentBrain(3, 3, mu, numberOfNodes=1000)
        deg = np.array(list(nx.degree(prefAttachment.network)))[:,1]
        degData[mu]= deg

    clusterData = {}
    systemSizes = [100, 1000, 10000, 100000]
    for mu in np.linspace(0, 1, 3):
        clusterNs = []
        for N in systemSizes:
            prefAttachment = GrowPrefAttachmentBrain(3, 3, mu, numberOfNodes=N)
            cluster = nx.average_clustering(prefAttachment.network)
            clusterNs.append(cluster)
        clusterData[mu] = np.array(clusterNs)


    clusterMData = []
    shortestPaths = []
    muData = np.logspace(-6, 0, 10)
    for mu in muData:
        prefAttachment = GrowPrefAttachmentBrain(3, 3, mu, numberOfNodes=10000)
        clusterMData.append(nx.average_clustering(prefAttachment.network))
        shortestPaths.append(nx.average_shortest_path_length(prefAttachment.network))
    clusterScaled = np.array(clusterMData)/clusterMData[0]
    shortestPathScaled = np.array(shortestPaths)/shortestPaths[0]


    plt.figure(1)
    markers = ['.', 'o', '^']
    colorvals = np.linspace(0, 0.6, len(clusterData))
    labels = list(clusterData.keys())
    for idx, clusterList in enumerate(clusterData.values()):
        plt.loglog(systemSizes, clusterList, c=str(colorvals[idx]), label='mu: '+str(labels[idx]), marker=markers[idx])
        plt.xlabel('N')
        plt.ylabel('C($mu$)')
        plt.legend()
    

    plt.figure()
    plt.loglog(muData, clusterScaled, c='r', marker='+', linewidth=0, label=r'C($\mu$)/C(0)')
    plt.loglog(muData, shortestPathScaled, c='b', marker='+', linewidth=0, label=r'L($\mu$)/L(0)',)
    plt.xlabel('$mu$')
    plt.yscale('linear')
    plt.legend()

def binary_search_function(seed, brain, m_min, m_max, precision, save=False):
    m = [m_min, m_max]
    print('Starting with m =', m)
    systemProbability = []
    averageInhibitoryFrac = []
    averageInhibitoryRaw = []
    multiplerList = []
    while m[1]-m[0] > precision: 
        
        for a_m in m:
            brain.add_inhibitory(a_m)
            brain.execute()
            if len(brain.numberOfExcitedNeurons[-1]) > 0:
                systemProbability.append(1)
            else:
                systemProbability.append(0)

            raw, frac = brain.get_effective_average_degree()
            averageInhibitoryFrac.append(np.mean(np.array(frac))) # averaged over the whole simulation
            averageInhibitoryRaw.append(np.mean(np.array(raw)))# averaged over the whole simulation
            multiplerList.append(a_m)
        
        m_med = m[0]+((m[1]- m[0])/2)
        brain.add_inhibitory(m_med)
        brain.execute()
        if len(brain.numberOfExcitedNeurons[-1]) > 0:
            # Prob 1
            m = [m_med, m[1]]
        else:
            # Prob 0
            m = [m[0], m_med]

        print('New m:', m)

    print('Seed=', seed, 'Final result: m_med =', m[0]+((m[1]- m[0])/2), 'm =',m, 'f=', averageInhibitoryFrac[-2:])
    averageInhibitoryFrac_s, systemProbability_s = zip(*sorted(zip(averageInhibitoryFrac, systemProbability), key=lambda pair: pair[0]))
    results_array = [averageInhibitoryFrac_s, systemProbability_s]

    multiplier_s1, systemProbability_s1 = zip(*sorted(zip(multiplerList, systemProbability), key=lambda pair: pair[0]))
    results_m_array = [multiplier_s1, systemProbability_s1]

    averageInhibRaw_s2, systemProbability_s2 = zip(*sorted(zip(averageInhibitoryRaw, systemProbability), key=lambda pair: pair[0]))
    results_r_array = [averageInhibRaw_s2, systemProbability_s2]

    m = [m_min, m_max]

    if save:
        # to save individual seed results
        filepath1 = 'epilepsy/multiprocess/results_f_'+str(seed)
        np.save(filepath1, np.array(results_array))
        
        filepath2 = 'epilepsy/multiprocess/results_m_'+str(seed)
        np.save(filepath2, np.array(results_m_array))

        filepath3 = 'epilepsy/multiprocess/results_r_'+str(seed)
        np.save(filepath3, np.array(results_r_array))

    
    return results_array, results_m_array, results_r_array

#@profile
def run_binary_search_risk_curve(brain, numberOfExperiments=10, m_min=0, m_max=20, precision=0.2, plot=False):
    '''
    Binary search for an optimum m range
    '''
    seedListTemp = np.random.randint(1000, size=2*numberOfExperiments)
    seedList = seedListTemp[numberOfExperiments:]
    print(seedList)

    results = {}
    results_m = {}
    results_r = {}

    if plot:
        plt.figure()

    n=0
    for seed in seedList:
        n += 1
        print('Running #', n, '/', numberOfExperiments, '...')
        resultsArray, results_mArray, results_rArray = binary_search_function(seed, brain, m_min, m_max, precision)
        
        results[seed] = resultsArray
        results_r[seed] = results_rArray
        results_m[seed] = results_mArray

    if plot:
        plt.figure()
        mvalues = np.array(list(results_m.values()))[:, 0]
        sysProbvalues = np.array(list(results_m.values()))[:, 1]
        plt.plot(np.mean(mvalues, axis=0), np.mean(sysProbvalues, axis=0), '-', color='black')
        plt.tick_params(labelsize=14)
        plt.xlabel('Inhibitory Edge Multiplier', size=16)
        plt.ylabel('Probability of a sustained signal', size=16)
        #plt.title('Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (brain.numberOfNodes, brain.refractoryPeriod, brain.neuronThreshold, brain.rewireProbability, brain.numberOfIterations)  )

        plt.figure()
        fracvalues = np.array(list(results.values()))[:, 0]
        sysProbvalues = np.array(list(results.values()))[:, 1]
        plt.plot(np.mean(fracvalues, axis=0), np.mean(sysProbvalues, axis=0), '-', color='black')
        plt.tick_params(labelsize=14)
        plt.xlabel('Average Effective Fraction of Inhibitory Edges', size=16)
        plt.ylabel('Probability of a sustained signal', size=16)
        #plt.title('Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (brain.numberOfNodes, brain.refractoryPeriod, brain.neuronThreshold, brain.rewireProbability, brain.numberOfIterations)  )

        plt.figure()
        rawvalues = np.array(list(results_r.values()))[:, 0]
        sysProbvalues = np.array(list(results_r.values()))[:, 1]
        plt.plot(np.mean(rawvalues, axis=0), np.mean(sysProbvalues, axis=0), '-', color='black') #Average over all repeats
        plt.tick_params(labelsize=14)
        plt.xlabel('Average Number of Inhibitory Edges per Excited Node', size=16)
        plt.ylabel('Probability of a sustained signal', size=16)
        #plt.title('Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (brain.numberOfNodes, brain.refractoryPeriod, brain.neuronThreshold, brain.rewireProbability, brain.numberOfIterations)  )

        plt.figure()
        fracvalues = np.array(list(results.values()))[:, 0]
        sysProbvalues = np.array(list(results.values()))[:, 1]
        plt.plot(np.mean(fracvalues, axis=0), np.mean(sysProbvalues, axis=0), '+', color='black')
        plt.tick_params(labelsize=14)
        plt.xlabel('Average Effective Fraction of Inhibitory Edges', size=16)
        plt.ylabel('Probability of a sustained signal', size=16)


    return results, results_r, results_m


def linear_risk_function(seed, brain, multiplierRange, save=True, plot=True):
    '''
    Cycles through range of param values and for a given network it will assess the probability of being excited
    i.e. for inhibitory variation
        Gradually increases the fraction of inhibitory edges in the edge matrix and assigns a probability of being excited at the end (1 or 0)
    
    P_inhibit used is the mean fraction per node
    '''

    systemProbability = []
    averageInhibitoryFrac =[]
    averageInhibitoryRaw = []
    multiplierList= []

    for m in multiplierRange:
        print('Calculating system state for m=%.2f' % m)
        brain.add_inhibitory(m) # updates the edge matrix for the brain
        brain.execute()
        if len(brain.numberOfExcitedNeurons[-1]) > 0:
            systemProbability.append(1)
        else:
            systemProbability.append(0)

        raw, frac = brain.get_effective_average_degree()
        averageInhibitoryFrac.append(np.mean(np.array(frac))) # averaged over the whole simulation
        averageInhibitoryRaw.append(np.mean(np.array(raw)))# averaged over the whole simulation
        multiplierList.append(m)
    
    averageInhibitoryFrac_s, systemProbability_s = zip(*sorted(zip(averageInhibitoryFrac, systemProbability), key=lambda pair: pair[0]))
    results_array = [averageInhibitoryFrac_s, systemProbability_s]

    multiplier_s1, systemProbability_s1 = zip(*sorted(zip(multiplierList, systemProbability), key=lambda pair: pair[0]))
    results_m_array = [multiplier_s1, systemProbability_s1]

    averageInhibRaw_s2, systemProbability_s2 = zip(*sorted(zip(averageInhibitoryRaw, systemProbability), key=lambda pair: pair[0]))
    results_r_array = [averageInhibRaw_s2, systemProbability_s2]

    if save:
        # to save individual seed results
        filepath1 = 'epilepsy/multiprocess/linear/results_f_'+str(seed)
        np.save(filepath1, np.array(results_array))
        
        filepath2 = 'epilepsy/multiprocess/linear/results_m_'+str(seed)
        np.save(filepath2, np.array(results_m_array))

        filepath3 = 'epilepsy/multiprocess/linear/results_r_'+str(seed)
        np.save(filepath3, np.array(results_r_array))

    if plot:
        plt.figure()
        plt.plot(multiplierRange, systemProbability, '.', label='deg=%d' % brain.averageDegree)
        plt.xlabel('Multiplier')
        plt.ylabel('Probability of a sustained signal')
        plt.title('Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (brain.numberOfNodes, brain.refractoryPeriod, brain.neuronThreshold, brain.rewireProbability, brain.numberOfIterations)  )
        plt.legend()

    return averageInhibitoryFrac, systemProbability

def run_k_transmission_variation(brain, kParams, charge):

    systemProb = []
    
    for k in kParams:
        brain.transmissionProb(charge, k)
        brain.execute()
        if len(brain.numberOfExcitedNeurons[-1]) > 0:
            systemProb.append(1)
        else:
            systemProb.append(0)

    plt.figure()
    plt.plot(kParams, systemProb, '.', label='Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (brain.numberOfNodes, brain.refractoryPeriod, brain.neuronThreshold, brain.rewireProbability, brain.numberOfIterations)  )
    plt.xlabel('k')
    plt.ylabel('Probability of a sustained signal')
    plt.legend()

def check_coverage(brain, numberOfNodes, start=0, end=None):
    if end is None:
        cov = np.unique(np.concatenate(brain.get_number_of_excited_neurons()))[start:]
    else:
        cov = np.unique(np.concatenate(brain.get_number_of_excited_neurons()))[start:end]
    return len(cov)/numberOfNodes

def firing_analysis(fireRange, numberOfNodes, numberOfIterations):

    coverageList = []
    for f in fireRange:
        small = SmallWorldBrain(numberOfNodes=numberOfNodes, averageDegree=5, rewireProbability=0.15, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=numberOfIterations, fractionToFire=f, inhibitoryEdgeMultiplier=80, fireRate=0.01, directed=False)
        small.execute()
        coverageList.append(check_coverage(brain=small, numberOfNodes=numberOfNodes))

    plt.figure()
    plt.plot(np.linspace(0, numberOfIterations, len(coverageList)), coverageList, '.', label='Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (small.numberOfNodes, small.refractoryPeriod, small.neuronThreshold, small.rewireProbability, small.numberOfIterations)  )
    plt.xlabel('Fraction of Nodes Fired')
    plt.ylabel('Coverage of Network')
    plt.legend()

def fire_time_analysis(drivingPeriodRange, numberOfNodes, numberOfIterations):

    coverageList = []
    for d in drivingPeriodRange:
        small = SmallWorldBrain(numberOfNodes=numberOfNodes, averageDegree=5, rewireProbability=0.15, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=numberOfIterations, fractionToFire=0.05, inhibitoryEdgeMultiplier=12, fireRate=0.01, directed=False)
        small.drivingPeriod = d
        small.execute()
        coverageList.append(check_coverage(brain=small, numberOfNodes=numberOfNodes))

    plt.figure()
    plt.plot(drivingPeriodRange, coverageList, '.', label='Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (small.numberOfNodes, small.refractoryPeriod, small.neuronThreshold, small.rewireProbability, small.numberOfIterations)  )
    plt.xlabel('Length of Driving Period')
    plt.ylabel('Coverage of Network')
    plt.legend()

def fire_rate_analysis(drivingRates, numberOfNodes, numberOfIterations):

    coverageList = []
    for r in drivingRates:
        small = SmallWorldBrain(numberOfNodes=numberOfNodes, averageDegree=5, rewireProbability=0.15, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=numberOfIterations, fractionToFire=0.05, inhibitoryEdgeMultiplier=12, fireRate=r, directed=False)
        small.execute()
        coverageList.append(check_coverage(brain=small, numberOfNodes=numberOfNodes))

    plt.figure()
    plt.plot(drivingRates, coverageList, '.', label='Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (small.numberOfNodes, small.refractoryPeriod, small.neuronThreshold, small.rewireProbability, small.numberOfIterations)  )
    plt.xlabel('Driving Rate')
    plt.ylabel('Coverage of Network')
    plt.legend()

def fire_area_analysis(fireRange, numberOfNodes, numberOfIterations):

    coverageList = []
    for f in fireRange:
        small = SmallWorldBrain(numberOfNodes=numberOfNodes, averageDegree=5, rewireProbability=0.15, neuronThreshold=1, refractoryPeriod=1, 
        numberOfIterations=numberOfIterations, fractionToFire=f, inhibitoryEdgeMultiplier=12, fireRate=0.01, directed=False)
        small.execute(makeRandom=False)
        coverageList.append(check_coverage(brain=small, numberOfNodes=numberOfNodes))
    plt.figure()
    plt.plot(np.linspace(0, numberOfIterations, len(coverageList)), coverageList, '.', label='Nodes=%d, ref=%d, threshold=%d, rewire=%.2f, time=%d' % (small.numberOfNodes, small.refractoryPeriod, small.neuronThreshold, small.rewireProbability, small.numberOfIterations)  )
    plt.xlabel('Fraction of Nodes Fired Selectively')
    plt.ylabel('Coverage of Network')
    plt.legend()


# Saving and loading experiments / data
def plot_load_experiment(filepath):
    plt.figure()
    f = open(filepath, 'rb')
    experimentDict = pickle.load(f)
    f.close()
    for i in list(experimentDict.keys()):
        plt.plot(np.linspace(0,1,len(experimentDict[i][0])), experimentDict[i][0], label=str(i))
        plt.fill_between(np.linspace(0,1,len(experimentDict[i][0])), experimentDict[i][0]-experimentDict[i][1], experimentDict[i][0]+experimentDict[i][1], alpha=0.4)

    plt.xlabel('Inhibitatory Probability')
    plt.ylabel('Probability of Excited at End')
    plt.title('fractionToFire: %.3f, neuronThreshold: %d, refrac:%d, deg:%d' % (0.025, 1, 3, 4))
    plt.legend()
