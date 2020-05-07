# MSci Project
# Snigdha Sen and Sophie Martin

import numpy as np
import networkx as nx 
import matplotlib.cm as cm
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
import random
import time
from numba import jit
import pickle
from itertools import combinations
from logbin230119 import logbin

class Base:

    def __init__(self, numberOfNodes=1000, numberOfIterations=20, refractoryPeriod=3, fractionToFire=1/1000, neuronThreshold=2,
    fireRate=0.1, inhibitoryEdgeMultiplier=0, inwardEdgeMultiplier=None, averageDegree=None, directed=False):

        # Variables for the network structure
        self.numberOfNodes = numberOfNodes
        # These get redefined in each subclass: empty graph here with random layout of nodes
        self.network = nx.Graph()
        self.nodePos = nx.random_layout(self.network)
        self.state = self.initialise_state()
        self.edgeMatrix = self.make_edge_matrix()
        self.averageDegree = averageDegree # Minimum value
        self.directed = directed
        self.inwardEdgeMultiplier = inwardEdgeMultiplier
        if self.directed and inwardEdgeMultiplier is None:
            raise Exception('Directed edge fraction not specified but directed set to True')
        
        # Variables for the propagation
        self.refractoryPeriod = refractoryPeriod
        self.neuronThreshold = neuronThreshold

        if self.averageDegree is not None and self.averageDegree < 2*self.neuronThreshold:
            raise Exception('Average degree too small for neuron threshold')
        self.probabilityOfFiring = fireRate*np.exp(-fireRate)
        self.fractionToFire = fractionToFire
        self.inhibitoryEdgeMultiplier = inhibitoryEdgeMultiplier
        self.numberOfIterations = numberOfIterations
        self.drivingPeriod = 1

        # Runtime parameters for analysis
        self.fireTimes = []
        self.numberOfExcitedNeurons = []

    def make_edge_matrix(self):
        # Initialises an undirected edge matrix links with all synapses excitatory
        # For non-zero directed edge probability begins with all directed outwards 

        matrix = sparse.dok_matrix((self.numberOfNodes, self.numberOfNodes))
        if len(nx.edges(self.network)) > 0:
            if self.directed: # Only add outwards edges
                for edge in nx.edges(self.network):
                    
                    if edge[0] < self.neuronThreshold and edge[1] >= (edge[0]-self.neuronThreshold)%self.numberOfNodes:
                        matrix[edge[1], edge[0]] = 1
                    else:
                        matrix[edge[0], edge[1]] = 1

            else: # Add edges in both directions by default
                for edge in nx.edges(self.network):

                    matrix[edge[0], edge[1]] = 1
                    matrix[edge[1], edge[0]] = 1

            #reachableNodes = np.count_nonzero(sparse.dok_matrix.sum(matrix, axis=0) >= self.neuronThreshold)
        
            #if reachableNodes != self.numberOfNodes:
                # Ensures each node has at least one inwards excitatory edge
                #nodesDisconnected = list(set(np.arange(0,self.numberOfNodes)) - set( np.nonzero(sparse.dok_matrix.sum(matrix, axis=0) >= self.neuronThreshold)[1]))
                #raise Exception('Bad network (disconnected). Some nodes not have enough incoming excitatory edges for specific neuron threshold')
            return matrix.tocsr()
        else:
            return None
    
    def manual_rewire(self, rewireProb):
        # Manually rewire some edges in a network whilst ensuring that all nodes can be reached
        
        numberToRewire = int(len(nx.edges(self.network))*rewireProb)
        edgesIndexesToRewire = np.random.choice(len(list(nx.edges(self.network))), numberToRewire, replace=False)
        edgesToRewire = [tuple(l) for l in np.array(list(nx.edges(self.network)))[edgesIndexesToRewire]]
        matrix = self.edgeMatrix.tolil()
        for edge in edgesToRewire:
            
            nodesExempt = 1
            probs = np.ones(self.numberOfNodes)
            probs[edge[0]] = 0

            ends = np.array(list(self.network.edges(edge[0])))[:, 1]
            probs[ends] = 0
            nodesExempt += len(ends)
            equalprob = 1/(self.numberOfNodes-nodesExempt)
            probs[probs != 0 ] = equalprob
            newEnd = np.random.choice(np.arange(0, self.numberOfNodes), p = probs)
            matrix[edge[0], newEnd] = matrix[edge[0], edge[1]]
            matrix[edge[0], edge[1]] = 0
            reachableNodes = np.count_nonzero(sparse.dok_matrix.sum(matrix, axis=0) >= self.neuronThreshold)

            while reachableNodes != self.numberOfNodes:
                matrix = self.edgeMatrix
                probs[newEnd] = 0
                nodesExempt += 1
                if self.numberOfNodes - nodesExempt != 0:
                    equalprob = 1/(self.numberOfNodes-nodesExempt)
                    probs[probs != 0 ] = equalprob
                    newEnd = np.random.choice(np.arange(0, self.numberOfNodes), p = probs)

                    matrix[edge[0], newEnd] = matrix[edge[0], edge[1]]
                    matrix[edge[0], edge[1]] = 0
                else:
                    print('skipping')
                    reachableNodes = self.numberOfNodes
            
            # Update the edge matrix and the graph
            self.network.remove_edge(edge[0], edge[1])
            self.network.add_edge(edge[0], newEnd)
            self.edgeMatrix = matrix
        
        self.edgeMatrix = matrix.tocsr()

    def add_inhibitory(self, scaler=None):
        # Scaler: ratio of inhib to excitatory across network in the form scaler:1
        # Add one new inhibitory edge by default if not specified
        if scaler is None:
            scaler = 1/len(nx.edges(self.network))
        
        elif scaler > 0:

            matrix = self.edgeMatrix.tolil() # For speed
            N = scaler*len(nx.edges(self.network)) - (sparse.find(matrix == -1)[1].shape[0])/2 # Number of edges to add or remove
            
            if N > 0: # Adding more
                if scaler*len(nx.edges(self.network)) > self.numberOfNodes**2 - (sparse.find(matrix !=0)[1].shape[0])/2:
                    print(N, self.numberOfNodes**2 - len(nx.edges(self.network)))
                    raise Exception('Cannot add more edges than there are spaces')
            
                elements = random.sample(list(combinations(range(self.numberOfNodes),2)), int(N))

                fail = sparse.find(matrix[np.array(elements)[:,0], np.array(elements)[:, 1]] != 0)[1] # locations where it an edge already exists
                
                while len(fail) > 0: 
                    # Find len(fail) other new elements to change
                    new = random.sample(list(set(combinations(range(self.numberOfNodes),2)) - set(elements)), len(fail)) # make nFail new ones
                    for idx, loc in enumerate(fail):
                        elements[loc] = new[idx]
                    fail = sparse.find(matrix[np.array(elements)[:,0], np.array(elements)[:, 1]] != 0)[1]

                matrix[np.array(elements)[:,0], np.array(elements)[:,1]] = -1 # Add an inhibitory edge where there isn't currently an edge
                matrix[np.array(elements)[:,1], np.array(elements)[:,0]] = -1 # Add an inhibitory edge where there isn't currently an edge

            elif N < 0: # If more inhibitory already exists remove some existing ones
                inhibRows = sparse.find(matrix == -1)[0]
                inhibCols = sparse.find(matrix == -1)[1]
                inhibs = np.vstack((inhibRows, inhibCols)).T
                elements = random.sample([tuple(x) for x in inhibs], int(np.abs(N)))

                matrix[np.array(elements)[:,0], np.array(elements)[:,1]] = 0 # Remove an inhibitory edge 
                matrix[np.array(elements)[:,1], np.array(elements)[:,0]] = 0 # Remove an inhibitory edge 

            self.edgeMatrix = matrix.tocsr() # updates the edge matrix


    def add_directedness(self, fraction=None):
        # Add one new inward edge by default if not specified
        if fraction is None:
            fraction = 1/len(nx.edges(self.network))
        

        matrix = self.edgeMatrix.tolil() # For speed
        n = 0

        while n < fraction*len(nx.edges(self.network)):
            print(n)
            row = np.random.randint(0, self.numberOfNodes-1, 1)
            col = np.random.randint(0, self.numberOfNodes-1, 1)

            while self.edgeMatrix[row, col] == 0: # TODO fix this
                row = np.random.randint(0, self.numberOfNodes-1, 1)
                col = np.random.randint(0, self.numberOfNodes-1, 1)

            matrix[col, row] = matrix[row, col]
            n+=1

        self.edgeMatrix = matrix.tocsr()

    def make_directed_networkx_graph(self):
        # Return the network since this is actually being updated (for plotting will make a difference)
        graph = nx.DiGraph(self.network, directed=True) 
        # By default all edges go both ways
        directedEdges = np.argwhere(np.abs(self.edgeMatrix==1)) 
        # Edge matrix is directed
        dirEdges = [tuple(l) for l in directedEdges]

        graph.remove_edges_from(list(graph.edges))
        graph.add_edges_from(dirEdges)
        return graph

    def reset_network(self):
        # Reset network after changing any parameters
        self.network = self.initialise_network()

    def initialise_state(self):
        return np.zeros(self.numberOfNodes)
    
    def initialise_network(self):
        # Undefined for base class, specific to child classes
        return None

    def transmissionProb(self, c, k=8.6e-3):
        # Skewed Fermi distribution, low probablity for Q=threshold, and increasing probability for Q>threshold
        # No 
        #  below the threshold
        if type(c) is int:
            probTransmit = 1-1/(np.exp((c-self.neuronThreshold)/(k*50))+1)
        elif len(c) > 0:
            probTransmit = []
            for a_c in c:
                probTransmit.append(1-1/(np.exp((a_c-self.neuronThreshold)/(k*50))+1))
        else:
            raise Exception('Neuron charges must be single value or array-like')
        return probTransmit

    def check_strongly_connected(self): 
        # Checks strongly connected components and returns the number of nodes in components with at least 2 nodes
        if nx.is_strongly_connected(self.network):
            print('Directed graph is strongly connected for all nodes')
        elif nx.is_weakly_connected(self.network):
            print('Directed graph is only weakly connected')
        else:
            print('Directed graph is not connected at all')

    def random_fire(self, time=0):
        if self.fractionToFire == 1/self.numberOfNodes:
            self.state[0] = 1

        else:
            availableNodes = np.where(self.state == 0)[0]
            if len(availableNodes) > self.fractionToFire*self.numberOfNodes: # Only when you have enough excitatory resting nodes can you excite
                randomFire = np.random.choice(availableNodes, int(self.fractionToFire*self.numberOfNodes), replace=False)
                self.state[randomFire] = 1

        self.fireTimes.append(time)

    def selective_fire(self, time=0):
        availableNodes = np.where(self.state == 0)[0]
        if len(availableNodes) > self.fractionToFire*self.numberOfNodes:
            selectiveFire = np.random.choice(availableNodes, 1, replace=False)
            n = 0
            while n < self.fractionToFire*self.numberOfNodes:
                self.state[selectiveFire+n] = 1
                n+=1

        self.fireTimes.append(time)

    def execute(self, movie=False, filepath='./simulations/brain_default.mp4', makeRandom=True):

        numberOfIterations = self.numberOfIterations
        parameterDict = {'numberOfNodes':self.numberOfNodes, 'refractoryPeriod':self.refractoryPeriod, 
         'neuronThreshold':self.neuronThreshold, 'fractionToFire':self.fractionToFire}

        if movie:
            self.run_simulation(numberOfIterations=numberOfIterations, parameters=parameterDict, frames=numberOfIterations, 
            filepath=filepath, movie=movie, drivingPeriod=self.drivingPeriod, makeRandom=makeRandom)

        else:
            self.run_simulation(numberOfIterations=numberOfIterations, parameters=parameterDict, frames=numberOfIterations, 
            filepath=filepath, movie=movie, drivingPeriod=self.drivingPeriod, makeRandom=makeRandom)

        return None

    def run_simulation(self, numberOfIterations, parameters, frames, filepath, movie, drivingPeriod, makeRandom=True):
        
        if makeRandom:
            self.random_fire()
        else:
            self.selective_fire()

        fig = plt.figure(1)

        anim = animation.FuncAnimation(fig, self.animate, frames=frames, fargs=(movie, drivingPeriod, makeRandom, ), blit=False, 
        repeat=False, init_func=self.init_animate)

        if movie:
            macWriter = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'))
            anim.save(filepath, writer=macWriter)
        else:
            macWriter = animation.FFMpegWriter(fps=100, metadata=dict(artist='Me'))
            anim.save('./simulations/no_movie_file.mp4', writer=macWriter)
        plt.close()

    def init_animate(self):
        #plt.axis('off')
        return None
    
    def animate(self, i, movie, drivingPeriod, fireTimes, makeRandom=True):
        norm_grey = colors.Normalize(vmin=1, vmax=self.refractoryPeriod+2) # White is excited, and end of the refrac. period is dark dark grey but not black
        greys = cm.get_cmap('Greys')
        colorMap = []
        if i>0:
            self.state = propagate(state=self.state, network=self.network, neuronThreshold=self.neuronThreshold, 
            transmissionProb = self.transmissionProb, refractoryPeriod=self.refractoryPeriod, timestep=i, edgeMatrix=self.edgeMatrix)
            if i < int(drivingPeriod):
                if np.random.uniform() <= self.probabilityOfFiring:
                    if makeRandom:
                        self.random_fire(i)
                    else:
                        self.selective_fire(i)

        self.numberOfExcitedNeurons.append(np.where(self.state==1)[0])

        if movie:
            for neuronState in self.state:
                if 0 < neuronState < self.refractoryPeriod+2:
                    colorMap.append(greys(norm_grey(neuronState)))
                else:
                    colorMap.append('black')

            nx.draw_networkx_nodes(self.network, self.nodePos, nodelist=nx.nodes(self.network), node_color=colorMap)
            nx.draw_networkx_edges(self.network, self.nodePos)
            nx.draw_networkx_labels(self.network, self.nodePos)

    # Plotting and results functions
    def save_brain_config(self, filepath='./config/default_brain_config.p'):
        with open(filepath, 'wb') as outfile:
            pickle.dump(self, outfile) 
    
    def load_brain_config(self, filepath):
        with open(filepath, 'rb') as outfile:
            loadedBrain = pickle.load(outfile)
        return loadedBrain
    
    def get_number_of_excited_neurons(self):
        return self.numberOfExcitedNeurons
    
    def get_fire_times(self):
        return self.fireTimes

    def get_degrees(self, nodeList=None):
    # Return the degree (excitatory and inhibitory) from the edge matrix for a list of nodes
        if nodeList is None:
            nodeList = np.arange(self.numberOfNodes)  
        
        degreeList = self.network.degree(nodeList)
        excitatoryDegree = np.array(list(degreeList))[:, 1]
        inhibitoryDegree=[]
        for node in nodeList: # Removing inhibitory edges
            inhibitoryDegree.append(np.abs(np.sum(self.edgeMatrix[node] == -1)))
        
        return excitatoryDegree, inhibitoryDegree

    #@profile
    def get_effective_charge(self):
        '''
        After executing a signal on the network, retrieve the effective edge degree (raw and fractional) throughout the simulation
        '''
        
        numberOfExcitedNeurons = self.get_number_of_excited_neurons()
        
        excit = []
        inh = []

        for t in np.arange(len(numberOfExcitedNeurons)-1):

            presynapticnodes = list(numberOfExcitedNeurons[t])
            postsynapticnodes = list(numberOfExcitedNeurons[t+1]) # postsynaptic excited only
       
            if len(presynapticnodes) >0: #Only do this whilst self-sustaining

                reducedmatrix = self.edgeMatrix[presynapticnodes] # Only look at presynaptic rows
                
                effectiveExList = []
                effectiveInList = []

                for n in postsynapticnodes:
                    effectiveInhibNum = len(sparse.find(reducedmatrix[:, n] == -1)[1])
                    effectiveExcitedNum = len(sparse.find(reducedmatrix[:, n] == 1)[1])
                    #effectiveInhibNum = len(list(filter(lambda x: x < 0, reducedmatrix[:, n].toarray())))
                    
                    effectiveExList.append(effectiveExcitedNum)
                    effectiveInList.append(effectiveInhibNum)

                inh.append(np.mean(effectiveInList)) # mean number of inhib per  node
                excit.append(np.mean(effectiveExList))  # mean number of excited per node
                q = np.mean(np.array(effectiveExList)-np.array(effectiveInList))
        return excit, inh, q

    def get_effective_average_degree(self):
        '''
        After executing a signal on the network, retrieve the effective inhib edge degree (raw and fractional) throughout the simulation
        '''
        
        numberOfExcitedNeurons = self.get_number_of_excited_neurons()
        
        frac = []
        raw = []

        for t in np.arange(len(numberOfExcitedNeurons)-1):

            presynapticnodes = list(numberOfExcitedNeurons[t])
            #postsynapticnodes = list(numberOfExcitedNeurons[t+1]) # postsynaptic excited only
            '''
            effectiveExFrac = []
            effectiveInFrac = []

            effectiveExList = []
            effectiveInList = []
            '''

            if len(presynapticnodes) >0: #Only do this whilst self-sustaining

                reducedmatrix = self.edgeMatrix[presynapticnodes] # Only look at presynaptic rows
                postsynapticnodes = np.unique(reducedmatrix.nonzero()[1]) # All end nodes of the excited ones (not including those that stochastically fire)

                a = np.array(sparse.csc_matrix.sum(reducedmatrix[:, postsynapticnodes], axis=0))[0]
                b = np.array(sparse.csc_matrix.sum(np.abs(reducedmatrix[:, postsynapticnodes]), axis=0))[0]

                effectiveInhibNum = (b - a)/2

                '''
                for n in postsynapticnodes:
                    #effectiveInhibNum = len(sparse.find(reducedmatrix[:, n] == -1)[1])
                    #effectiveInhibNum = len(list(filter(lambda x: x < 0, reducedmatrix[:, n].toarray())))
                    
                    
                    effectiveExList.append(effectiveExcitedNum)
                    effectiveInList.append(effectiveInhibNum)

                    if effectiveInhibNum !=0 or effectiveExcitedNum!=0: # When there is some non zero edges (not a random fire despite not having any)
                        effectiveExFrac.append(effectiveExcitedNum/(effectiveExcitedNum+effectiveInhibNum)) # fraction of ex per node
                        effectiveInFrac.append(effectiveInhibNum/(effectiveExcitedNum+effectiveInhibNum))

                    elif effectiveExcitedNum == 0:
                        firedAnyway += 1 # Count nodes that fire stochastically despite no excitatory edges
                '''

                frac.append(np.mean(effectiveInhibNum/b)) # mean fraction of excitatory and inhib per edge
                raw.append(np.mean(effectiveInhibNum))

        return raw, frac

    def plot_scatter(self, numberOfExcitedNeurons, numberOfIterations, specificNodes=None):

        plt.figure(figsize=(8,6))

        if specificNodes is None:
            for xe, ye in zip(np.arange(numberOfIterations), [list(i) for i in numberOfExcitedNeurons]):
                plt.scatter([xe] * len(ye), ye, color='k', s=0.05)
        
        else:
            colorDict = {}
            for i in specificNodes:
                colorDict[i] = list(np.random.random(size=3))

            for xe, ye in zip(np.arange(numberOfIterations), [list(i) for i in numberOfExcitedNeurons]):
                for y in ye:
                    if y in specificNodes:
                        plt.scatter(xe, y, color=tuple(colorDict[y]), s=0.05, label=str(y))
        #plt.title('Node Activity')
        plt.tick_params(labelsize=16)
        plt.ylabel('Node', size=16)
        plt.xlabel('Timestep', size=16)

  
    def plot_eeg(self, numberOfExcitedNeurons, numberOfIterations):
        numList = []
        for _, ye in zip(np.arange(numberOfIterations), [list(i) for i in numberOfExcitedNeurons]):
            numList.append(len(ye))

        fireTimes = self.fireTimes
        
        plt.figure()
        plt.plot(np.arange(numberOfIterations), np.array(numList)/self.numberOfNodes, label='nodes: %d, fire: %d, avdeg:%d, inhib: %.2f, refrac: %d, threshold: %d' % (self.numberOfNodes, int(self.fractionToFire*self.numberOfNodes), self.averageDegree, np.sum(self.edgeMatrix == -1), self.refractoryPeriod, self.neuronThreshold), color='black')
        for times in fireTimes:
            plt.axvline(times, color='gray')
        plt.xlabel('Timestep', size=14)
        plt.tick_params(labelsize=14)
        #plt.ylabel('$N_{neurons}/N_{max}$', size=14)
        plt.ylabel('$N_{excited}/N_{total}$', size=14)
        plt.legend()
        

    def plot_fourier_transform_eeg(self, numberOfExcitedNeurons, numberOfIterations):

        numList = []
        for _, ye in zip(np.arange(numberOfIterations), [list(i) for i in numberOfExcitedNeurons]):
            numList.append(len(ye))
        fourierTransform = np.fft.fft(np.array((numList)))

        fireTimes = self.fireTimes

        plt.figure()
        plt.plot(np.arange(numberOfIterations), fourierTransform, label='nodes: %d, fire: %d, avdeg:%d, inhib: %.2f, refrac: %d, threshold: %d' % (self.numberOfNodes, int(self.fractionToFire*self.numberOfNodes), self.averageDegree, np.sum(self.edgeMatrix == -1), self.refractoryPeriod, self.neuronThreshold), color='black')
        for times in fireTimes:
            plt.axvline(times, color='gray')
        plt.xlabel('Timestep', size=14)
        plt.tick_params(labelsize=14)
        plt.ylabel('Fourier Transform of $N_{excited}$', size=14)
        plt.legend()
        
def propagate(state, network, neuronThreshold, transmissionProb, refractoryPeriod, timestep, edgeMatrix):

    updatedState = np.copy(state)
    excitedNeurons = np.where(updatedState == 1)[0]

    if len(excitedNeurons) > 0:

        updatedState[excitedNeurons] += 1
        excitedOnly = edgeMatrix[excitedNeurons]
        # Find end neurons connected to currently excited nodes (but not excited nodes themselves!)
        connectedNeurons = list(set(np.unique(sparse.find(excitedOnly!=0)[1])).difference(set(excitedNeurons)))
        if len(list(set(connectedNeurons) & set(excitedNeurons))) > 0:
            raise Exception('Excited neurons should not be captured in propagate')
        
        neuronCharge = np.array(sparse.csc_matrix.sum(excitedOnly[:, connectedNeurons], axis=0))
        # neuronsAboveThreshold = np.where(neuronCharge >= neuronThreshold)[1] - we do not want to exclude any neurons based on charge 

        # Stochastic propagation according to a probability function transmissionProb
        tempVar = np.random.uniform(size=len(connectedNeurons))
        #transNeuronsCharge = neuronCharge[0][neuronsAboveThreshold]
        if len(neuronCharge[0]) > 0:
            transmitIdx = np.where(tempVar <= transmissionProb(neuronCharge[0]))[0] 
            
            if len(transmitIdx > 0): 
                updatedState[np.array(connectedNeurons)[transmitIdx]] += 1

    refractoryNeurons = np.where(updatedState > 1)[0]
    if len(refractoryNeurons) > 0:
        updatedState[refractoryNeurons] = np.mod(state[refractoryNeurons]+1, refractoryPeriod+2)

    return updatedState


class RandomBrain(Base):

    def __init__(self, *args, **kwargs):

        super(RandomBrain, self).__init__(*args, **kwargs)

        # Reinitialising parameters specific to this model
        self.network = self.initialise_network()
        self.nodePos = nx.shell_layout(self.network)
        self.edgeMatrix = self.make_edge_matrix()
        self.add_inhibitory(self.inhibitoryEdgeMultiplier)
        if self.directed:
            self.network = self.make_directed_networkx_graph()
            self.add_directedness(self.inwardEdgeMultiplier)

    def initialise_network(self):
        return nx.erdos_renyi_graph(self.numberOfNodes, self.averageDegree/self.numberOfNodes)


class SmallWorldBrain(Base):

    def __init__(self, rewireProbability, *args, **kwargs):

        super(SmallWorldBrain, self).__init__(*args, **kwargs)

        # Reinitialising parameters specific to this model

        self.rewireProbability = rewireProbability # Associated probability required for small world
        self.network = self.initialise_network()
        self.nodePos = nx.circular_layout(self.network)
        self.edgeMatrix = self.make_edge_matrix()
        self.add_inhibitory(self.inhibitoryEdgeMultiplier)
        if self.directed:
            self.network = self.make_directed_networkx_graph()
            self.add_directedness(self.inwardEdgeMultiplier)
        #self.manual_rewire(rewireProbability)

    def initialise_network(self):
        graph = nx.watts_strogatz_graph(self.numberOfNodes, self.averageDegree, self.rewireProbability)
        return graph


class BarabasiAlbertBrain(Base):

    def __init__(self, rewireProbability, *args, **kwargs):

        super(BarabasiAlbertBrain, self).__init__(*args, **kwargs)

        # Reinitialising parameters specific to this model
        self.network = self.initialise_network()
        self.nodePos = nx.circular_layout(self.network)
        self.edgeMatrix = self.make_edge_matrix()
        self.add_inhibitory(self.inhibitoryEdgeMultiplier)
        if self.directed:
            self.network = self.make_directed_networkx_graph()
            self.add_directedness(self.inwardEdgeMultiplier)
        self.manual_rewire(rewireProbability)


    def initialise_network(self):
        graph = nx.barabasi_albert_graph(self.numberOfNodes, self.averageDegree)
        return graph
        

class GrowPrefAttachmentBrain(Base):

    def __init__(self, m, vertexNumber, mu, rewireProbability=0, *args, **kwargs):

        # Initialises a empty network with parameters required for propagation like the others
        super(GrowPrefAttachmentBrain, self).__init__(*args, **kwargs)

        self.mu = mu
        self.vertexNumber = vertexNumber
        self.m = m
        self.averageDegree = m
        self.network = self.initialise_network()
        self.nodePos = nx.circular_layout(self.network)
        self.add_inhibitory(self.inhibitoryEdgeMultiplier)
        if self.directed:
            self.network = self.make_directed_networkx_graph()
            self.add_directedness(self.inwardEdgeMultiplier)
        self.manual_rewire(rewireProbability)

    def initialise_network(self):
        return self.grow_preferential_attachment()

    def grow_preferential_attachment(self):
        np.random.seed(100)
        m = self.m
        mu = self.mu
        N = self.numberOfNodes
        vertexNumber = self.vertexNumber
        
        g = nx.complete_graph(m) 
        if self.directed:
            singleEdges = [tuple(l) for l in list(g.edges())]
            for idx, edge in enumerate(singleEdges):
                # Flip the end edge in the initial complete graph loop
                if edge[0] == 0 and edge[1] == len(g.nodes)-1:
                    singleEdges[idx] = tuple([edge[1], edge[0]])

            g = nx.DiGraph(g)
            g.remove_edges_from(list(g.edges))
            g.add_edges_from(singleEdges)

            
        systemState = np.ones(m)

        preferenceList = []
        
        counter = 0    
        while counter < m:
            value = 0
            while value < m:
                preferenceList.append(counter)
                value += 1
            counter += 1
            
        while vertexNumber < N:
            g.add_node(vertexNumber)
            lenPrefList = len(preferenceList) - 1
            newVar = 0
            while newVar < m:
                if np.random.random() < mu:
                    randomNumber = np.random.randint(0, lenPrefList)
                    indexChoice = preferenceList[randomNumber]
                else:
                    indexChoice = np.random.choice(np.where(systemState == 1)[0])
                edgeTest = g.has_edge(vertexNumber,indexChoice)
                while edgeTest == True:
                    if np.random.random() < mu:
                        randomNumber = np.random.randint(0, lenPrefList)
                        indexChoice = preferenceList[randomNumber]
                    else:
                        indexChoice = np.random.choice(np.where(systemState == 1)[0])
                    edgeTest = g.has_edge(vertexNumber,indexChoice)
    
                g.add_edge(vertexNumber, indexChoice)
                preferenceList.append(indexChoice)
                preferenceList.append(vertexNumber)
                newVar += 1
            systemState = np.append(systemState, 1)
            vertexNumber += 1
            #print(vertexNumber, mu)
            if vertexNumber > m:
                
                activeDegrees = np.array(list(nx.degree(g)))[:,1][np.where(systemState == 1)[0]]
                a = (np.sum(1/activeDegrees))**(-1)
                probList = a*(1/activeDegrees)

                deactivate = np.random.choice(np.where(systemState == 1)[0], p=probList)
                systemState[deactivate] = 0

        return g

    def plot_log_degree_distribution(self, degrees):
        # degrees is an array of degrees or a dict containing multiple degree arrays

        if len(degrees) == self.numberOfNodes: 
            # i.e. one degree list
            logDataX,logDataY,_ = logbin(degrees,1.1,zeros=False)
            plt.figure()
            plt.loglog( logDataX,logDataY, c='b', label='Data')
            #plt.plot(logDataX,np.log(random),c='r',label='Theoretical fit $p_{\infty}(k)$')
            plt.xlabel('log k')
            plt.ylabel('p(k)')
            plt.legend()
            
        else:
            plt.figure()
            markers= ['.', 'o', '^']
            colorvals = np.linspace(0, 0.6, len(degrees))
            labels = list(degrees.keys())
            for idx, degList in enumerate(degrees.values()):
                logDataX,logDataY,_ = logbin(degList,1.1,zeros=False)
                plt.loglog(logDataX,logDataY, '.', c=str(colorvals[idx]),label='mu: '+str(labels[idx]), marker=markers[idx])
            plt.xlabel('$k$')
            plt.ylabel('$P(k)$')
            plt.legend()

    def graph_comparison(self, m, vertexNumber, mu):

        plt.figure()
        plt.plot()
    