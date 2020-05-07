from multiprocessing import Pool
import analysis
import models
import random
import numpy as np

if __name__ == '__main__':
    np.random.seed(80)
    random.seed(80)
    N = 1000
    t = 1000
    seeds = np.random.randint(1000, size=50)
    small = models.SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.1, neuronThreshold=1, refractoryPeriod=1, 
numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=0, fireRate=0.01, directed=False)
    '''iterable = [(seed, small, 0, 100, 5, True) for seed in seeds]
    with Pool(3) as p:
        print(p.starmap(analysis.binary_search_function, iterable))'''
    for seed in seeds[:11]:
        #analysis.linear_risk_function(seed, small, np.linspace(0, 60, 5), save=True)
        analysis.binary_search_function(seed, small, 0, 50, 5, save=False)