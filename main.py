# MSci Project
# Snigdha Sen and Sophie Martin
 
from models import RandomBrain, SmallWorldBrain, GrowPrefAttachmentBrain, BarabasiAlbertBrain
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sp
import time
import analysis 
import glob
import network_comparison
import models
import pickle
import networkx as nx
from logbin230119 import logbin
from scipy.interpolate import CubicSpline
import matplotlib.animation as animation
import profile

# Set random seeds using both methods
np.random.seed(100)
random.seed(100)

N = 10000
t = 1000
'''
##################################
# Plotting the Cluster Path length variation for different N
##################################

plt.figure(figsize=(8,6))
a=0.095
ps = np.logspace(-3, -1, 15)
ps = np.append(ps, np.linspace(0.15, 1, 15))

names = ['100', '1000', '10000']
colors= ['black', 'orange', 'brown']
for idx, files in enumerate([['results/k_5/ccoeff_100.npy','results/k_5/pathlength_100.npy'], ['results/k_5/ccoeff_1000.npy','results/k_5/pathlength_1000.npy'], ['results/k_5/ccoeff_10000.npy','results/k_5/pathlength_10000.npy']]):
    cs = np.load(files[0])
    ls = np.load(files[1])
    cubicspline = CubicSpline(ps, cs/ls)
    ps_smooth = np.linspace(0.0005, 1, 100)
    plt.plot(ps, cs/ls, 'o', color=colors[idx])
    plt.plot(ps_smooth, cubicspline(ps_smooth), '-', color=colors[idx], label='N='+names[idx])

plt.xlabel('$p_r$', size=16)
plt.legend(prop={'size': 16})
plt.tick_params(labelsize=16)
plt.axvline(a, color='red', alpha=0.8)
plt.ylabel('$C/L$', size=16)

##################################
# Plotting the Cluster Path length variation for 10000 nodes with shaded region
##################################

network_comparison.run_c_l_p_variation(N, load=True, filename=['results/k_5/ccoeff_10000.npy', 'results/k_5/pathlength_10000.npy'], save=False, plot=True)

##################################
# Running a binary search
##################################

small = SmallWorldBrain(numberOfNodes=N, averageDegree=5, rewireProbability=0.15, neuronThreshold=1, refractoryPeriod=1, 
numberOfIterations=t, fractionToFire=0.05, inhibitoryEdgeMultiplier=0, fireRate=0.01, directed=False)
results = analysis.run_binary_search_risk_curve(small, numberOfExperiments=10, m_min=0, m_max=100, precision=5, plot=True)

with open('results/risk_curves/risk_curve_10000_2404_2.p', 'wb') as outfile:
    pickle.dump(results, outfile) 


##################################
# Averaging over repeats for CL plot
##################################

plt.figure(figsize=(8,6))
ps = np.logspace(-3, -1, 15)
ps = np.append(ps, np.linspace(0.15, 1, 15))

ccoeffnames = ['ccoeffs_100_*', 'ccoeffs_*', 'ccoeffs_10000_*']
pathlengthnames = ['pathlength_100_*', 'pathlength_*', 'pathlength_10000_*']

colors= ['black', 'orange', 'brown']
labels = ['100', '1000', '10000']

for nameidx in range(len(ccoeffnames)):
    fileListCcoeff = glob.glob('results/cp_repeats/'+str(ccoeffnames[nameidx]))
    fileListPathlength = glob.glob('results/cp_repeats/'+str(pathlengthnames[nameidx]))

    ratios = []
    for idx in range(len(fileListCcoeff)):
        cs = np.load(fileListCcoeff[idx])
        ls = np.load(fileListPathlength[idx])
        ratios.append(cs/ls)

    standarderrors = []
    meanratios= []
    for p_num in range(len(ps)):
        standarderror = np.std(np.array(ratios)[:, p_num])
        meanratio = np.mean(np.array(ratios)[:, p_num])
        standarderrors.append(standarderror)
        meanratios.append(meanratio)

    cubicspline = CubicSpline(ps, meanratios)
    polyfit = np.polyfit(ps, meanratios, 9)
    f = np.poly1d(polyfit)
    ps_smooth = np.linspace(0.0005, 1, 100)


    polyfit = np.polyfit(ps,  np.array(meanratios)-(np.array(standarderrors))/np.sqrt(10), 9)
    f_minus_error = np.poly1d(polyfit) 

    polyfit = np.polyfit(ps,  np.array(meanratios)+(np.array(standarderrors))/np.sqrt(10), 9)
    f_plus_error = np.poly1d(polyfit) 

    print(f_minus_error(ps_smooth))
    plt.plot(ps, meanratios, 'o', color=colors[nameidx], label='N='+str(labels[nameidx]))
    plt.fill_between(ps_smooth, f_minus_error(ps_smooth), f_plus_error(ps_smooth), alpha=0.4, color=colors[nameidx])
    #plt.plot(ps_smooth[1:], cubicspline(ps_smooth)[1:], '-', color=colors[nameidx], label='Cubic Spline N='+str(labels[nameidx]))

    plt.plot(ps_smooth[1:], f(ps_smooth)[1:], '-', color=colors[nameidx])

idxMax = np.where(f(ps_smooth) == max(f(ps_smooth)))[0]
print(ps_smooth[idxMax])
plt.axvline(ps_smooth[idxMax], color='red', alpha=0.8)
plt.xlabel('$p_r$', size=16)
plt.legend(prop={'size': 16})
plt.tick_params(labelsize=16)
plt.ylabel('$C/L$', size=16)


##################################
# Plot regimes and calculate mean Q 
##################################
analysis.plot_critical_regimes(N, t)

##################################
# Pull risk curve data into a single result
##################################

plt.figure(figsize=(8,6))
from collections import defaultdict


def curve_fit(x, a, c):
    return (1 - a/(c+np.exp(-0.005*x)))

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

fileList= glob.glob('epilepsy/multiprocess/results_m_*')
#linearfileList = glob.glob('epilepsy/multiprocess/linear/results_m_*')
multiplierArray=[]
sysProbs= []

for filename in [fileList[8]]:
    binary_data = np.load(filename)
    #linear_data = np.load(linearfileList[idx])
    
    locs =  list(list_duplicates(binary_data[0]))
    if len(locs)>0:
        for l in locs:
            vals =  []
            rs = []
            for idx, _ in enumerate(l[1]):
                vals.append(binary_data[1][l[1][idx]])
            if l[0] < 20:
                lowest = np.argwhere(vals == min(vals))[:,0]
                for h in lowest:
                    binary_data[0][l[1][h]]=np.nan
                    binary_data[1][l[1][h]]=np.nan
            if l[0] > 20:
                highest = np.argwhere(vals == max(vals))[:,0]
                for h in highest:
                    binary_data[0][l[1][h]]=np.nan
                    binary_data[1][l[1][h]]=np.nan
    
    multiplierArray.append(binary_data[0])
    sysProbs.append(binary_data[1])

standarderrors = []
meansysProbs= []
fractionaverage = np.nanmean(np.array(multiplierArray), axis=0)

for fracidx in range(len(fractionaverage)):
    meansysProb = np.nanmean(np.array(sysProbs)[:, fracidx])
    standarderror = np.nanstd(np.array(sysProbs)[:, fracidx])
    standarderrors.append(standarderror)
    meansysProbs.append(meansysProb)

n=22
remove = []
#remove = [4,8]
a = np.delete(fractionaverage, remove)
b =  np.delete(meansysProbs,remove)
c = np.delete(standarderrors,remove)
fractionaverage = a
meansysProbs = b
standarderrors = c
fractionaverage, meansysProbs = zip(*sorted(zip(fractionaverage, meansysProbs), key=lambda pair: pair[0]))

s =1
e = 9
#plt.plot([0]+list(fractionaverage),[1]+list(meansysProbs), '+', color='black')
plt.errorbar([0]+list(fractionaverage)[1:], [1]+list(meansysProbs)[1:], [0]+list(np.array(standarderrors)/np.sqrt(n))[1:], color='black', marker='+', linestyle='-', capsize=3, label='Data')
plt.tick_params(labelsize=16)
#plt.xlabel(r'$\alpha$', size=16)
plt.xlabel(r'$\alpha$', size=16)
#plt.xlabel(r'$\left\langle k_{inhibitory} \right\rangle$', size=16)
plt.ylabel('$P_s$', size=16)

sub_axes = plt.axes((.22, .25, .35, .35)) 
sub_axes.set_ylim(0, 1)
sub_axes.errorbar(fractionaverage[s:e], meansysProbs[s:e], standarderrors[s:e]/np.sqrt(n), marker='+', color='black')
#sub_axes.fill_between(fractionaverage[s:e], np.array(meansysProbs[s:e])-((np.array(standarderrors[s:e]))/np.sqrt(n)), np.array(meansysProbs[s:e])+(np.array(standarderrors[s:e]))/np.sqrt(n), alpha=0.4, color='black')
sub_axes.set_xlabel(r'$f_{eff}$', size=16)
sub_axes.set_ylabel('$P_s$', size=16)


#popt, pcov = sp.optimize.curve_fit(curve_fit, [0]+list(fractionaverage), [1]+list(meansysProbs))
#print(popt)
#plt.plot(np.linspace(0,max(fractionaverage), 30), curve_fit(np.linspace(0,max(fractionaverage), 30), *popt), 'r-', label='Sigmoid Fit')
#plt.legend(prop={'size':16})

##################################
# Plot alpha vs k 
##################################

plt.figure(figsize=(8,6))
from collections import defaultdict


fileList= glob.glob('epilepsy/multiprocess/results_r_*')
#linearfileList = glob.glob('epilepsy/multiprocess/linear/results_m_*')
multiplierArray=[]

for filename in fileList:
    binary_data = np.load(filename)
    multiplierArray.append(binary_data[0])

multiplierArray2=[]

fileList= glob.glob('epilepsy/multiprocess/results_m_*')
#linearfileList = glob.glob('epilepsy/multiprocess/linear/results_m_*')

for filename in fileList:
    binary_data = np.load(filename)
    multiplierArray2.append(binary_data[0])

meank =np.nanmean(np.array(multiplierArray), axis=0)
meanalpha = np.nanmean(np.array(multiplierArray2), axis=0)
stdk = np.nanstd(np.array(multiplierArray), axis=0)
stdalpha = np.nanstd(np.array(multiplierArray2), axis=0)

plt.plot([0]+meanalpha[1:], [0]+meank[1:], color='black', markersize=10, marker='.', linestyle='None')
plt.ylabel(r'$\left\langle k_{inhibitory} \right\rangle$', size=20)
plt.xlabel(r'$\alpha$', size=20)

plt.tick_params(labelsize=16)
#[3.53653331e-05 2.55972933e+00 1.00585982e+00 3.79428836e-05] - raw

'''

analysis.plot_critical_regimes(N, t)

plt.show()