# -*- coding: utf-8 -*-

#import nitime modules
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer
from nitime.timeseries import TimeSeries

import matplotlib.pyplot as plt
import numpy as np
from os import path as op

import logging

def f_psd(ts, TR=1.94):
    """
    Calculate power spectra of time serie
    :param ts:
    :param TR:
    :return:
    """
    timeserie = TimeSeries(ts, sampling_interval=TR)
    s_analyzer = SpectralAnalyzer(timeserie)
    return s_analyzer.psd

def f_density(values):
    """
    Density function calculation from numpy array
    :param values:
    :return:
    """
    return map(lambda value: value/values.sum(), values)

def bp_filter(ts, pub=0.15, plb=0.02, TR=1.94):
    """
    TODO: timeseries band pass filter
    :param ts:
    :param pub:
    :param plb:
    :param TR:
    :return:
    """
    F = FilterAnalyzer(ts, ub=pub, lb=plb)
    return F.fir

def ssH(density_func):
    """
    spectral shannon entropy implementation
    :param density_func:
    :return:
    """
    if not density_func.any():
        return 0.
    entropy = 0.
    for density in density_func:
        if density > 0:
            entropy += density*np.log(1/density)
    return entropy

def p_entropy(ts, order=3, lag=3):
    """
    Calculate the Permutation Entropy permutation entropy implementation
    [1]Permutation entropy: a natural complexity measure for time series.Bandt C1, Pompe B
        https://www.ncbi.nlm.nih.gov/pubmed/12005759
    [2]http://tocsy.pik-potsdam.de/petropy.php
    [3]http://es.mathworks.com/matlabcentral/fileexchange/37289-permutation-entropy?focused=3770660&tab=function

    :param ts: time series for analysis
    :param order: order of permutation
    :param lag: time delay
    :return:
    """
    from itertools import permutations
    n = len(ts)
    permutations = np.array(list(permutations(range(order))))
    c = [0] * len(permutations)

    for i in range(n - lag * (order - 1)):
        sorted_index_array = np.array(np.argsort(ts[i:i + lag * order:lag], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe

#########################PLOT AND SAVE FUNCTIONS######################################

def plotAndSavePSD(freq,val,subj,region,path,session=0):
    """
    Plot and save power spectral results
    :param freq:
    :param val:
    :param subj:
    :param region:
    :param path:
    :param session:
    :return:
    """
    fig = plt.figure()
    plt.plot(freq, val)
    plt.title('PSD for subject '+subj+ ' for region '+str(region))
    plt.xlabel('frequency (hertz)')
    plt.ylabel('Power Spectral Density')
    fig.savefig(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_PSD.png"))
    np.savetxt(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_psd_values.csv"), val, delimiter=",")
    np.savetxt(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_psd_freqs.csv"), freq, delimiter=",")
    plt.close()


def plotAndSaveEntropy(entropies, subj, path, session):
    """
    Plot and save entropy
    :param entropies: 
    :param subj: 
    :param path: 
    :param session: 
    :return: 
    """
    fig = plt.figure()
    plt.plot(entropies)
    plt.xlabel('Extracted region')
    plt.ylabel('Spectral Shannon Entropy')
    plt.title('SSE for subject: '+subj)
    fig.savefig(op.join(path,"session_id_"+str(session)+"_ssentropy.png"))
    np.savetxt(op.join(path,"session_id_"+str(session)+"_ssentropy.csv"), entropies, delimiter=",")
    plt.close()

def plotAndSavePermEntropy(entropies, subj, path, session):
    """
    Plot and save entropy
    :param entropies:
    :param subj:
    :param path:
    :param session:
    :return:
    """
    fig = plt.figure()
    plt.plot(entropies)
    plt.xlabel('Extracted region')
    plt.ylabel('Permutation Entropy')
    plt.title('PE for subject: '+subj)
    fig.savefig(op.join(path,"session_id_"+str(session)+"_pentropy.png"))
    np.savetxt(op.join(path,"session_id_"+str(session)+"_pentropy.csv"), entropies, delimiter=",")
    plt.close()