# -*- coding: utf-8 -*-

# In[]:
#import nitime modules
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer
from nitime.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from os import path as op

#import logging
#logging.getLogger().setLevel(logging.DEBUG)


# In[]:

#functions      
#spectral shannon entropy implementation 
def ssH(density_func):
    if not density_func.any():
        return 0.
    entropy = 0.
    for density in density_func:
        if density > 0:
            entropy += density*np.log(1/density)
    return entropy

#psd function
def f_psd(ts, TR=1.94):
  timeserie = TimeSeries(ts, sampling_interval=TR)
  s_analyzer = SpectralAnalyzer(timeserie)
  return s_analyzer.psd
  
def f_density(values):
  return map(lambda value: value/values.sum(), values)

def bp_filter(ts, pub=0.15, plb=0.02, TR=1.94):
  timeserie = TimeSeries(ts, sampling_interval=TR)
  #F = FilterAnalyzer(timeserie, ub=pub, lb=plb)
  #return F.fir
  return timeserie
    
#########################PLOT FUNCTIONS######################################
def plotAndSavePSD(freq,val,subj,region,path,session=0):
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
  fig = plt.figure()
  plt.plot(entropies)
  plt.xlabel('Extracted region')
  plt.ylabel('Spectral Shannon Entropy')
  plt.title('SSE for subject: '+subj)
  fig.savefig(op.join(path,"session_id_"+str(session)+"_ssentropy.png"))
  np.savetxt(op.join(path,"session_id_"+str(session)+"_ssentropy.csv"), entropies, delimiter=",")
  plt.close()