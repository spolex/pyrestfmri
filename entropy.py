# -*- coding: utf-8 -*-

# In[]:


#import nitime modules
from nitime.analysis import SpectralAnalyzer
from nitime.timeseries import TimeSeries

import logging
from os import path as op
from os import makedirs as om
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
import matplotlib.pyplot as plt

# set up params with default values
TR = 1.94
n_regions= 48
session_list = [1] #sessions start in 1
subject_list = ['T003','T004','T006','T013', 'T014', 'T015', 'T017','T019', 'T021', 'T023','T024','T025', 'T026', 'T027', 'T028', 'T029', 'T030', 'T031', 'T032', 'T033']
ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz_extracted_ts.csv' 

# set up working dirs
data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'
out_dir = 'entropy'

# In[]:

#utils
from itertools import chain, imap

def flatmap(f, items):
  return chain.from_iterable(imap(f, items))

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

def plotAndSavePSD(freq,val,subj,region,path,session=0):
  fig = plt.figure()
  plt.plot(freq, val)
  plt.title('PSD for subject '+subj+ ' for region '+str(region))
  plt.xlabel('frequency (hertz)')
  plt.ylabel('Power Spectral Density')
  fig.savefig(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_PSD.png"))
  np.savetxt(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_psd_values.csv"), val, delimiter=",")
  np.savetxt(op.join(path,"session_id_"+str(session)+"_network_"+str(region)+"_psd_freqs.csv"), freq, delimiter=",")

def plotAndSaveEntropy(entropies, subj, path, session):
  fig = plt.figure()
  plt.plot(entropies)
  plt.xlabel('Extracted region')
  plt.ylabel('Spectral Shannon Entropy')
  plt.title('SSE for subject: '+subj)
  fig.savefig(op.join(path,"session_id_"+str(session)+"_ssentropy.png"))
  np.savetxt(op.join(path,"session_id_"+str(session)+"_ssentropy.csv"), entropies, delimiter=",")
  
def f_density(values):
  return map(lambda value: value/values.sum(), values)
    
def createDir(directory):
  if not op.exists(directory):
    om(directory)
  return directory
# In[]
#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
output_dirs = map(lambda session: map(lambda subj:createDir(op.join(subj,out_dir)),session),input_dirs)
input_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))

# In[]
# Extract ts for each subject's region 
# Calculate PSD for each subject's region 
subject_region_ts = map(lambda filename: np.genfromtxt(filename, delimiter=',').T,input_filenames)
subject_region_psd = map(lambda region:map(f_psd,region),subject_region_ts)    

# In[]:
# Calculate density function for each subject's region 
subject_region_psd_values = map(lambda subject: map(lambda region:region[1],subject),subject_region_psd)
subject_region_f_density = map(lambda subject: map(lambda region:f_density(region),subject),subject_region_psd_values)

# In[]
# Calculate entropy for each subject's region 
subject_region_entropy = map(lambda subject: map(lambda region:ssH(np.asarray(region)),subject),subject_region_f_density)

# In[]
# plot ans save psd freqs and values
for i_session, session in enumerate(session_list):
  for i_subj,subject in enumerate(subject_region_psd):
    for i_reg,region in enumerate(subject):
      freq = region[0]
      val = region[1]
      subj = subject_list[i_subj]
      plotAndSavePSD(freq, val, subj, i_reg+1, output_dirs[i_session][i_subj],session)

# In[]
# plot ans save networks' entropies for each subject
for i_session, session in enumerate(session_list):
  for i_subj,subject in enumerate(subject_region_entropy):
    subj = subject_list[i_subj]
    plotAndSaveEntropy(subject, subj, output_dirs[i_session][i_subj],session)
      
