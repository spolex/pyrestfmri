# -*- coding: utf-8 -*-

# In[]:

from utils import flatmap,create_dir
from os import path as op
from entropy import f_psd, ssH, f_density, plotAndSavePSD, plotAndSaveEntropy, bp_filter
import logging
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np

# set up params with default values
TR = 1.94
n_regions= 48
session_list = [1] #sessions start in 1
#subject_list = ['T003', 'T004','T006','T013', 'T014', 'T015', 'T017', 'T018', 
#                'T019', 'T021', 'T023','T024','T025', 'T026', 'T027', 'T028', 
#                'T029', 'T030', 'T031', 'T032', 'T033', 'T035', 'T039', 'T040',
#                'T042', 'T043', 'T045', 'T046', 'T056', 'T058', 'T059', 'T060',
#                'T061', 'T062', 'T063', 'T064', 'T065', 'T066', 'T067', 'T068',
#                'T069', 'T070', 'T071', 'T072', 'T073', 'T074', 'T075', 'T076',
#                'T077', 'T078', 'T079', 'T080', 'T081', 'T082']
subject_list = ['T026']
ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz_extracted_ts.csv' 

# set up working dirs
data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'
out_dir = 'entropy'

# In[]
#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
output_dirs = map(lambda session: map(lambda subj:create_dir(op.join(subj,out_dir)),session),input_dirs)
input_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))

# In[]
# Extract ts for each subject's region 
# Calculate PSD for each subject's region 
subject_region_ts = map(lambda filename: np.genfromtxt(filename, delimiter=',').T,input_filenames)
subject_region_ts_bandpass_fir = map(lambda region:map(bp_filter,region),subject_region_ts) 
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
      
# In[]
