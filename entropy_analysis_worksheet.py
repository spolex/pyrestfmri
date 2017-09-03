# -*- coding: utf-8 -*-

# In[]:
from utils import flatmap,create_dir
from os import path as op
from entropy import f_psd, ssH, f_density, p_entropy, plotAndSavePSD, plotAndSaveEntropy, plotAndSavePermEntropy, bp_filter
import logging
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np

# set up params with default values
TR = 1.94
n_regions= 48
session_list = [1] #sessions start in 1
subject_list = ['T003', 'T004','T006','T013', 'T014', 'T015', 'T017', 'T018',
                'T019', 'T021', 'T023','T024','T025', 'T026', 'T027', 'T028',
                'T029', 'T030', 'T031', 'T032', 'T033', 'T035', 'T039', 'T040',
                'T042', 'T043', 'T045', 'T046', 'T056', 'T058', 'T059', 'T060',
                'T061', 'T062', 'T063', 'T064', 'T065', 'T066', 'T067', 'T068',
                'T069', 'T070', 'T071', 'T072', 'T073', 'T074', 'T075', 'T076',
                'T077', 'T078', 'T079', 'T080', 'T081', 'T082']
# subject_list = ['T003','T004']

logging.debug("Subject ids: " + str(subject_list))

ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz_extracted_ts.csv'

logging.info("Timeseries files path: "+ts_file)

# set up working dirs
data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'
out_dir = 'entropy'

logging.debug("Data location: "+data_dir)
logging.debug("Output will be written: "+op.join(data_dir,'subject_preproc_dir',out_dir))

# In[]
#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
output_dirs = map(lambda session: map(lambda subj:create_dir(op.join(subj,out_dir)),session),input_dirs)
input_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))

logging.debug("Input filenames: ")
logging.debug(input_filenames)

# In[]
# Extract ts for each subject's region 
# Calculate PSD for each subject's region
logging.info("Start PSD calc for each subject's regions...")
subject_region_ts = map(lambda filename: np.genfromtxt(filename, delimiter=',').T,input_filenames)
#subject_region_ts_bandpass_fir = map(lambda region:map(bp_filter,region),subject_region_ts)
subject_region_psd = map(lambda region:map(f_psd,region),subject_region_ts)    
logging.info("Finish PSD calc for each subject's regions...")

# In[]:
# Calculate density function for each subject's region
logging.info("Start density function calc for each subject's regions...")
subject_region_psd_values = map(lambda subject: map(lambda region:region[1],subject),subject_region_psd)
subject_region_f_density = map(lambda subject: map(lambda region:f_density(region),subject),subject_region_psd_values)
logging.info("Finish density function calc for each subject's regions...")

# In[]
# Calculate entropy for each subject's region
logging.info("Start SSE calc for each subject's regions...")
subject_region_entropy = map(lambda subject: map(lambda region:ssH(np.asarray(region)),subject),subject_region_f_density)
logging.info("Finish SSE calc for each subject's regions...")
np.savetxt(op.join(data_dir, "et_ssh_entropy.csv"), subject_region_entropy,delimiter=",")

# In[]
# Calculate permutation entropy for each subject's region
logging.info("Start SSE calc for each subject's regions...")
subject_region_permutation_entropy = map(lambda subject: map(lambda region:p_entropy(region),subject),subject_region_psd_values)
logging.info("Finish SSE calc for each subject's regions...")

np.savetxt(op.join(data_dir, "et_permutation_entropy.csv"), subject_region_permutation_entropy,delimiter=",")

# In[]
# plot and save psd freqs and values
logging.info("Starting plot and save PSD freqs and values for each subject's region...")
for i_session, session in enumerate(session_list):
  logging.debug("Plot and save PSD for session %d ", i_session+1)
  for i_subj,subject in enumerate(subject_region_psd):
    logging.debug("Plot and save PSD for session %d and subject %s", i_session+1,subject_list[i_subj])
    for i_reg,region in enumerate(subject):
      logging.debug("Plot and save PSD for session %d, subject %s and region %d", i_session+1, subject_list[i_subj], i_reg+1)
      freq = region[0]
      val = region[1]
      subj = subject_list[i_subj]
      plotAndSavePSD(freq, val, subj, i_reg+1, output_dirs[i_session][i_subj],session)
logging.info("Finishing plot and save PSD freqs and values for each subject's region...")

# In[]
# plot ans save networks' entropies for each subject
logging.info("Starting plot and save SSE for each subject's region...")
for i_session, session in enumerate(session_list):
  logging.debug("Plot and save SSE for session %d ", i_session+1)
  for i_subj,subject in enumerate(subject_region_entropy):
    logging.debug("Plot and save SSE for session %d and subject %s", i_session+1,subject_list[i_subj])
    subj = subject_list[i_subj]
    plotAndSaveEntropy(subject, subj, output_dirs[i_session][i_subj],session)
logging.info("Finishing plot and save SSE for each subject's region...")

logging.info("Starting plot and save PE for each subject's region...")
for i_session, session in enumerate(session_list):
  logging.debug("Plot and save PE for session %d ", i_session+1)
  for i_subj,subject in enumerate(subject_region_permutation_entropy):
    logging.debug("Plot and save PE for session %d and subject %s", i_session+1,subject_list[i_subj])
    subj = subject_list[i_subj]
    plotAndSavePermEntropy(subject, subj, output_dirs[i_session][i_subj],session)
logging.info("Finishing plot and save PE for each subject's region...")


# In[]
# TODO test permutation entropy implementation