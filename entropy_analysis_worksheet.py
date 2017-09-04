# -*- coding: utf-8 -*-

# In[]:
from utils import flatmap,create_dir,experiment_config
from os import path as op
import numpy as np
from entropy import f_psd, ssH, f_density, p_entropy, plotAndSavePSD, plotAndSaveEntropy, plotAndSavePermEntropy
import logging
import argparse

# set up argparser
parser = argparse.ArgumentParser(description="Process Shannon entropy and Permutation entropy")
parser.add_argument("config", type=str, help="Configuration file path", nargs='?', default="config.json")
parser.add_argument("psd_save", type=bool, help="Save calculated PSD for each subject's regions in individual file", nargs='?', default=False)
parser.add_argument("ssh_save", type=bool, help="Save calculated SSH for each subject's regions in individual file", nargs='?', default=False)
parser.add_argument("pe_save", type=bool, help="Save calculated SSH for each subject's regions in individual file", nargs='?', default=False)
args = parser.parse_args()

# get experiment configuration
experiment = experiment_config()["experiment"]

# set up envvironment
logging.getLogger().setLevel(experiment["log_level"])

# set up working dirs
data_dir = experiment["file_paths"]["data_dir"]
out_dir = experiment["file_paths"]["entropy_outdir"]
logging.debug("Data location: "+data_dir)
logging.debug("Output will be written: "+op.join(data_dir,'subject_preproc_dir',out_dir))

# set up fmri parameters
TR = experiment["t_r"]
n_regions= experiment["#regions"]
session_list = [1] #TODO allow more than one session

# set up files' path
subject_list = experiment["subject_ids"]
logging.debug("Subject ids: " + str(subject_list))

# set up ts file path and name from working_dir
ts_file = experiment["file_paths"]["ts_file"]
logging.info("Timeseries files path: "+ts_file)

# set up
header = map(lambda region: 'reg_'+str(region+1),range(n_regions))
header = 'subject_id,'+','.join(header)
logging.debug("To save entropy results header will be: "+header)

# In[]
#set up subject's input data
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
#save results into file
np.savetxt(op.join(data_dir, "et_psd.csv"), np.column_stack((subject_list,subject_region_psd_values)),
           delimiter=",", fmt="%s", header=header, comments='')

# In[]
# Calculate entropy for each subject's region
logging.info("Start SSE calc for each subject's regions...")
subject_region_entropy = map(lambda subject: map(lambda region:ssH(np.asarray(region)),subject),subject_region_f_density)
logging.info("Finish SSE calc for each subject's regions...")
#save results into file
np.savetxt(op.join(data_dir, "et_ssh_entropy.csv"), np.column_stack((subject_list,subject_region_entropy)),
           delimiter=",", fmt="%s", header=header, comments='')

# In[]
# Calculate permutation entropy for each subject's region
logging.info("Start SSE calc for each subject's regions...")
subject_region_permutation_entropy = map(lambda subject: map(lambda region:p_entropy(region),subject),subject_region_psd_values)
logging.info("Finish SSE calc for each subject's regions...")
#save results into file
np.savetxt(op.join(data_dir, "et_permutation_entropy.csv"), np.column_stack((subject_list,subject_region_permutation_entropy)),
            delimiter=",", fmt="%s", header=header, comments='')

# In[]
# plot and save psd freqs and values
if(args.psd_save):
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
if(args.ssh_save):
    logging.info("Starting plot and save SSE for each subject's region...")
    for i_session, session in enumerate(session_list):
      logging.debug("Plot and save SSE for session %d ", i_session+1)
      for i_subj,subject in enumerate(subject_region_entropy):
        logging.debug("Plot and save SSE for session %d and subject %s", i_session+1,subject_list[i_subj])
        subj = subject_list[i_subj]
        plotAndSaveEntropy(subject, subj, output_dirs[i_session][i_subj],session)
    logging.info("Finishing plot and save SSE for each subject's region...")

if(args.pe_save):
    logging.info("Starting plot and save PE for each subject's region...")
    for i_session, session in enumerate(session_list):
      logging.debug("Plot and save PE for session %d ", i_session+1)
      for i_subj,subject in enumerate(subject_region_permutation_entropy):
        logging.debug("Plot and save PE for session %d and subject %s", i_session+1,subject_list[i_subj])
        subj = subject_list[i_subj]
        plotAndSavePermEntropy(subject, subj, output_dirs[i_session][i_subj],session)
    logging.info("Finishing plot and save PE for each subject's region...")