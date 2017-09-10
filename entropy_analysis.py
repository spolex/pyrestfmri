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
parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="config.json")
parser.add_argument("-p","--psd_save", action="store_true", help="Save calculated PSD for each subject's regions in individual file, 1 activate")
parser.add_argument("-s","--ssh_save", action="store_true", help="Save calculated SSH for each subject's regions in individual file, 1 activate")
parser.add_argument("-e","--pe_save", action="store_true", help="Save calculated SSH for each subject's regions in individual file, 1 activate")
parser.add_argument("-m","--mdsl", action="store_true", help="Header from mdsl labels")

args = parser.parse_args()

# get experiment configuration
experiment = experiment_config(args.config)["experiment"]

# set up logging envvironment
logging.getLogger().setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["entropy"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")


# set up working dirs
data_dir = experiment["files_path"]["preproc_data_dir"]
out_dir = experiment["files_path"]["entropy"]["outdir"]
out_prefix = experiment["split"]

logging.debug("Preprocessed data location: "+data_dir)
logging.debug("Output will be written: "+op.join(data_dir,'subject_preproc_dir',out_dir))

# set up fmri parameters
# time repetition
TR = experiment["t_r"]
# number of regions
n_regions= 38 if args.mdsl else experiment["#regions"]
session_list = [1] #TODO allow more than one session

# set up files' path
subject_list = experiment["subjects_id"]
logging.debug("Subject ids: " + str(subject_list))

# set up ts file path and name from working_dir
filename = '/'.join(experiment["files_path"]["ts_image"].split('/')[0:-1])+'/msdl'
ts_file = op.join(filename,experiment["files_path"]["mdsl_ts_file"]) if args.mdsl else op.join(filename,experiment["files_path"]["ts_file"])
logging.info("Timeseries files path: "+ts_file)

# set up header for file
if args.mdsl:
  from nilearn import datasets
  atlas = datasets.fetch_atlas_msdl()
  labels =atlas["labels"]
  header = labels
else:
  header = map(lambda region: 'reg_'+str(region+1),range(n_regions))

header = 'subj_id,'+','.join(header)
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
subject_region_psd = map(lambda region:map(f_psd,region),subject_region_ts)
logging.info("Finish PSD calc for each subject's regions...")

# In[]:
# Calculate density function for each subject's region
logging.info("Start density function calc for each subject's regions...")
subject_region_psd_values = map(lambda subject: map(lambda region:region[1],subject),subject_region_psd)
subject_region_psd_freqs = map(lambda subject: map(lambda region:region[0],subject),subject_region_psd)

subject_region_f_density = map(lambda subject: map(lambda region:f_density(region),subject),subject_region_psd_values)
logging.info("Finish density function calc for each subject's regions...")
#save results into file
#np.savetxt(op.join(data_dir, "et_psd.csv"), np.column_stack((subject_list,subject_region_psd_values)),
#           delimiter=",", fmt="%s", header=header, comments='')

# In[]
# Calculate entropy for each subject's region
logging.info("Start SSE calc for each subject's regions...")
subject_region_entropy = map(lambda subject: map(lambda region:ssH(np.asarray(region)),subject),subject_region_f_density)
#subject_region_entropy = map(lambda region:map(ssH2,region),subject_region_ts)
logging.info("Finish SSE calc for each subject's regions...")
#save results into file
#np.savetxt(op.join(data_dir, out_prefix+"_ssh_entropy.csv"), np.column_stack((subject_list,subject_region_entropy)),
#           delimiter=",", fmt="%s", header=header, comments='')

np.savetxt(op.join(data_dir, out_prefix+"_ssh_entropy.csv"), np.column_stack((subject_list,subject_region_entropy)),
           delimiter=",", fmt="%s", header=header, comments='')

# In[]
# Calculate permutation entropy for each subject's region
logging.info("Start PE calc for each subject's regions...")
subject_region_permutation_entropy = map(lambda subject: map(lambda region:p_entropy(region),subject),subject_region_psd_values)
logging.info("Finish PE calc for each subject's regions...")
#save results into file
np.savetxt(op.join(data_dir, out_prefix+"_permutation_entropy.csv"), np.column_stack((subject_list,subject_region_permutation_entropy)),
            delimiter=",", fmt="%s", header=header, comments='')

# In[]
# plot and save psd freqs and values
if args.psd_save:
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
if args.ssh_save:
    logging.info("Starting plot and save SSE for each subject's region...")
    for i_session, session in enumerate(session_list):
      logging.debug("Plot and save SSE for session %d ", i_session+1)
      for i_subj,subject in enumerate(subject_region_entropy):
        logging.debug("Plot and save SSE for session %d and subject %s", i_session+1,subject_list[i_subj])
        subj = subject_list[i_subj]
        plotAndSaveEntropy(subject, subj, output_dirs[i_session][i_subj],session)
    logging.info("Finishing plot and save SSE for each subject's region...")

if args.pe_save:
    logging.info("Starting plot and save PE for each subject's region...")
    for i_session, session in enumerate(session_list):
      logging.debug("Plot and save PE for session %d ", i_session+1)
      for i_subj,subject in enumerate(subject_region_permutation_entropy):
        logging.debug("Plot and save PE for session %d and subject %s", i_session+1,subject_list[i_subj])
        subj = subject_list[i_subj]
        plotAndSavePermEntropy(subject, subj, output_dirs[i_session][i_subj],session)
    logging.info("Finishing plot and save PE for each subject's region...")

logging.info("Finish entropy analysis ;-)")