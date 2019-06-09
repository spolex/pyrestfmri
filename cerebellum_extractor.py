# In[]:
from utils import create_dir,flatmap,experiment_config, update_experiment,plot,plot_connectcome,plot_extracted

#import nilearn modules
from nilearn.connectome import ConnectivityMeasure

from os import path as op
import os
import logging
import numpy as np
from nilearn import (image, plotting)
from nilearn.regions import RegionExtractor
from nilearn.input_data import NiftiMapsMasker
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Functional atlas based region extractor")
parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="conf/config.json")
parser.add_argument("-i","--highpass", type=float, help="high pass filter, default none", nargs='?', default=None)
parser.add_argument("-l","--lowpass", type=float, help="low pass filter, default none", nargs='?', default=None)
parser.add_argument("-v","--verbose", type=int, help="verbose leevel, default 10", nargs='?', default=10)
parser.add_argument("-e", "--standarize", action="store_true", help="If standardize is True, the time-series are centered and normed: their mean is put to 0 and their variance to 1 in the time dimension.")
parser.add_argument("-s","--fwhm", type=float, help="Smooth threshold, default None", nargs='?', default=None)

args=parser.parse_args()

# get experiment configuration
config = experiment_config(args.config)
experiment = config["experiment"]

logging.getLogger().setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["r_extractor"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")

# set up files' path
labels = np.array(experiment["labels"])
subject_list = np.array(experiment["subjects_id"])
logging.debug("Subject ids: " + str(subject_list))

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]


#In[]:read components image from file

TR = experiment["t_r"]
session_list = [1] # sessions start in 1 TODO allow more than one session

# set up ts file path and name from working_dir
ts_image = experiment["files_path"]["ts_image"]
logging.info("Timeseries image path: "+ts_image)
cf_file = experiment["files_path"]["preproc"]["noise_components"]
cbl_dir = data_dir+'/'+'cerebelo'
create_dir(cbl_dir)


#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)

#functional images and components confounds
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_image),session),sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session),sessions_subjects_dir))

# Loading atlas image stored in 'maps'
cbl_img = experiment["files_path"]["brain_atlas"]["cerebellum"]

#Region extracted
extractor = RegionExtractor(cbl_img, verbose=args.verbose, thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions', memory="nilearn_cache", memory_level=2,
                            t_r=TR, high_pass=args.highpass, low_pass=args.lowpass, standardize=args.standarize,
                            min_region_size=args.region_size)

extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]
# Each region index is stored in index_
regions_index = extractor.index_

