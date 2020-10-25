from commons.viz import flatmap,experiment_config
import argparse
from nilearn.input_data import NiftiMapsMasker
import logging
import numpy as np
from os import path as op
from fmriapi import extract_cbl

parser = argparse.ArgumentParser(description="Cerebellum atlas based region extractor")
parser.add_argument("-v","--verbose", help="verbose", action='store_true')
parser.add_argument("-e", "--standarize", action="store_true", help="If standardize is True, the time-series are "
                                                                    "centered and normed: their mean is put to 0 and "
                                                                    "their variance to 1 in the time dimension.")
parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="conf/config.json")
parser.add_argument("-m",'--maps', help='Use mapsmasker instead of masker', action='store_true')
args = parser.parse_args()

# get experiment configuration
config = experiment_config(args.config)
experiment = config["experiment"]

logging.getLogger("cbl_extractor").setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["cbl_extractor"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]

# set up files' path
labels = np.array(experiment["labels"])
subject_list = np.array(experiment["subjects_id"])
logging.debug("Subject ids: " + str(subject_list))
ts_image = experiment["files_path"]["ts_image"]
cf_file = experiment["files_path"]["preproc"]["noise_components"]
session_list = [1]
TR = experiment["t_r"]
atlas_filename = experiment["files_path"]["cbl_extractor"]["cbl_atlas"]

#set up data dirs
subjects_pref = list(map(lambda subject: '_subject_id_'+(subject), subject_list))
sessions_subjects_dir = list(map(lambda session: list(map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref)), session_list))
#flattened all filenames
input_dirs = list(map(lambda session: list(map(lambda subj:op.join(data_dir,subj),session)),sessions_subjects_dir))

#functional images and components confounds
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_image),session), sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session), sessions_subjects_dir))

masker = NiftiMapsMasker(maps_img=atlas_filename, memory_level=1, detrend=True, verbose=args.verbose, t_r=TR)
masker.fit()

for i, (filename, confound) in enumerate(zip(func_filenames, confounds_components)):
    extract_cbl(filename, confound, masker, op.join(data_dir,"cbl",subject_list[i]))
