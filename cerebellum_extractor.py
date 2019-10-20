from utils import create_dir,flatmap,experiment_config, update_experiment,plot,plot_connectcome,plot_extracted
import argparse
from nilearn.input_data import NiftiMapsMasker
import logging
import numpy as np
from os import path as op
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Cerebellum atlas based region extractor")
parser.add_argument("-v","--verbose", type=int, help="verbose leevel, default 10", nargs='?', default=10)
parser.add_argument("-e", "--standarize", action="store_true", help="If standardize is True, the time-series are centered and normed: their mean is put to 0 and their variance to 1 in the time dimension.")
parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="conf/config_old.json")
args=parser.parse_args()

# get experiment configuration
config = experiment_config(args.config)
experiment = config["experiment"]

logging.getLogger().setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["r_extractor"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]

# set up files' path
labels = np.array(experiment["labels"])
subject_list = np.array(experiment["subjects_id"])
logging.debug("Subject ids: " + str(subject_list))


ts_image = experiment["files_path"]["ts_image"]
cf_file = experiment["files_path"]["preproc"]["noise_components"]
session_list = [1] # sessions start in 1 TODO allow more than one session

#set up data dirs
subjects_pref = list(map(lambda subject: '_subject_id_'+(subject), subject_list))
sessions_subjects_dir = list(map(lambda session: list(map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref)), session_list))
#flattened all filenames
input_dirs = list(map(lambda session: list(map(lambda subj:op.join(data_dir,subj),session)),sessions_subjects_dir))

#functional images and components confounds
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_image),session), sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session), sessions_subjects_dir))

TR = experiment["t_r"]
atlas_filename = "/home/elekin/datos/TEMPLATES/Cerebellum-MNIfnirt-prob-2mm.nii.gz"

masker = NiftiMapsMasker(maps_img=atlas_filename, memory='nilearn_cache', memory_level=1, detrend=True,
                         verbose=True)
masker.fit()

for filename, confound in zip(func_filenames, confounds_components):
    logging.debug(filename)
    masker_timeseries_each_subject = masker.transform(filename,confounds=confound)
    rdo_dir = '/'.join(filename.split('/')[0:-1])+'/cbl'
    if not op.exists(rdo_dir):
        create_dir(rdo_dir)
    np.savetxt(rdo_dir+"/cbl_extracted_ts.csv",masker_timeseries_each_subject, delimiter=",")
    fig = plt.figure()
    plt.plot(masker_timeseries_each_subject)
    plt.xlabel('')
    plt.ylabel('')
    fig.savefig(rdo_dir+"/masker_extracted_ts" + ".png")
    plt.close()
    # save cbl image
    cbl_filename = rdo_dir + '/cbl_extracted.nii.gz'
    cbl_img = masker.inverse_transform(masker_timeseries_each_subject)
    cbl_img.to_filename(op.join(cbl_filename))
