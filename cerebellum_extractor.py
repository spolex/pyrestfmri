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
atlas_filename = experiment["files_path"]["brain_atlas"]["cerebellum"]

masker = NiftiMapsMasker(low_pass=args.lowpass, high_pass=args.highpass, t_r=TR,
                           maps_img=atlas_filename, standardize=args.standarize,
                           memory='nilearn_cache', verbose=args.verbose, smoothing_fwhm=args.fwhm)
masker.fit()
masker_extracted_img = masker.maps_img_
# Total number of regions extracted
masker_n_regions_extracted = masker_extracted_img.shape[-1]

# Visualization of region extraction results
  
title = ('%d regions are extracted from mdls atlas'
        % (masker_n_regions_extracted))
plotting.plot_prob_atlas(masker.maps_img, view_type='filled_contours',
                            title=title,
                           output_file=op.join(cbl_dir,"_func_map_ica_"+str(masker_n_regions_extracted)+".png")
                           )

connectome_measure = ConnectivityMeasure(kind='correlation')
masker_correlations = []
for filename, confound in zip(func_filenames, confounds_components):
    masker_timeseries_each_subject = masker.transform(filename, confounds=confound)
    filename = '/'.join(filename.split('/')[0:-1]) + '/cbl'
    create_dir(filename)
    np.savetxt(filename + "/masker_extracted_ts.csv", masker_timeseries_each_subject, delimiter=",")
    fig = plt.figure()
    plt.plot(masker_timeseries_each_subject)
    plt.xlabel('')
    plt.ylabel('')
    fig.savefig(filename + "/masker_extracted_ts" + ".png")
    plt.close()
    # call fit_transform from ConnectivityMeasure object
    masker_correlation = connectome_measure.fit_transform([masker_timeseries_each_subject])
    # saving each subject correlation to correlations
    masker_correlations.append(masker_correlation)
#
## Mean of all correlations
# masker_mean_correlations = np.mean(masker_correlations, axis=0).reshape(n_regions_extracted,
#                                                          n_regions_extracted)
# In[]
# Mean of all correlations
mask_mean_correlations = np.mean(masker_correlations, axis=0).reshape(len(labels),
                                                                      len(labels))
experiment["files_path"]["cbl_ts_file"] = "cbl/masker_extracted_ts.csv"
config["experiment"]=experiment
update_experiment(config, args.config)