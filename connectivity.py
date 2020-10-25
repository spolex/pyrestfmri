#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Iñigo Sanchez Mendez
Functional connectivity extractor tool, CanICA and DictLearn based (FastICA)

"""

from __future__ import print_function, division, unicode_literals, absolute_import
from utils import create_dir,flatmap, experiment_config, update_experiment
import os.path as op
import numpy as np
from nipype import config, logging as nl
import argparse
import os
import logging

parser = argparse.ArgumentParser(description="Functional Brain Atlas extractor tool")

parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="conf/config_test.json")
parser.add_argument("-f","--canica", action="store_true", help="Build CanICA based functional brain atlas. default false")
parser.add_argument("-d","--dictlearn", action="store_true", help="Build Dictlearn based functional brain atlas. default true")
parser.add_argument("-n","--n_components", type=int, help="Number of components to build functional networks, default 20", nargs='?', default=20)
parser.add_argument("-j","--n_jobs", type=int, help="Parallelism degree, default all CPUs except one (-2)", nargs='?', default=1)
parser.add_argument("-s","--fwhm", type=float, help="Smooth threshold, default None", nargs='?', default=None)
parser.add_argument("-i","--highpass", type=float, help="high pass filter, default none", nargs='?', default=None)
parser.add_argument("-l","--lowpass", type=float, help="low pass filter, default none", nargs='?', default=None)
parser.add_argument("-v","--verbose", type=int, help="Default max: 10", nargs='?', default=1)
parser.add_argument("-e", "--standarize", action="store_true", help="If standardize is True, the time-series are centered and normed: their mean is put to 0 and their variance to 1 in the time dimension.")
parser.add_argument("-b","--subjects", type=str, help="File location contains the list of subjects to calculate connectivity", nargs='?',default="conf/et.txt")

args = parser.parse_args()

fwhm = 'none' if args.fwhm is None else str(int(args.fwhm))

# get experiment configuration
local_config=experiment_config(args.config)
experiment = local_config["experiment"]

# logging config

#logging
log_dir = experiment["files_path"]["brain_atlas"]["log"]
log_level = experiment["log_level"]
msg_format ="%(asctime)s - %(levelname)s - %(process)s - {%(pathname)s:%(lineno)d}- %(message)s"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
config.enable_debug_mode()
config.set_log_dir(log_dir)
config.update_config({'logging': {'log_directory': log_dir,
                                  'log_to_file': True}})
nl.update_logging(config)

logging.basicConfig(filename=log_dir+'/main.log', filemode ='w', format=msg_format)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


logger.info("Functional brain connectivity extraction")
logger.debug("Configuration file is "+args.config)
logger.debug("Smoothing parameter is "+str(args.fwhm))
logger.debug("Experiment loaded")

# set up files' path
subject_list = np.genfromtxt(args.subjects,dtype="str",skip_header=1)
logger.debug("Subject ids: " + str(subject_list))

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]

#In[]:read components image from file
TR = experiment["t_r"]
session_list = [1]

# set up ts file path and name from working_dir
ts_file = experiment["files_path"]["ts_image"]
ts_filename = ts_file.split("/")
ts_filename = ts_filename[0]
logger.info("Timeseries files path: "+ts_file)
cf_file = experiment["files_path"]["preproc"]["noise_components"]

#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session),sessions_subjects_dir))

# update configuration file used number of components
experiment["#components"] = args.n_components
local_config["experiment"] = experiment
update_experiment(local_config, args.config)
split = experiment["split"]

# Extract resting-state networks with CanICA

# ---------------------------------
if(args.canica):
    from nilearn.decomposition import CanICA

    logger.info("Canica algorithm starting...")
    canica = CanICA(n_components=args.n_components, smoothing_fwhm=args.fwhm,
                   memory_level=2, threshold=3., verbose=args.verbose, random_state=0,
                   n_jobs=args.n_jobs, t_r=TR, standardize=args.standarize,
                   high_pass=args.highpass, low_pass=args.lowpass)
    canica.fit(func_filenames, confounds=confounds_components)

    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    # components_img is a Nifti Image object, and can be saved to a file with
    # the following line:
    canica_dir = data_dir+'/'+split+'/'+'canica'
    create_dir(canica_dir)
    filename = canica_dir+'/'+fwhm+'_resting_state_all.nii.gz'
    # save components image
    components_img.to_filename(op.join(data_dir,filename))
    # update configuration file
    experiment["files_path"]["brain_atlas"]["components_img"] = filename
    local_config["experiment"] = experiment

    # -----------------------------------------------------------------
    from nilearn.plotting import plot_prob_atlas

    # Plot all ICA components together
    plot_prob_atlas(components_img, title='CanICA based Brain Atlas', view_type="filled_contours",
                    output_file=op.join(canica_dir,fwhm+'_resting_state_all_plot_prob_atlas'))
# In[]
# ------------------------------------------------------------------
if(args.dictlearn):
    # Extract resting-state networks with DictionaryLearning

    # Import dictionary learning algorithm from decomposition module and call the
    # object and fit the model to the functional datasets
    from nilearn.decomposition import DictLearning
    logger.info("Dict Learning algorithm starting...")


    # Initialize DictLearning object
    dict_learn = DictLearning(n_components=args.n_components, verbose=args.verbose,
                              smoothing_fwhm=args.fwhm, t_r=TR, alpha=10, memory_level=2,
                              random_state=0, n_jobs=args.n_jobs, standardize=args.standarize,
                              high_pass=args.highpass, low_pass=args.lowpass)
    # Fit to the data
    dict_learn.fit(func_filenames, confounds=confounds_components)
    # Resting state networks/maps
    components_img_dic = dict_learn.masker_.inverse_transform(dict_learn.components_)
    
    dict_dir = data_dir+'/'+split+'/'+'dict'
    create_dir(dict_dir)
    filename =dict_dir+'/'+fwhm+'_resting_state_all.nii.gz'
    # save components image
    components_img_dic.to_filename(op.join(data_dir, filename))
    # update configuration file
    experiment["files_path"]["brain_atlas"]["components_img_1"] = filename
    local_config["experiment"] = experiment

    # Visualization of resting state networks
    # Show networks using plotting utilities
    from nilearn import plotting

    plotting.plot_prob_atlas(components_img_dic, view_type='filled_contours',
                             title='Dictionary Learning based Brain Atlas',
                             output_file=op.join(dict_dir,fwhm+'_resting_state_all_plot_prob_atlas'))

update_experiment(local_config, args.config)