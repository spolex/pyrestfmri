#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: spolex

Functional brain atlas extractor tool, CanICA and DictLearn based (FastICA)

"""

# In[]:
from utils import flatmap, experiment_config
import os.path as op
import argparse
import logging

parser = argparse.ArgumentParser(description="Functional Brain Atlas extractor tool")

parser.add_argument("config", type=str, help="Configuration file path", nargs='?', default="conf/config.json")
parser.add_argument("canica", type=bool, help="Build CanICA based functional brain atlas. default false", nargs='?', default=False)
parser.add_argument("dictlearn", type=bool, help="Build Dictlearn based functional brain atlas. default true", nargs='?', default=True)
parser.add_argument("n_components", type=int, help="Number of components to build functional networks, default 20", nargs='?', default=20)
parser.add_argument("n_jobs", type=int, help="Parallelism degree, default all CPUs except one (-2)", nargs='?', default=-2)
parser.add_argument("fwhm", type=int, help="Smooth threshold, default None", nargs='?', default=None)
parser.add_argument("verbose", type=int, help="Default max: 10", nargs='?', default=10)



args=parser.parse_args()

# get experiment configuration
experiment = experiment_config(args.config)["experiment"]

logging.getLogger().setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["brain_atlas_extractor_log_file"])

# set up files' path
subject_list = experiment["subjects_id"]
logging.debug("Subject ids: " + str(subject_list))

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]

#In[]:read components image from file

TR = experiment["t_r"]
n_regions= experiment["#regions"]
session_list = [1] # sessions start in 1 TODO allow more than one session
subject_list = experiment["subjects_id"]
logging.debug("subjects_id"+str(subject_list))

# set up ts file path and name from working_dir
ts_file = experiment["files_path"]["ts_image"]
logging.info("Timeseries files path: "+ts_file)

#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))

# Extract resting-state networks with CanICA

# ---------------------------------
if(args.canica):
    from nilearn.decomposition import CanICA


    canica = CanICA(n_components=args.n_components, smoothing_fwhm=args.fwhm,
                   memory="nilearn_cache", memory_level=2,
                   threshold=3., verbose=args.verbose, random_state=0
                   ,n_jobs=args.n_jobs)
    canica.fit(func_filenames)

    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    # components_img is a Nifti Image object, and can be saved to a file with
    # the following line:
    components_img.to_filename(op.join(data_dir,'canica_resting_state_all.nii.gz'))

    # -----------------------------------------------------------------
    from nilearn.plotting import plot_prob_atlas

    # Plot all ICA components together
    plot_prob_atlas(components_img, title='All CanICA components')

# ------------------------------------------------------------------
if(args.dictlearn):
    # Extract resting-state networks with DictionaryLearning

    # Import dictionary learning algorithm from decomposition module and call the
    # object and fit the model to the functional datasets
    from nilearn.decomposition import DictLearning

    # Initialize DictLearning object
    dict_learn = DictLearning(n_components=args.n_components,verbose=args.verbose,smoothing_fwhm=args.fwhm,
                              standardize=False,t_r=TR,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0, n_jobs=args.n_jobs)
    # Fit to the data
    dict_learn.fit(func_filenames)
    # Resting state networks/maps
    components_img_dic = dict_learn.masker_.inverse_transform(dict_learn.components_)
    components_img_dic.to_filename(op.join(data_dir,'dic_learn_resting_state_all.nii.gz'))

    # Visualization of resting state networks
    # Show networks using plotting utilities
    from nilearn import plotting

    plotting.plot_prob_atlas(components_img_dic, view_type='filled_contours',
                             title='Dictionary Learning maps')