#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 22:58:03 2017

@author: spolex

Herramienta para extraer los mapas funcionales basada en el algoritmo Dic Learn y CanICA
//TODO Analizar ambos algoritmos a fin de seleccionar uno de los dos 
 
"""

from nilearn import (image)
import os.path as op
from nilearn.datasets import load_mni152_template,load_mni152_brain_mask
template = load_mni152_template()
mask=load_mni152_brain_mask()

data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'

func_filenames = [op.join(data_dir, '_session_id_1_subject_id_T003', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T004', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T006', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
 #                 op.join(data_dir, '_session_id_1_subject_id_T0012', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T013', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T014', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T015', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T017', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T019', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T021', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T023', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T024', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T025', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T026', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T027', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T028', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T029', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T030', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T031', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T032', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T033', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz')
                 ]

# ---------------------------------
from nilearn.decomposition import CanICA

canica = CanICA(n_components=20, smoothing_fwhm=None,
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0
                ,n_jobs=-2)
canica.fit(func_filenames)

# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename(op.join(data_dir,'canica_resting_state_all.nii.gz'))

# -----------------------------------------------------------------
from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(image.resample_img(components_img, target_affine=template.affine), title='All ICA components')

# ------------------------------------------------------------------
# Extract resting-state networks with DictionaryLearning

# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=20,
                          memory="nilearn_cache", memory_level=2,
                          random_state=0, n_jobs=-2)
# Fit to the data
dict_learn.fit(func_filenames)
# Resting state networks/maps
components_img_dic = dict_learn.masker_.inverse_transform(dict_learn.components_)
components_img_dic.to_filename(op.join(data_dir,'dic_learn_resting_state_all.nii.gz'))

# Visualization of resting state networks
# Show networks using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(image.resample_img(components_img_dic, target_affine=template.affine), view_type='filled_contours',
                         title='Dictionary Learning maps')