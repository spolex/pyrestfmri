#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 25 22:58:03 2017

@author: spolex

Herramienta para extraer los mapas funcionales basada 
en el algoritmo Dic Learn y CanICA.
//TODO Analizar ambos algoritmos a fin de seleccionar uno de los dos.
 
"""

# In[]:
from utils import flatmap
import os.path as op
from nilearn.datasets import load_mni152_template,load_mni152_brain_mask
template = load_mni152_template()
mask=load_mni152_brain_mask()

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
#ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz' 
ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz' 


# set up working dirs
data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'

#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))

# In[]:

# ---------------------------------
#from nilearn.decomposition import CanICA
#
#canica = CanICA(n_components=20, smoothing_fwhm=None,
#                memory="nilearn_cache", memory_level=2,
#                threshold=3., verbose=10, random_state=0
#                ,n_jobs=-2)
#canica.fit(func_filenames)
#
## Retrieve the independent components in brain space
#components_img = canica.masker_.inverse_transform(canica.components_)
## components_img is a Nifti Image object, and can be saved to a file with
## the following line:
#components_img.to_filename(op.join(data_dir,'canica_resting_state_all.nii.gz'))

# -----------------------------------------------------------------
#from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
#plot_prob_atlas(image.resample_img(components_img, target_affine=template.affine), title='All ICA components')

# ------------------------------------------------------------------
# Extract resting-state networks with DictionaryLearning

# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=20,verbose=10,smoothing_fwhm=None,
                          standardize=False,t_r=TR,
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
plotting.plot_prob_atlas(components_img_dic, view_type='filled_contours',
                         title='Dictionary Learning maps')