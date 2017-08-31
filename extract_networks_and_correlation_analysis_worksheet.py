# In[]:
from utils import flatmap
#import nilearn modules
from nilearn.connectome import ConnectivityMeasure

from os import path as op
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
from nilearn import (image, plotting)
from nilearn.regions import RegionExtractor
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_template

# get mni template
template = load_mni152_template()

# set up working dir
data_path = op.join(os.getcwd(), 'data')


#In[]:read components image from file

data_dir = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM/datos/preproc'

# set up params with default values TODO get from nipype inputs (select subjects ids, )
TR = 1.94
n_regions= 48
session_list = [1] #sessions start in 1
#subject_list = ['T003', 'T004','T006','T013', 'T014', 'T015', 'T017', 'T018', 
#                'T019', 'T021', 'T023','T024','T025', 'T026', 'T027', 'T028', 
#                'T029', 'T030', 'T031', 'T032', 'T033', 'T035', 'T039', 'T040',
#                'T042', 'T043', 'T045', 'T046', 'T056', 'T058', 'T059', 'T060',
#                'T061', 'T062', 'T063', 'T064', 'T065', 'T066', 'T067', 'T068',
#                'T069', 'T070', 'T071', 'T072', 'T073', 'T074', 'T075', 'T076',
#                'T077', 'T078', 'T079', 'T080', 'T081', 'T082']
subject_list = ['T026']

#ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz' 
ts_file = '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz' 
cf_file = 'compcor/noise_components.txt'

#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)

#functional images and components confounds TODO get from nipypeinputs
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_file),session),sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session),sessions_subjects_dir))

#TODO nipype: get from regions_input_file
components_img = image.load_img(os.path.join(data_dir,"dic_learn_resting_state_all.nii.gz"))
hdr = components_img.header
shape = components_img.shape
num_comp = shape[3]
  
# In[]
#Region extracted 
extractor = RegionExtractor(components_img, verbose=10, thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions', memory="nilearn_cache", memory_level=2,
                            standardize=True, detrend = True, low_pass = 0.15, high_pass=0.02, t_r=1.94)
extractor.fit()

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]
# Each region index is stored in index_
regions_index = extractor.index_

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, num_comp))
#//TODO resample image in order to do better visualization
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title,
                         output_file=op.join(data_dir,"canica_func_map_comp_"+str(num_comp)+"_1.png")
#                         threshold=0.002
                         )

# In[]
# Compute correlation coefficients
# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn

correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename,confound in zip(func_filenames, confounds_components):
  print(filename)
  # call transform from RegionExtractor object to extract timeseries signals
  timeseries_each_subject = extractor.transform(filename, confounds=confound)
  np.savetxt(filename+"_extracted_ts.csv",timeseries_each_subject, delimiter=",")
  # call fit_transform from ConnectivityMeasure object
  correlation = connectome_measure.fit_transform([timeseries_each_subject])
  # saving each subject correlation to correlations
  correlations.append(correlation)

# Mean of all correlations
mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

# In[]
# Plot resulting connectcomes
regions_imgs = image.iter_img(regions_extracted_img)
coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
title = 'Correlation interactions between %d regions' % n_regions_extracted
fig = plt.figure(0)
plt.imshow(mean_correlations, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.bwr)
plt.colorbar()
plt.title(title)
fig.savefig(os.path.join(data_dir,"mean_correlation_matrix.png"))
plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title, output_file=op.join(data_dir,'plot_connectcome_all_mean.png'))

# In[]:
# Plot region extracted for all networks
for index in range(0,num_comp):
  img = image.index_img(components_img, index)
  coords = plotting.find_xyz_cut_coords(img)
  plotting.plot_stat_map(image.index_img(components_img, index), 
                         #bg_img=anat, 
                         cut_coords=coords, title='DMN Component '+str(index+1),
                         output_file=os.path.join(data_dir,"canica_resting_state_all_"+str(index+1)+"_1.png"))
  
# In[]
# Now, we plot (right side) same network after region extraction to show that connected regions are nicely seperated.
for index in range(0,num_comp):
  regions_indices_of_map3 = np.where(np.array(regions_index) == index)
  display = plotting.plot_anat(cut_coords=coords,
#                               output_file=os.path.join(data_dir,"canica_resting_state_nwk_all_"+str(index+1)+"_1.png"),
                               title='Regions from network '+str(index+1))

  # Add as an overlay all the regions of each index
  colors = 'rgbcmyk'
  for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
      display.add_overlay(
                          image.index_img(regions_extracted_img, each_index_of_map3),
                          cmap=plotting.cm.alpha_cmap(color))
  display.savefig(os.path.join(data_dir,"canica_resting_state_"+str(index+1)+"_regions_1.png"))
  display.close()
  
  plotting.show()
