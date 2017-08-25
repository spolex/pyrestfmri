# In[]:

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

func_filenames = [op.join(data_dir, '_session_id_1_subject_id_T003', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T004', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
                  op.join(data_dir, '_session_id_1_subject_id_T006', '_fwhm_4/smooth/detrend_regfilt_filt_smooth.nii.gz'),
 #                 op.join(data_dir, '_session_id_1_subject_id_T012', 'f1.nii.gz'),
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
confounds_components = [op.join(data_dir, '_session_id_1_subject_id_T003', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T004', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T006', 'compcor/noise_components.txt'),
       #                 op.join(data_dir, '_session_id_1_subject_id_T012', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T013', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T014', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T015', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T017', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T019', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T021', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T023', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T024', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T025', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T026', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T027', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T028', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T029', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T030', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T031', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T032', 'compcor/noise_components.txt'),
                        op.join(data_dir, '_session_id_1_subject_id_T033', 'compcor/noise_components.txt')
                        ] 

components_img = image.load_img(os.path.join(data_dir,"dic_learn_resting_state_all.nii.gz"))
hdr = components_img.header
shape = components_img.shape
num_comp = shape[3]
  
# In[]
#Region extracted


extractor = RegionExtractor(components_img, verbose=10, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            memory="nilearn_cache", memory_level=2,
                            standardize=True, min_region_size=1350,
                            t_r=1.94)
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
# Now, we plot (right side) same network after region extraction to show that connected regions are nicely seperated. Each brain extracted region is identified as separate color.
# For this, we take the indices of the all regions extracted related to original
# network given as 4.
for index in range(0,num_comp):
  regions_indices_of_map3 = np.where(np.array(regions_index) == index)
  display = plotting.plot_anat(cut_coords=coords,
                               title='Regions from network '+str(index+1))

  # Add as an overlay all the regions of each index
  colors = 'rgbcmyk'
  for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
      display.add_overlay(
                          image.index_img(regions_extracted_img, each_index_of_map3),
                          cmap=plotting.cm.alpha_cmap(color))
  
  plotting.show()