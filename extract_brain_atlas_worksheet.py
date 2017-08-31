
# coding: utf-8

# In[]:

#import nilearn modules
from nilearn import (image, plotting, decomposition)
from nilearn.input_data import NiftiMapsMasker 
from nilearn.regions import RegionExtractor
from nilearn.connectome import ConnectivityMeasure

#import nitime modules
from nitime.analysis import SpectralAnalyzer
from nitime.timeseries import TimeSeries
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask

import os,logging
logging.getLogger().setLevel(logging.DEBUG)
import matplotlib.pyplot as plt
import numpy as np

# set up working dir
data_path = os.path.join(os.getcwd(), 'data')


# In[]:
# load used data
# load fmri data
fmri_img_path = os.path.join(data_path,"subject001/func_data.nii.gz") 
data = image.load_img(fmri_img_path)
data.shape
# load smri data //TODO improve bg image to visualize data
anat = load_mni152_template()
anat.shape
# load mask //TOFO BINARIZE MASK
mask = load_mni152_brain_mask()
# In[]:
# # Extracting functional brain atlas:
# 
# Before analyzing functional connectivity, we need to reduce the dimensionality of the problem. To do that, we estimate an atlas directly on our data.
# 
# In this post-process we will:
#     * Use Canonical ICA to create a brain atlas
#     * Generating ROIs from the atlas
#     * Building a connectome

#canica = decomposition.CanICA(
#                              verbose=5,
#                              t_r=1.94, #sample interval
#                              target_affine=anat.affine, # resample
#                              #mask=mask_path, # mask 
#                              high_pass=0.01,
#                              low_pass=0.08,
#                              detrend=True, 
#                              n_components=20,
#                              n_jobs=-2,
#                              mask_strategy='epi'
#                              )
canica = decomposition.CanICA(mask=mask,
                              smoothing_fwhm=5.,
                              n_components=20,
                              high_pass=0.01,
                              low_pass=0.1,
                              detrend=True, 
                              n_jobs=-2,memory="nilearn_cache", memory_level=2,
                              threshold=3., verbose=10, random_state=0,
                              t_r=1.94, target_affine=anat.affine
                              )


# In[]:
logging.debug(canica)

# In[]:
canica.fit(data)

# In[]:
# save as file
components_img = canica.masker_.inverse_transform(canica.components_)# explain what this has done
components_img.to_filename(os.path.join(data_path,"subject001/run_000.feat/func_reg_data_canica_1.nii.gz"))

# In[]:read components image from file
components_img = image.load_img(os.path.join(data_path,"subject001/run_000.feat/func_reg_data_canica_1.nii.gz"))

# In[]:
# We visualize the generated atlas

index = 2
img = image.index_img(components_img, index)

coords = plotting.find_xyz_cut_coords(img)
plotting.plot_stat_map(image.index_img(components_img, index), 
                       #bg_img=anat, 
                       cut_coords=coords, title='DMN',
                       output_file=os.path.join(data_path,"subject001/run_000.feat/func_reg_data_component_"+str(index+1)+"_1.png"))


# In[]:
# ## Generating ROIs from the atlas
masker = NiftiMapsMasker(components_img, verbose=10,
                         standardize=False, detrend=False,
                         t_r=1.94, memory="nilearn_cache", memory_level=2
                         #low_pass=0.1, high_pass=0.01
                         )

logging.debug(masker) # to see the initialization of the parameters.

# In[]:
    
# Extract time series from the atlas
time_series = masker.fit_transform(fmri_img_path)

# In[]:

# Save as csv
np.savetxt(os.path.join(data_path,"subject001/run_000.feat/timeseries_1.csv"), time_series.T, delimiter=',')

# In[]:

fig = plt.figure(1)    
plt.plot(time_series)
plt.title('Timeseries for single subject shown for 20 brain regions')
plt.xlabel('Number of regions')
plt.ylabel('Normalized signal')
#plt.show()
fig.savefig(os.path.join(data_path,"subject001/run_000.feat/filtered_reg_data_canic_ts_singlesub_20BR_ts_1.png"))

# In[]:

extractor = RegionExtractor(components_img, verbose=5, threshold=3.,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            memory="nilearn_cache", memory_level=2,
                            standardize=False, min_region_size=1350,
                            low_pass=0.1, high_pass=0.01, t_r=1.94)


# In[]:
# Just call fit() to process for regions extraction
extractor.fit()


# In[]:

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]
# Each region index is stored in index_
regions_index = extractor.index_


# In[]:

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, 20))
#//TODO resample image in order to do better visualization
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title, threshold=0.002,
                         #bg_img=image.mean_img(data), 
                         output_file=os.path.join(data_path,
                                                  "subject001/run_000.feat/filtered_reg_data_canic_extracted_regions_1.png"))


# In[]:

# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn

# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
# call transform from RegionExtractor object to extract timeseries signals
timeseries_subject = extractor.transform(data)
# call fit_transform from ConnectivityMeasure object
correlation = connectome_measure.fit_transform([timeseries_subject]).reshape(n_regions_extracted,
                                                          n_regions_extracted)
np.savetxt(os.path.join(data_path,"subject001/run_000.feat/timeseries_subject.csv"), timeseries_subject.T, delimiter=',')

# In[]:

fig1 = plt.figure(2)    
plt.plot(timeseries_subject[:20])
plt.title('Timeseries for single subject shown for 20 brain regions')
plt.xlabel('Number of regions')
plt.ylabel('Normalized signal')
plt.show()
fig1.savefig(os.path.join(data_path,"subject001/run_000.feat/filtered_reg_data_canic_ts_singlesub_20BR_regions_ts.png"))


# ### Plot resulting connectomes

# In[]:

regions_imgs = image.iter_img(regions_extracted_img)
coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
title = 'Correlation interactions between %d regions' % n_regions_extracted
fig2 = plt.figure(3)
plt.imshow(correlation, interpolation="nearest", cmap=plt.cm.bwr)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(correlation, coords_connectome,edge_threshold='95%', 
                         title=title,
                         output_file=os.path.join(data_path,
                                      "subject001/run_000.feat/filtered_reg_data_canic_ts_singlesub_20BR_connectcome_brain_ts.png"))
fig2.savefig(os.path.join(data_path,
                          "subject001/run_000.feat/filtered_reg_data_canic_ts_singlesub_20BR_cor_matrix_ts.png"))



# In[93]:

## Plot regions extracted for only one specific 
# First, we plot a network of index=4 without region extraction (left plot)
index=0
img = image.index_img(components_img, index)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False,
                                 title='Showing one specific network '+str(index), 
                                 output_file=os.path.join(data_path,
                                      "subject001/run_000.feat/filtered_reg_data_network_"+str(index)+".png"))


# In[]:

# For this, we take the indices of the all regions extracted related to original
# network given as index.
regions_indices_of_map3 = np.where(np.array(regions_index) == index)

display = plotting.plot_anat(cut_coords=coords,
                             title='Regions from this network '+str(index),
                             #output_file=os.path.join(data_path,
                             #         "subject001/run_000.feat/filtered_reg_data_regions_network"+str(index)+".png")
                             )
# Add as an overlay all the regions of index 4
colors = 'rgbcmyk'
for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

# In[]:
# Extarct ts for given network
# uncomment to read from ts file
ts = TimeSeries(np.loadtxt(os.path.join(data_path,"subject001/run_000.feat/timeseries_subject.csv"), delimiter=','), sampling_interval=1.94)
# ts = TimeSeries(timeseries_subject[:,regions_indices_of_map3][:,0,:].T, sampling_interval=1.94)
s_analyzer = SpectralAnalyzer(ts)

# In[]:
# Calculate PSD for given network
psd = s_analyzer.psd
fig = plt.figure(4)    
values = psd[1]
freqs = psd[0]
plt.plot(freqs,
         values[1])
plt.xlabel('frequency (hertz)')
plt.ylabel('Power Spectral Density')
fig.savefig(os.path.join(data_path,"subject001/run_000.feat/filtered_reg_data_canic_ts_singlesub_20BR_ts.png"))

# In[] 

#spectral shannon entropy implementation 
def ssH(density_func):
    if not density_func.any():
        return 0.
    entropy = 0.
    for density in density_func:
        if density > 0:
            entropy += density*np.log(1/density)
    return entropy

# In[]
# Density function
f_density = map(lambda value: value/values.sum(), values)
# plot density function between r1 and r2
plt.plot(freqs,
         f_density[118])
plt.xlabel('Frequency (hertz)')
plt.ylabel('Density Function')


# In[]
#calculate SSE
entropy = map(ssH, f_density)
print("Calculated Shannon entropy for each reagion:")
print(entropy)

# In[]
# Plot
fig = plt.figure(5)    
plt.plot(range(1,189,1),
         entropy)
plt.xlabel('Extracted regions')
plt.ylabel('Spectral Shannon Entropy')
fig.savefig(os.path.join(data_path,"subject001/run_000.feat/spectral_shannon_entropy.png"))

# In[]
# Save as text file
np.savetxt(os.path.join(data_path,"subject001/run_000.feat/spectral_shannon_entropy_1.csv"), entropy, delimiter=',')

# In[]

