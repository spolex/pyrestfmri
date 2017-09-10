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
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Functional atlas based region extractor")
parser.add_argument("-c","--config", type=str, help="Configuration file path", nargs='?', default="conf/config.json")
parser.add_argument("-b","--plot_connectcome", action="store_true", help="Plot brain connectcome")
parser.add_argument("-o","--plot_components", action="store_true", help="Plot region extracted for all components")
parser.add_argument("-r","--plot_regions", action="store_true", help="""Plot (right side) same network after region extraction to show "
                                                    that connected regions are nicely seperated""")
parser.add_argument("-i","--highpass", type=float, help="high pass filter, default none", nargs='?', default=None)
parser.add_argument("-l","--lowpass", type=float, help="low pass filter, default none", nargs='?', default=None)
parser.add_argument("-v","--verbose", type=int, help="verbose leevel, default 10", nargs='?', default=10)
parser.add_argument("-d", "--dict", action="store_true", help="Use dictlearning builded components image")
parser.add_argument("-m", "--msdl", action="store_true", help="Use the MSDL atlas of functional regions in rest.")
parser.add_argument("-e", "--standarize", action="store_true", help="If standardize is True, the time-series are centered and normed: their mean is put to 0 and their variance to 1 in the time dimension.")
parser.add_argument("-rs","--region_size", type=int, help="The parameter min_region_size=1350 mm^3 is to keep the minimum number of extracted regions,Default max: 1350", nargs='?', default=1350)


args=parser.parse_args()

# get experiment configuration
config = experiment_config(args.config)
experiment = config["experiment"]


logging.getLogger().setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["r_extractor"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")

# set up files' path
subject_list = experiment["subjects_id"]
logging.debug("Subject ids: " + str(subject_list))

# set up working dir
data_dir = experiment["files_path"]["preproc_data_dir"]


#In[]:read components image from file

TR = experiment["t_r"]
session_list = [1] # sessions start in 1 TODO allow more than one session
subject_list = experiment["subjects_id"]
logging.debug("Loading subjects: "+str(subject_list))

# set up ts file path and name from working_dir
ts_image = experiment["files_path"]["ts_image"]
logging.info("Timeseries image path: "+ts_image)
cf_file = experiment["files_path"]["preproc"]["noise_components"]
algorithm = 'dict' if args.dict else 'canica'
split =experiment["split"]
main_dir = data_dir+'/'+split+'/'+ algorithm
create_dir(main_dir)
mdsl_dir = data_dir+'/'+split+'/'+'mdsl'
create_dir(mdsl_dir)

#set up data dirs
subjects_pref = map(lambda subject: '_subject_id_'+(subject), subject_list)
sessions_subjects_dir = map(lambda session: map(lambda subject_pref: '_session_id_'+str(session)+subject_pref,subjects_pref), session_list)
#flattened all filenames
input_dirs = map(lambda session: map(lambda subj:op.join(data_dir,subj),session),sessions_subjects_dir)

#functional images and components confounds
func_filenames = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,ts_image),session),sessions_subjects_dir))
confounds_components = list(flatmap(lambda session: map(lambda subj:op.join(data_dir,subj,cf_file),session),sessions_subjects_dir))

components_filename = experiment["files_path"]["brain_atlas"]["components_img_1"] if args.dict else experiment["files_path"]["brain_atlas"]["components_img"]
components_img = image.load_img(os.path.join(data_dir, components_filename))

# In[]
#Region extracted
extractor = RegionExtractor(components_img, verbose=args.verbose, thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions', memory="nilearn_cache", memory_level=2,
                            t_r=TR, high_pass=args.highpass, low_pass=args.lowpass, standardize=args.standarize,
                            min_region_size=args.region_size)
extractor.fit()

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]
# Each region index is stored in index_
regions_index = extractor.index_

experiment["#regions"] = n_regions_extracted
config["experiment"] = experiment
num_comp = experiment["#components"]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components. %s'
         % (n_regions_extracted, num_comp, algorithm))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title,
                         output_file=op.join(main_dir,"_func_map_ica_"+str(num_comp)+".png")
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
  # call transform from RegionExtractor object to extract timeseries signals
  timeseries_each_subject = extractor.transform(filename, confounds=confound)
  np.savetxt(filename+"_extracted_ts.csv",timeseries_each_subject, delimiter=",")
  fig = plt.figure()
  plt.plot(timeseries_each_subject)
  plt.xlabel('')
  plt.ylabel('')
  fig.savefig(filename+'_'+algorithm+"_extracted_ts" + ".png")
  plt.close()

  # call fit_transform from ConnectivityMeasure object
  correlation = connectome_measure.fit_transform([timeseries_each_subject])
  # saving each subject correlation to correlations
  correlations.append(correlation)

# Mean of all correlations
mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)
experiment["files_path"]["ts_file"]=algorithm+"_extracted_ts.csv"
# Plot resulting correlation matrix
np.fill_diagonal(mean_correlations, 0)
title = 'Correlation interactions between %d regions, %s' % (n_regions_extracted,algorithm)
fig = plt.figure(figsize=(10,10))
plt.imshow(mean_correlations, interpolation="nearest",
           vmax=0.9, vmin=-0.9, cmap="RdBu_r")
plt.colorbar()
plt.title(title)
fig.savefig(os.path.join(main_dir,"_mean_correlation_matrix.png"))

# In[]

# Plot resulting connectcomes
if(args.plot_connectcome):
    plot_connectcome(regions_extracted_img, mean_correlations, title, op.join(main_dir,'_plot_connectcome_all_mean.png'))
# In[]:
# Plot region extracted for all components

if(args.plot_components):    
    plot(components_img,num_comp,os.path.join(main_dir,"_resting_state_ica_"))
    # Now, we plot (right side) same network after region extraction to show that connected regions are nicely seperated.
    plot_extracted(components_img,regions_extracted_img,num_comp,regions_index,os.path.join(main_dir,"_resting_state_ica_"))

# In[]
# 
if args.msdl:
  
  from nilearn import datasets
  from nilearn.input_data import NiftiMapsMasker
  
  atlas = datasets.fetch_atlas_msdl()
  atlas_filename = atlas.maps

  # Loading atlas image stored in 'maps'
  atlas_filename = atlas['maps']
  # Loading atlas data stored in 'labels'
  labels = atlas['labels']
  masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=args.standarize,
                         memory='nilearn_cache', verbose=args.verbose)
  masker.fit()
  masker_extracted_img = masker.maps_img_
  # Total number of regions extracted
  masker_n_regions_extracted = masker_extracted_img.shape[-1]

  # Visualization of region extraction results
  
  title = ('%d regions are extracted from mdls atlas'
           % (masker_n_regions_extracted))
  plotting.plot_prob_atlas(masker.maps_img, view_type='filled_contours',
                           title=title,
                           output_file=op.join(mdsl_dir,"_func_map_ica_"+str(masker_n_regions_extracted)+".png")
                           )
  
  masker_correlations=[]
  for filename,confound in zip(func_filenames, confounds_components):
    masker_timeseries_each_subject = masker.transform(filename,confounds=confound)
    filename = '/'.join(filename.split('/')[0:-1])+'/msdl'
    create_dir(filename)
    np.savetxt(filename+"/masker_extracted_ts.csv",masker_timeseries_each_subject, delimiter=",")
    fig = plt.figure()
    plt.plot(masker_timeseries_each_subject)
    plt.xlabel('')
    plt.ylabel('')
    fig.savefig(filename+"/masker_extracted_ts" + ".png")
    plt.close()
    # call fit_transform from ConnectivityMeasure object
    masker_correlation = connectome_measure.fit_transform([masker_timeseries_each_subject])
    # saving each subject correlation to correlations
    masker_correlations.append(masker_correlation)
  #
  ## Mean of all correlations
  #masker_mean_correlations = np.mean(masker_correlations, axis=0).reshape(n_regions_extracted,
  #                                                          n_regions_extracted)
  # In[]  
  # Mean of all correlations
  mask_mean_correlations = np.mean(masker_correlations, axis=0).reshape(len(labels),
                                                            len(labels))
  experiment["files_path"]["mdsl_ts_file"]="mdsl/masker_extracted_ts.csv"

# In[]
  # Plot resulting correlation matrix
  #title = 'Correlation interactions between %d regions, %s' % (n_regions_extracted,algorithm)
  fig = plt.figure(figsize=(10,10))
  # Mask out the major diagonal
  np.fill_diagonal(mask_mean_correlations, 0)
  plt.imshow(mask_mean_correlations, interpolation="nearest",
             vmax=0.8, vmin=-0.8, cmap="RdBu_r")
  plt.colorbar()
  # And display the labels
  x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
  y_ticks = plt.yticks(range(len(labels)), labels)
  #plt.title(title)
  fig.savefig(os.path.join(mdsl_dir,"mask_mean_correlation_matrix.png"))

  if(args.plot_components):      
    plot(atlas_filename,filepath=op.join(mdsl_dir,"_resting_state_ica_"),num_components=len(labels))
  # Plot resulting connectcomes
  
  if(args.plot_connectcome):
      plot_connectcome(masker_extracted_img, mask_mean_correlations, "", op.join(mdsl_dir,'_plot_connectcome_all_mean.png'))
      # Now, we plot (right side) same network after region extraction to   that connected regions are nicely seperated.
      plot_extracted(masker_extracted_img, masker.labels_,os.path.join(mdsl_dir,"_resting_state_ica_"))
experiment["files_path"]["masker_ts_file"]=algorithm+"masker_extracted_ts.csv"
config["experiment"]=experiment
update_experiment(config, args.config)