#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: spolex
"""

from __future__ import print_function, division, unicode_literals, absolute_import
from os.path import join as opj
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MCFLIRT, BET, SUSAN, SliceTimer, TemporalFilter,ImageStats,Threshold,FilterRegressor
from nipype.interfaces.ants import Registration
from nipype.interfaces.nipy.preprocess import Trim
from nipype.interfaces.afni import Detrend
from nipype.algorithms import confounds
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Function,IdentityInterface, Merge
from nipype.interfaces.ants import ApplyTransforms
import argparse
from utils import experiment_config
import logging


# set up argparser
parser = argparse.ArgumentParser(description="Rest fmri preprocess pipeline")
parser.add_argument("-c","--config", type=str, help="Configuration file path, default file is config_old.json", nargs='?', default="conf/config_test.json")
parser.add_argument("-m","--move_plot", action="store_true", help="MCFLIRT: Plot translation and rotations movement and save .par files")
parser.add_argument("-f","--fwhm", type=str, help="Smooth filter's threshold list. [5] by default", nargs='?', default="[5]")
parser.add_argument("-b","--brightness_threshold", type=float, help="Smooth filter brightness' threshold. 1000.0 by default", nargs='?', default=None)
parser.add_argument("-p","--parallelism", type=int, help="Multiproc parallelism configuration, default no parallelism", nargs='?', default=2)
parser.add_argument("-i","--highpass", type=float, help="high pass filter, default 0.01", nargs='?', default=0.01)
parser.add_argument("-l","--lowpass", type=float, help="low pass filter, default 0.08", nargs='?', default=0.08)


args = parser.parse_args()

# load experiment configuration
experiment = experiment_config(args.config)["experiment"]
#logging
logging.getLogger("preproc").setLevel(experiment["log_level"])
logging.basicConfig(filename=experiment["files_path"]["preproc"]["log"], filemode ='w', format="%(asctime)s - %(levelname)s - %(message)s")

# set working dirs
experiment_dir = experiment["files_path"]["root"]
base_dir = experiment["files_path"]["working_dir"]
data_dir = experiment["files_path"]["preproc"]["data_path"]
output_dir = experiment["files_path"]["preproc"]["output"]

subject_list=experiment["subjects_id"]

# session id list
session_list=[1]#TODO allow more than one session

# smoothe filters threshold
fwhm = eval(args.fwhm) or [8]

#treshold
brightness_threshold = args.brightness_threshold or 1000.0

# time repetition
TR = experiment["t_r"]

# plots
mc_plots=['rotations','translations'] if args.move_plot else None

## Interfaces configuration
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# SelectFiles - to grab the data (alternativ to DataGrabber)
#anat_file = opj('{subject_id}', 'sub-{subject_id}','ses-1','anat','sub-{subject_id}_ses-1_acq-highres_t1w.nii.gz')
#func_file = opj('{subject_id}', 'sub-{subject_id}','ses-1','func','sub-{subject_id}_ses-1_task-rest_bold.nii.gz')
anat_file = opj('{subject_id}', 'mprage.nii.gz')
func_file = opj('{subject_id}', 'f1.nii.gz')

templates = {'anat': anat_file,
             'func': func_file}

selectfiles = Node(SelectFiles(templates,base_directory=data_dir),name="selectfiles")

# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory=experiment_dir,container=output_dir),name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', ''),
                 ('_task-flanker', ''),
                 ('_mcf.nii_mean_reg', '_mean'),
                 ('.nii.par', '.par')
                 ]
subjFolders = [('%s_%s/' % (sess, sub), '%s/%s' % (sub, sess))
               for sess in session_list
               for sub in subject_list]
subjFolders += [('%s_%s' % (sub, sess), '')
                for sess in session_list
                for sub in subject_list]
subjFolders += [('%s%s_' % (sess, sub), '')
                for sess in session_list
                for sub in subject_list]
substitutions.extend(subjFolders)

datasink.inputs.substitutions = substitutions

# Select number of volumes
trim = Node(interface=Trim(), output_type='NIFTI_GZ',name='select_volumes')
trim.inputs.begin_index=3 # remove first 3 volume
trim.inputs.end_index=161 # get only until volume 162

# Neck remover:
fov = Node(interface=fsl.RobustFOV(), name='Neck_remover', iterfield=['in_file'])

# Normalize to MNI:
flt = Node(fsl.FLIRT(bins=640, cost_func='mutualinfo'), name="Normalize", iterfield=['in_file'])

# Brain extraction, robust:
bet = Node(interface=BET(), name='skull_strip', iterfield=['in_file'])
bet.inputs.frac = 0.4
bet.inputs.robust = True

# Slice Timing correction
slice_timing_correction = Node(SliceTimer(time_repetition=TR), name="slice_timer")

# MCFLIRT - motion correction
mcflirt = Node(MCFLIRT(mean_vol=True, save_plots=args.move_plot), name="mcflirt")

# Plot estimated motion parametersfrom realignment
if args.move_plot:
    plotter = Node(fsl.PlotMotionParams(), name="motion_correction_plots")
    plotter.inputs.in_source='fsl'
    plotter.iterables = ('plot_type',mc_plots)

# coregistration step based on affine transformation using ANTs
coreg = Node(Registration(), name='CoregAnts')
coreg.inputs.output_transform_prefix = 'func2highres'
coreg.inputs.output_warped_image = 'func2highres.nii.gz'
coreg.inputs.output_transform_prefix = "func2highres_"
coreg.inputs.transforms = ['Rigid', 'Affine']
coreg.inputs.transform_parameters = [(0.1,), (0.1,)]
coreg.inputs.number_of_iterations = [[100, 100]]*3 
coreg.inputs.dimension = 3
coreg.inputs.write_composite_transform = True
coreg.inputs.collapse_output_transforms = False
coreg.inputs.metric = ['Mattes'] * 2 
coreg.inputs.metric_weight = [1] * 2 
coreg.inputs.radius_or_number_of_bins = [32] * 2 
coreg.inputs.sampling_strategy = ['Regular'] * 2 
coreg.inputs.sampling_percentage = [0.3] * 2 
coreg.inputs.convergence_threshold = [1.e-8] * 2 
coreg.inputs.convergence_window_size = [20] * 2
coreg.inputs.smoothing_sigmas = [[4, 2]] * 2 
coreg.inputs.sigma_units = ['vox'] * 4
coreg.inputs.shrink_factors = [[6, 4]] + [[3, 2]]
coreg.inputs.use_estimate_learning_rate_once = [True] * 2
coreg.inputs.use_histogram_matching = [False] * 2 
coreg.inputs.initial_moving_transform_com = True
coreg.inputs.verbose = True


# registration or normalization step based on symmetric diffeomorphic image registration (SyN) using ANTs 
reg = Node(Registration(), name='NormalizationAnts')
reg.inputs.output_transform_prefix = 'highres2template'
reg.inputs.output_warped_image = 'highres2template.nii.gz'
reg.inputs.output_transform_prefix = "highres2template_"
reg.inputs.initial_moving_transform_com = False
reg.inputs.transforms = ['Affine', 'SyN']
reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = True
reg.inputs.initial_moving_transform_com = True
reg.inputs.metric = ['Mattes'] * 2
reg.inputs.metric_weight = [1] * 2
reg.inputs.radius_or_number_of_bins = [32] * 2
reg.inputs.sampling_strategy = ['Random', None]
reg.inputs.sampling_percentage = [0.05, None]
reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
reg.inputs.convergence_window_size = [20] * 2 
reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
reg.inputs.sigma_units = ['vox'] * 2
reg.inputs.shrink_factors = [[2,1], [3,2,1]]
reg.inputs.use_estimate_learning_rate_once = [True] * 2
reg.inputs.use_histogram_matching = [True] * 2
reg.inputs.verbose = False

 # combine transforms
merge = Node(Merge(2), iterfield=['in2'], name='mergexfm')

# apply the combined transform 
applyTransFunc = Node(ApplyTransforms(), iterfield=['input_image', 'transforms'],name='applyTransFunc')
applyTransFunc.inputs.input_image_type = 3
applyTransFunc.inputs.interpolation = 'BSpline'
applyTransFunc.inputs.invert_transform_flags = [False, False]
#applyTransFunc.inputs.terminal_output = 'file'

# Remove signal of non interest
tsnr = Node(confounds.TSNR(regress_poly=2), name='tsnr')
getthresh = Node(interface=ImageStats(op_string='-p 98'),name='getthreshold')
threshold_stddev = Node(Threshold(), name='threshold')
compcor = Node(confounds.ACompCor(components_file="noise_components.txt"),pre_filter=False,name='compcor')
remove_noise = Node(FilterRegressor(filter_all=True),name='remove_noise')

#delete header for noise_components.txt
def remove_header(in_file):
  from os.path import abspath as opa
  with open(in_file, 'r') as fin:
    data = fin.read().splitlines(True)
  out_file =  'noise_components_no_header.txt'
  with open(out_file, 'w') as fout:
    fout.writelines(data[1:])
  return opa(out_file)
    
remove_file_header = Node(Function(input_names=["in_file"],output_names=["out_file"],function=remove_header), name='header_removal')

    
# Band pass filter
bandpass_filter = Node(TemporalFilter(highpass_sigma=args.highpass, lowpass_sigma=args.lowpass),name='bandpass_filter')

# Detrend
detrend = Node(Detrend(), name="detrend",output_type='NIFTI_GZ')
detrend.inputs.args = '-polort 2'

# Smooth - image smoothing
smooth = Node(SUSAN(), name="smooth")
smooth.iterables = ("fwhm", fwhm)
smooth.inputs.brightness_threshold = brightness_threshold

# Infosource - a function free node to iterate over the list of subject names
# fetch input
infosource = Node(IdentityInterface(fields=['subject_id', 'session_id', 'Brain_template','Template','Template_3mm']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]


############################# PIPELINE #################################################################################
# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = output_dir

# Connect all components of the preprocessing workflow
preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id' )
preproc.connect(selectfiles, 'func', trim, 'in_file')
preproc.connect(selectfiles, 'anat', fov, 'in_file')
preproc.connect(fov, 'out_roi', flt, 'in_file')
preproc.connect(infosource, 'Template', flt, 'reference')
preproc.connect(flt, 'out_file', bet, 'in_file')
preproc.connect(trim, 'out_file', slice_timing_correction, 'in_file')
preproc.connect(slice_timing_correction, 'slice_time_corrected_file', mcflirt, 'in_file')
if(args.move_plot):preproc.connect(mcflirt, 'par_file', plotter, 'in_file')
#func2highres
preproc.connect(bet, 'out_file', coreg, 'fixed_image')
preproc.connect(mcflirt, 'mean_img', coreg, 'moving_image')
#anat2standard
preproc.connect(bet, 'out_file', reg, 'moving_image')
preproc.connect(infosource, 'Brain_template', reg, 'fixed_image')
# get transform of functional image to template and apply it to the functional 
#images to template_3mm (same space as     
# template)
preproc.connect(infosource, 'Template_3mm', applyTransFunc, 'reference_image')
preproc.connect(mcflirt,'out_file', applyTransFunc, 'input_image')
preproc.connect(coreg, 'composite_transform', merge, 'in1')
preproc.connect(reg, 'composite_transform', merge, 'in2')
preproc.connect(merge, 'out', applyTransFunc, 'transforms')
#artifact detection
preproc.connect(applyTransFunc, 'output_image', tsnr, 'in_file')
preproc.connect(tsnr, 'stddev_file', threshold_stddev, 'in_file')

preproc.connect(tsnr, 'stddev_file', getthresh, 'in_file')
preproc.connect(getthresh, 'out_stat', threshold_stddev, 'thresh')
preproc.connect(applyTransFunc, 'output_image', compcor, 'realigned_file')
    
preproc.connect(threshold_stddev, 'out_file', compcor, 'mask_files')
preproc.connect(tsnr, 'detrended_file', remove_noise, 'in_file') 
preproc.connect(compcor, 'components_file', remove_file_header, 'in_file')
preproc.connect(remove_file_header, 'out_file', remove_noise, 'design_file')
#bandpass filter
preproc.connect(remove_noise, 'out_file',bandpass_filter,'in_file')
#smooth
preproc.connect(bandpass_filter, 'out_file', smooth, 'in_file')
#datasink
preproc.connect(mcflirt, 'par_file', datasink, 'preproc.@par')
preproc.connect(smooth, 'smoothed_file', datasink, 'preproc.@smooth')
preproc.connect(bandpass_filter, 'out_file', datasink, 'preproc.@bandpass_filter')
preproc.connect(bet, 'out_file', datasink, 'preproc.@skull_strip')
if(args.move_plot):preproc.connect(plotter, 'out_file', datasink, 'preproc.@motion_plots')

#set up templates to register
preproc.inputs.infosource.Template = opj(base_dir,experiment["files_path"]["preproc"]["register"]["template"])
preproc.inputs.infosource.Brain_template = opj(base_dir,experiment["files_path"]["preproc"]["register"]["brain_template"])
preproc.inputs.infosource.Template_3mm = opj(base_dir,experiment["files_path"]["preproc"]["register"]["template_3mm"])

# visualizamos el workfow

# Create preproc output graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=opj(preproc.base_dir, 'preproc', 'graph.png'))

# Visualize the detailed graph
preproc.write_graph(graph2use='flat', format='png', simple_form=True)
Image(filename=opj(preproc.base_dir, 'preproc', 'graph_detailed.png'))

#preproc.run()
preproc.run('MultiProc', plugin_args={'n_procs': args.parallelism})
