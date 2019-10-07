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
from nipype import config, logging
from utils import experiment_config


# set up argparser
parser = argparse.ArgumentParser(description="Rest fmri preprocess pipeline")
parser.add_argument("-c","--config", type=str, help="Configuration file path, default file is config_old.json", nargs='?', default="conf/config_test.json")
parser.add_argument("-m","--move_plot", action="store_true", help="MCFLIRT: Plot translation and rotations movement and save .par files")
args = parser.parse_args()

#load experiment configuration
experiment = experiment_config(args.config)["experiment"]
# set up envvironment
config.enable_debug_mode()
config.set('execution', 'stop_on_first_crash', 'true')
config.set('execution', 'remove_unnecessary_outputs', 'true')
config.set('logging', 'workflow_level', experiment["log_level"])
config.set('logging', 'interface_level', experiment["log_level"])
config.set('logging', 'log_to_file', True)
config.set('logging', 'log_directory', experiment["preproc_log_dir"])
logging.update_logging(config)

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

subject_list = experiment["subjects_id"]
session_list = [1]

anat_file = opj('{subject_id}', 'mprage.nii.gz')
func_file = opj('{subject_id}', 'f1.nii.gz')

templates = {'anat': anat_file,
             'func': func_file}

home = experiment["files_path"]["working_dir"]
data_dir = experiment["files_path"]["preproc"]["data_path"]
output_dir = experiment["files_path"]["preproc"]["output"]

# Infosource - a function free node to iterate over the list of subject names
# fetch input
infosource = Node(IdentityInterface(fields=['subject_id', 'session_id', 'Template','Template_3mm']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]

selectfiles = Node(SelectFiles(templates, base_directory=data_dir), name="selectfiles")

# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory=home, container=output_dir), name="datasink")

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

# Brain extraction:
bet = Node(interface=fsl.BET(), name='skull_strip', iterfield=['in_file'])
bet.inputs.frac = 0.4
bet.inputs.robust = True

# Select number of volumes
trim = Node(interface=Trim(),output_type='NIFTI_GZ',name='select_volumes')
# remove first 3 volume
trim.inputs.begin_index = 3
# get only until volume 162
trim.inputs.end_index = 161

# Slice Timing correction
# time repetition
TR = experiment["t_r"]
slice_timing_correction = Node(SliceTimer(time_repetition=TR), name="slice_timer")

# MCFLIRT - motion correction
mcflirt = Node(MCFLIRT(mean_vol=True, save_plots=args.move_plot), name="mcflirt")

# Plot estimated motion parametersfrom realignment
if args.move_plot:
    # plots
    mc_plots = ['rotations', 'translations']
    plotter = Node(fsl.PlotMotionParams(), name="motion_correction_plots")
    plotter.inputs.in_source = 'fsl'
    plotter.iterables = ('plot_type', mc_plots)

############################# PIPELINE #################################################################################
# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = output_dir

preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')

preproc.connect(selectfiles, 'anat', bet, 'in_file')

preproc.connect(selectfiles, 'func', trim, 'in_file')
preproc.connect(trim, 'out_file', slice_timing_correction, 'in_file')
preproc.connect(slice_timing_correction, 'slice_time_corrected_file', mcflirt, 'in_file')
if args.move_plot:
    preproc.connect(mcflirt, 'par_file', plotter, 'in_file')


preproc.connect(bet, 'out_file', datasink, 'preproc.@skull_strip')
preproc.connect(mcflirt, 'par_file', datasink, 'preproc.@par')


# Create preproc output graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=opj(preproc.base_dir, 'preproc', 'graph.png'))

# Visualize the detailed graph
preproc.write_graph(graph2use='flat', format='png', simple_form=True)
Image(filename=opj(preproc.base_dir, 'preproc', 'graph_detailed.png'))

preproc.run('MultiProc', plugin_args={'n_procs': 16})
