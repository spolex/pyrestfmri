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
parser.add_argument("-c","--config", type=str, help="Configuration file path, default file is config_old.json", nargs='?', default="conf/config_test_old.json")
parser.add_argument("-m","--move_plot", action="store_true", help="MCFLIRT: Plot translation and rotations movement and save .par files")
parser.add_argument("-f","--fwhm", type=str, help="Smooth filter's threshold list. [5] by default", nargs='?', default="[5]")
parser.add_argument("-b","--brightness_threshold", type=float, help="Smooth filter brightness' threshold. 1000.0 by default", nargs='?', default=None)
parser.add_argument("-p","--parallelism", type=int, help="Multiproc parallelism configuration, default no parallelism", nargs='?', default=16)
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
from pprint import pprint
pprint(experiment)
home = experiment["files_path"]["working_dir"]
data_dir = experiment["files_path"]["preproc"]["data_path"]
output_dir = experiment["files_path"]["preproc"]["output"]

# Remove signal of non interest
tsnr = Node(confounds.TSNR(regress_poly=2), name='tsnr')
getthresh = Node(interface=ImageStats(op_string='-p 98'), name='getthreshold')
threshold_stddev = Node(Threshold(), name='threshold')
compcor = Node(confounds.ACompCor(components_file="noise_components.txt"), pre_filter=False, name='compcor')
remove_noise = Node(FilterRegressor(filter_all=True), name='remove_noise')


# delete header for noise_components.txt
def remove_header(in_file):
    from os.path import abspath as opa
    with open(in_file, 'r') as fin:
        data = fin.read().splitlines(True)
    out_file = 'noise_components_no_header.txt'
    with open(out_file, 'w') as fout:
        fout.writelines(data[1:])
    return opa(out_file)


remove_file_header = Node(Function(input_names=["in_file"], output_names=["out_file"], function=remove_header),
                          name='header_removal')

# Band pass filter
bandpass_filter = Node(TemporalFilter(highpass_sigma=args.highpass, lowpass_sigma=args.lowpass), name='bandpass_filter')

# Detrend
detrend = Node(Detrend(), name="detrend", output_type='NIFTI_GZ')
detrend.inputs.args = '-polort 2'

# Smooth - image smoothing
# smoothe filters threshold
fwhm = eval(args.fwhm) or [8]
#treshold
brightness_threshold = args.brightness_threshold or 1000.0
smooth = Node(SUSAN(), name="smooth")
smooth.iterables = ("fwhm", fwhm)
smooth.inputs.brightness_threshold = brightness_threshold

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
    plotter.inputs.in_source='fsl'
    plotter.iterables = ('plot_type',mc_plots)

# apply the combined transform
applyTransFunc = Node(ApplyTransforms(), iterfield=['input_image', 'transforms'],name='applyTransFunc')
applyTransFunc.inputs.input_image_type = 3
applyTransFunc.inputs.interpolation = 'BSpline'
applyTransFunc.inputs.invert_transform_flags = [False, False]
#applyTransFunc.inputs.terminal_output = 'file'

############################# PIPELINE #################################################################################
# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = output_dir

preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')

preproc.connect(selectfiles, 'anat', bet, 'in_file')

preproc.connect(selectfiles, 'func', trim, 'in_file')
preproc.connect(trim, 'out_file', slice_timing_correction, 'in_file')
preproc.connect(slice_timing_correction, 'slice_time_corrected_file', mcflirt, 'in_file')
preproc.connect(bet, 'out_file', datasink, 'preproc.@skull_strip')
preproc.connect(mcflirt, 'par_file', datasink, 'preproc.@par')
if args.move_plot:
    preproc.connect(plotter, 'out_file', datasink, 'preproc.@motion_plots')
# get transform of functional image to template and apply it to the functional
#images to template_3mm (same space as
# template)
preproc.connect(infosource, 'Template_3mm', applyTransFunc, 'reference_image')
preproc.connect(mcflirt,'out_file', applyTransFunc, 'input_image')
if args.move_plot:
    preproc.connect(plotter, 'out_file', datasink, 'preproc.@motion_plots')


# Create preproc output graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=opj(preproc.base_dir, 'preproc', 'graph.png'))

# Visualize the detailed graph
preproc.write_graph(graph2use='flat', format='png', simple_form=True)
Image(filename=opj(preproc.base_dir, 'preproc', 'graph_detailed.png'))

preproc.run('MultiProc', plugin_args={'n_procs': args.parallelism})
