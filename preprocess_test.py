from nipype.interfaces import fsl
from os.path import join as opj
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces import fsl
from utils import experiment_config
import argparse
from nipype import config, logging


# set up argparser
parser = argparse.ArgumentParser(description="Rest fmri preprocess pipeline")
parser.add_argument("-c","--config", type=str, help="Configuration file path, default file is config_old.json", nargs='?', default="conf/config_test.json")
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

############################# PIPELINE #################################################################################
# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = output_dir

preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')
preproc.connect(selectfiles, 'anat', bet, 'in_file')
preproc.connect(bet, 'out_file', datasink, 'preproc.@skull_strip')

# Create preproc output graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=opj(preproc.base_dir, 'preproc', 'graph.png'))

# Visualize the detailed graph
preproc.write_graph(graph2use='flat', format='png', simple_form=True)
Image(filename=opj(preproc.base_dir, 'preproc', 'graph_detailed.png'))

preproc.run('MultiProc', plugin_args={'n_procs': 8})
