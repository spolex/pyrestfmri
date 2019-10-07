from nipype.interfaces import fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

skullstrip = fsl.BET()
skullstrip.inputs.in_file = "/home/elekin/datos/C024/mprage.nii.gz"
skullstrip.inputs.out_file = "home/elekin/results/C024/skull_mprage.nii.gz"
res = skullstrip.run()