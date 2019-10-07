from nipype.interfaces.fsl import BET

skullstrip = BET()
skullstrip.inputs.in_file = "/home/elekin/datos/C024/mprage.nii.gz"
skullstrip.inputs.out_file = "home/elekin/results/C024/skull_mprage.nii.gz"
res = skullstrip.run()