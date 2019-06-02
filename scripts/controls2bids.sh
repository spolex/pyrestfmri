#environment variables
INPUT_PATH=$HOME/datos-niix
OUTPUT_PATH=$HOME/datos-niix
SUBJS="$(ls "$INPUT_PATH" | grep C0* | tr '\n' ' ')"
#SUBJS=( C021 C024 )
#CONVERSOR=none

#!/bin/bash
# declare an array called array and define 3 vales
array=( one two three )
for i in "${SUBJS[@]}"
do
	echo $i
	mkdir -p $OUTPUT_PATH/$i/sub-${i}/ses-1/func
	mkdir -p $OUTPUT_PATH/$i/sub-${i}/ses-1/anat
	mv $INPUT_PATH/FMRI/$i/FMRI.nii.gz $OUTPUT_PATH/$i/sub-${i}/ses-1/func/sub-${i}_ses-1_task-rest_bold.nii.gz
    mv $INPUT_PATH/FMRI/$i/FMRI.json $OUTPUT_PATH/$i/sub-${i}/ses-1/func/sub-${i}_ses-1_task-rest_bold.json
	mv $INPUT_PATH/CoG/$i/3D.nii.gz.gz $OUTPUT_PATH/$i/sub-${i}/ses-1/anat/sub-${i}_ses-1_acq-highres_t1w.nii.gz
    mv $INPUT_PATH/CoG/$i/3D.json $OUTPUT_PATH/$i/sub-${i}/ses-1/anat/sub-${i}_ses-1_acq-highres_t1w.json
done

