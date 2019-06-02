#environment variables
INPUT_PATH=$HOME/datos-dicom
OUTPUT_PATH=$HOME/datos-niix
#SUBJS="$(ls "$INPUT_PATH" | grep C0* | tr '\n' ' ')"
SUBJS=( C021 C024 )
#CONVERSOR=none

#!/bin/bash
# declare an array called array and define 3 vales
array=( one two three )
for i in "${SUBJS[@]}"
do
	echo $i
done

