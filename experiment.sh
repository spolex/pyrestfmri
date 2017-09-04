#!/bin/sh
echo "Starting rest fmri analysis::::::::::::::::"
echo "Starting rest fmri preprocess::::::::::::::::"
preproc = python preprocess.py parallelism 2 config conf/config_test.json
if $preproc = 0;then

    echo "fmri preprocessed:::::::::::::::::::::"
    echo "Building functional brain atlas::::::::::::::::"
    brain_atlas = python brain_atlas_extractor.py config conf/config_test.json

    if brain_atlas = 0; then

        echo "Brain Atlas builded:::::::::::::::::::"
        echo "Regions extracting from brain atlas::::::::::::::::"
        extract_regions = region_extractor.py config conf/config_test.json

        if extract_regions = 0; then

            echo "Region extracted::::::::::::::::"
            echo "Calculate Shannon Spectral Entropy & Permutation Entropy::::::::::::::::"
            h = entropy_analysis.py config conf/config_test.json

            if test; then
                echo "Finishing entropy analysis::::::::::::::::"
                echo "Finishing rest fmri analysis::::::::::::::::"
                exit $ERRCODE
            else
                echo "Entropy analysis fails, see log file to know more"
                exit $ERRCODE
            fi

        else
            echo "Region extractor fails, see log file to know more"
            exit $ERRCODE
        fi
    else
        echo "Brain atlas generator fails, see log file to know more"
        exit $ERRCODE
    fi
else
    echo "Preprocessing pipeline fails, see log file to know more"
    exit $ERRCODE
fi