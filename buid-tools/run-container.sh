#!/bin/sh

WORKSPACE=$HOME/workspace/pyrestfmri/
INPUTDIR=/data/elekin/data/origin/fmri/
OUTPUTDIR=/data/elekin/data/results/preproc-test/

#training docker with GPU
docker run --user $(id -u) --rm --name pyrestfmri \
  --volume "$INPUTDIR":/home/jovyan/datos \
  --volume "$WORKSPACE":/home/jovyan/pyrestfmri \
  --volume "$OUTPUTDIR":/home/jovyan/results \
  -it pyrestfmri