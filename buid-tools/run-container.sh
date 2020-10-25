#!/bin/sh

WORKSPACE=$HOME/workspaces/elekin/pyrestfmri/
INPUTDIR=/data/elekin/data/origin/fmri/
OUTPUTDIR=/data/elekin/data/results/preproc-test/

#training docker with GPU
docker run --user $(id -u) --rm --name pyrestfmri \
  --volume /home/elekin/datos:"$INPUTDIR" \
  --volume /home/elekin/pyrestfmri:"$WORKSPACE" \
  --volume /home/elekin/results:"$OUTPUTDIR" \
  -it pyrestfmri