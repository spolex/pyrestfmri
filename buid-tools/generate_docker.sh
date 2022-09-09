#!/bin/sh

set -e
 # --run "pip3 install -U nipype matplotlib nibabel nitime numpy scikit-learn pandas nilearn notebook" \
 #  --user elekin \
#  --workdir='/home/elekin'\.

# Generate Dockerfile or Singularity recipe.
docker run --rm repronim/neurodocker:latest generate docker \
  --label maintainer="Inigo Sanchez <jisanchez003@ehu.es>" \
  --base=jupyter/tensorflow-notebook\
  --pkg-manager=apt \
  --install gcc vim emacs-nox python3-pip python3-dev libssl-dev libffi-dev libxml2-dev libxslt1-dev\
  --fsl version=6.0.1 \
  --ants version=2.3.1 \
  --workdir='/home/jovyan/pyrestfmri'\
  --run "pip3 install -U nipype nibabel nitime nilearn" \
  --volume /home/jovyan/datos \
  --volume /home/jovyan/pyrestfmri \
  --volume /home/jovyan/results \
  -o Dockerfile

