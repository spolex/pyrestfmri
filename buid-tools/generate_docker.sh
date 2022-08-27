#!/bin/sh

set -e

# Generate Dockerfile or Singularity recipe.
docker run --rm repronim/neurodocker:0.7.0 generate docker \
  --label maintainer="Inigo Sanchez <jisanchez003@ehu.es>" \
  --base=neurodebian:stretch-non-free \
  --pkg-manager=apt \
  --install fsl vim emacs-nox python3-pip \
  --fsl version=6.0.1 \
  --ants version=2.3.1 \
  --user elekin \
  --workdir='/home/elekin'\
  --run "pip3 install nitime nipype matplotlib nibabel nitime numpy dcm2niix scikit-learn pygpu pandas nilearn" \
  --volume /home/elekin/datos \
  --volume /home/elekin/pyrestfmri \
  --volume /home/elekin/results: \
  -o Dockerfile

