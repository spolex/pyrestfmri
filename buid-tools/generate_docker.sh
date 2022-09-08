#!/bin/sh

set -e

# Generate Dockerfile or Singularity recipe.
docker run --rm repronim/neurodocker:latest generate docker \
  --label maintainer="Inigo Sanchez <jisanchez003@ehu.es>" \
  --base=neurodebian:stretch-non-free \
  --pkg-manager=apt \
  --install gcc fsl vim emacs-nox python3-pip python3-setuptools python3-dev\
  --fsl version=6.0.1 \
  --ants version=2.3.1 \
  --user elekin \
  --workdir='/home/elekin'\
  --run "pip3 install -U nipype matplotlib nibabel nitime numpy scikit-learn pandas nilearn" \
  --volume /home/elekin/datos \
  --volume /home/elekin/pyrestfmri \
  --volume /home/elekin/results: \
  -o Dockerfile

