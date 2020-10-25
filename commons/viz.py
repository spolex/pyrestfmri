#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 29 23:07:46 2017

@author: spolex
"""

import logging
from nilearn import plotting, image
import numpy as np

def plot(components_img, num_components, filepath):
    for index in range(0, num_components):
        img = image.index_img(components_img, index)
        coords = plotting.find_xyz_cut_coords(img)
        plotting.plot_stat_map(image.index_img(components_img, index),
                               cut_coords=coords, title='ICA ' + str(index + 1),
                               output_file=filepath + str(index + 1) + '.png')


def plot_connectcome(imagen, correlation_matrix, title, filepath):
    regions_imgs = image.iter_img(imagen)
    coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
    logging.debug("Connectcome coords ar: ")
    logging.debug(coords_connectome)
    plotting.plot_connectome(correlation_matrix, coords_connectome,
                             edge_threshold='80%', title=title, output_file=filepath)


def plot_extracted(components_img, regions_extracted_img, num_comp, regions_index, filepath):
    for index in range(0, num_comp):
        regions_indices_of_map3 = np.where(np.array(regions_index) == index)
        img = image.index_img(components_img, index)
        coords = plotting.find_xyz_cut_coords(img)
        display = plotting.plot_anat(title='Regions from component ' + str(index + 1), cut_coords=coords)
        colors = 'rgbcmyk'
        for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
            regions_image = image.index_img(regions_extracted_img, each_index_of_map3)
            display.add_overlay(
                regions_image,
                cmap=plotting.cm.alpha_cmap(color))
        display.savefig(filepath + str(each_index_of_map3) + '_regions.png')
        display.close()