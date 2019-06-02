#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 29 23:07:46 2017

@author: spolex
"""
from os import path as op
from os import makedirs as om
from itertools import chain
import json
import logging
from nilearn import plotting,image

def experiment_config(filename="conf/config.json"):
    """
    Load experiment configuration file
    :param filename:
    :return: dict with configuration
    """
    try:
        with open(filename) as config_file:
            return json.load(config_file)
    except IOError:
        logging.error(filename+" doesn't exist, an existing file is needed like config file")
        return -1

def update_experiment(config, filename):
    """

    :param experiment: new experiment configuration
    :param filename:
    :return:
    """
    try:
        with open(filename,'w') as config_file:
            json.dump(config, config_file, indent=2)
            return 0
    except IOError:
        logging.error(filename+" doesn't exist, an existing file is needed like config file")
        return -1

def flatmap(f, items):
  return chain.from_iterable(map(f, items))

def create_dir(directory):
  if not op.exists(directory):
    om(directory)
  return directory

import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    References:
      [1]https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def plot(components_img,num_components,filepath):  
    for index in range(0,num_components):
      img = image.index_img(components_img, index)
      coords = plotting.find_xyz_cut_coords(img)
      plotting.plot_stat_map(image.index_img(components_img, index),
                             cut_coords=coords, title='ICA '+str(index+1),
                             output_file=filepath+str(index+1)+'.png')

def plot_connectcome(imagen, correlation_matrix, title, filepath):
    regions_imgs = image.iter_img(imagen)
    coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
    logging.debug("Connectcome coords ar: ")
    logging.debug(coords_connectome)
    plotting.plot_connectome(correlation_matrix, coords_connectome,
                         edge_threshold='80%', title=title, output_file=filepath)
    
    
def plot_extracted(components_img,regions_extracted_img,num_comp,regions_index,filepath):    
  for index in range(0,num_comp):
    regions_indices_of_map3 = np.where(np.array(regions_index) == index)
    img = image.index_img(components_img, index)
    coords = plotting.find_xyz_cut_coords(img)
    display = plotting.plot_anat(title='Regions from component '+str(index+1), cut_coords=coords)      
    colors = 'rgbcmyk'
    for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
      regions_image = image.index_img(regions_extracted_img, each_index_of_map3)
      display.add_overlay(
                          regions_image,
                          cmap=plotting.cm.alpha_cmap(color))
    display.savefig(filepath+str(each_index_of_map3)+'_regions.png')
    display.close()