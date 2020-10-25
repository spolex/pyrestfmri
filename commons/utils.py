from logger import logging
import json
from itertools import chain
from os import path as op
from os import makedirs as om
import numpy as np


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


def create_dir(directory):
    if not op.exists(directory):
        om(directory)
    return directory


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
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def experiment_config(filename="conf/config_old.json"):
    """
    Load experiment configuration file
    :param filename:
    :return: dict with configuration
    """
    try:
        with open(filename) as config_file:
            return json.load(config_file)
    except IOError:
        logging.error(filename + " doesn't exist, an existing file is needed like config file")
        return -1


def update_experiment(config, filename):
    """

    :param config:
    :param experiment: new experiment configuration
    :param filename:
    :return:
    """
    with open(filename, 'w') as config_file:
        json.dump(config, config_file, indent=2)
        return 0


def get_dirs(config):
    """

    Parameters
    ----------
    config: json object contains experiment configuration

    Returns
    -------
    base_dir, data_dir, experiment_dir, output_dir, subject_list

    """
    experiment_dir = config["files_path"]["root"]
    base_dir = config["files_path"]["working_dir"]
    data_dir = config["files_path"]["preproc"]["data_path"]
    output_dir = config["files_path"]["preproc"]["output"]
    subject_list = config["subjects_id"]
    return base_dir, data_dir, experiment_dir, output_dir, subject_list
