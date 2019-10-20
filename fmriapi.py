from utils import create_dir
from matplotlib import pyplot as plt
import numpy as np
from os import path as op
import logging


def extract_cbl(func_file, confound_file, masker, rdo_dir):
    """

    :param func_file:
    :param confound_file:
    :param masker:
    :return:
    """
    logging.debug("Extracting cerebellum from subject: "+func_file)
    ts = masker.transform(func_file, confounds=confound_file)
    #rdo_dir = '/'.join(func_file.split('/')[0:-1]) + '/cbl'
    if not op.exists(rdo_dir):
        create_dir(rdo_dir)
    np.savetxt(rdo_dir + "/cbl_extracted_ts.csv", ts, delimiter=",")
    fig = plt.figure()
    plt.plot(ts)
    plt.xlabel('')
    plt.ylabel('')
    fig.savefig(rdo_dir + "/masker_extracted_ts" + ".png")
    plt.close()
    logging.debug("Time series extracted from cerebellum")
    # save cbl image
    cbl_filename = rdo_dir + '/cbl_extracted.nii.gz'
    cbl_img = masker.inverse_transform(ts)
    cbl_img.to_filename(op.join(cbl_filename))
    logging.debug("Saved extracted image of cerebellum")
    return ts
