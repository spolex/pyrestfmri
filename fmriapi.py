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
    logging.debug("Extractin cerebellum from subject: "+func_file)
    masker_timeseries_each_subject = masker.transform(func_file, confounds=confound_file)
    #rdo_dir = '/'.join(func_file.split('/')[0:-1]) + '/cbl'
    if not op.exists(rdo_dir):
        create_dir(rdo_dir)
    np.savetxt(rdo_dir + "/cbl_extracted_ts.csv", masker_timeseries_each_subject, delimiter=",")
    fig = plt.figure()
    plt.plot(masker_timeseries_each_subject)
    plt.xlabel('')
    plt.ylabel('')
    fig.savefig(rdo_dir + "/masker_extracted_ts" + ".png")
    plt.close()
    # save cbl image
    cbl_filename = rdo_dir + '/cbl_extracted.nii.gz'
    cbl_img = masker.inverse_transform(masker_timeseries_each_subject)
    cbl_img.to_filename(op.join(cbl_filename))