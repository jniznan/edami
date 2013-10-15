import numpy as np


def nan_rmse(A, B):
    '''
    Returns RMSE between two numpy arrays
    '''
    dat = (A - B) ** 2
    mdat = np.ma.masked_array(dat, np.isnan(dat))
    return np.sqrt(np.mean(mdat))
