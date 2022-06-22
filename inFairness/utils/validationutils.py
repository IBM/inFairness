import numpy as np


def is_tensor_binary(data: np.ndarray):
    """Checks if the data is binary (0/1) or not. Return True if it is binary data

    Parameters
    --------------
        data: np.ndarray
            Data to validata if binary or not

    Returns
    ----------
        is_binary: bool
            True if data is binary. False if not
    """

    nonbindata = (data != 0) & (data != 1)
    has_nonbin_data = True in nonbindata
    return not has_nonbin_data
