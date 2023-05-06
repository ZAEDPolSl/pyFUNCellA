import numpy as np


def freeman_tukey(data):
    if np.any(data < 0):
        raise ValueError(
            "Some of the analyzed samples are smaller than 0 - check your preprocessing or raw data."
        )
    res = np.sqrt(data) + np.sqrt(data + 1)
    return res
