# python/utils/preprocessing.py

import numpy as np


def normalize_images(x: np.ndarray):
    """
    Ensure images are in [0,1] float32.
    """
    x = x.astype("float32")
    if x.max() > 1.0:
        x /= 255.0
    return x
