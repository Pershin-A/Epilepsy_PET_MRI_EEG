import numpy as np


def proportional_scaling(
    pet: np.ndarray,
    mask: np.ndarray,
):
    """
    SPM-like proportional scaling using provided mask.
    """

    values = pet[mask]
    if values.size == 0:
        raise ValueError("Mask is empty")

    mean_val = values.mean()
    if mean_val <= 0:
        raise ValueError("Mean PET value is non-positive")

    return pet / mean_val
