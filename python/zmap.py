import numpy as np


def compute_zmap(
    pet_scaled: np.ndarray,
    mask: np.ndarray,
):
    """
    Compute single-subject Z-map inside mask.
    """

    values = pet_scaled[mask]
    mu = values.mean()
    sigma = values.std(ddof=1)

    if sigma == 0:
        raise ValueError("Zero variance in PET values")

    z = np.zeros_like(pet_scaled, dtype=np.float32)
    z[mask] = (pet_scaled[mask] - mu) / sigma

    return z
