import numpy as np
import nibabel as nib


def compute_percentile_map(
    data: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    mode: str = "high"
):
    """
    mode: 'high' = hyper, 'low' = hypo
    """
    values = data[mask]

    if mode == "high":
        thr = np.percentile(values, percentile)
        binmap = (data >= thr) & mask
    elif mode == "low":
        thr = np.percentile(values, 100 - percentile)
        binmap = (data <= thr) & mask
    else:
        raise ValueError("mode must be 'high' or 'low'")

    return binmap.astype(np.uint8), float(thr)
