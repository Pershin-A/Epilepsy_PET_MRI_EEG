import numpy as np
import nibabel as nib


def compute_pet_eeg_overlap(
    pet_cluster_mask: np.ndarray,
    eeg_source_map: np.ndarray,
    threshold: float = 0.9
):
    """
    eeg_source_map — normalized [0..1]
    """

    eeg_bin = eeg_source_map > threshold
    pet_bin = pet_cluster_mask > 0

    intersection = eeg_bin & pet_bin

    dice = (
        2 * intersection.sum() /
        (eeg_bin.sum() + pet_bin.sum() + 1e-6)
    )

    return {
        "dice": float(dice),
        "overlap_voxels": int(intersection.sum()),
        "pet_voxels": int(pet_bin.sum()),
        "eeg_voxels": int(eeg_bin.sum())
    }
