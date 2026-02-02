import numpy as np


def create_gm_mask(
    pet: np.ndarray,
    gm: np.ndarray,
    gm_threshold: float = 0.3,
    pet_percentile: float = 20.0,
    min_voxels: int = 10000,
):
    """
    Create final GM mask in MNI space:
    GM probability thresholded AND PET-based brain mask.
    """

    if pet.shape != gm.shape:
        raise ValueError(
            f"Shape mismatch: PET {pet.shape} vs GM {gm.shape}"
        )

    # --- GM mask ---
    gm_mask = gm > gm_threshold

    # --- PET brain mask ---
    pet_positive = pet[pet > 0]
    if pet_positive.size == 0:
        raise ValueError("PET contains no positive voxels")

    pet_thresh = np.percentile(pet_positive, pet_percentile)
    pet_brain_mask = pet > pet_thresh

    # --- Final mask ---
    final_mask = gm_mask & pet_brain_mask

    nvox = int(final_mask.sum())
    if nvox < min_voxels:
        raise RuntimeError(
            f"GM mask too small ({nvox} voxels)"
        )

    return final_mask
