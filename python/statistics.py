import numpy as np

def compute_zmap(pet_data, gm_mask):
    values = pet_data[gm_mask]
    mean = np.mean(values)
    std = np.std(values)
    z = np.zeros_like(pet_data)
    z[gm_mask] = (pet_data[gm_mask] - mean) / std
    return z
