import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

def proportional_scaling(pet_img, gm_mask):
    data = pet_img.get_fdata()
    mean_gm = np.mean(data[gm_mask])
    return data / mean_gm

def smooth_pet(data, affine, fwhm_mm):
    voxel_size = np.abs(np.diag(affine)[:3])
    sigma = fwhm_mm / (2.355 * voxel_size)
    return gaussian_filter(data, sigma=sigma)
