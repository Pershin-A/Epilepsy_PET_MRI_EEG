import numpy as np
import nibabel as nib
from scipy.ndimage import label


def find_z_clusters(
    zmap: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    atlas: np.ndarray,
    labels: dict,
    z_thresh: float = -2.0,
    min_size: int = 100,
):
    """
    Find Z clusters and compute cluster statistics.
    All inputs MUST be 3D.
    """

    # --- SAFETY CHECKS ---
    if atlas.ndim == 4:
        atlas = atlas[..., 0]

    if zmap.shape != mask.shape or atlas.shape != zmap.shape:
        raise RuntimeError(
            f"Shape mismatch: zmap {zmap.shape}, mask {mask.shape}, atlas {atlas.shape}"
        )

    # --- threshold
    binary = (zmap < z_thresh) & mask

    # --- 26-connectivity
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, n_clust = label(binary, structure=structure)

    clusters = []
    cluster_mask = np.zeros_like(zmap, dtype=np.int16)

    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    cluster_id = 1

    for cid in range(1, n_clust + 1):
        voxels = np.where(labeled == cid)
        nvox = len(voxels[0])

        if nvox < min_size:
            continue

        zvals = zmap[voxels]
        peak_idx = np.argmin(zvals)

        peak_voxel = (
            voxels[0][peak_idx],
            voxels[1][peak_idx],
            voxels[2][peak_idx],
        )

        peak_mni = nib.affines.apply_affine(affine, peak_voxel)

        hemi = "Left" if peak_voxel[0] < zmap.shape[0] // 2 else "Right"

        roi_ids, counts = np.unique(atlas[voxels], return_counts=True)
        valid = roi_ids > 0
        roi_ids = roi_ids[valid]
        counts = counts[valid]

        if len(roi_ids) > 0:
            dominant_roi = labels.get(
                roi_ids[np.argmax(counts)], "Unknown"
            )
        else:
            dominant_roi = "Outside atlas"

        com_voxel = (
            voxels[0].mean(),
            voxels[1].mean(),
            voxels[2].mean()
        )

        com_mni = nib.affines.apply_affine(affine, com_voxel)


        clusters.append({
            "Cluster": cluster_id,
            "Voxels": nvox,
            "Volume_ml": nvox * voxel_volume / 1000.0,
            "Peak_Z": float(zvals.min()),
            "Peak_x": float(peak_mni[0]),
            "Peak_y": float(peak_mni[1]),
            "Peak_z": float(peak_mni[2]),
            "Hemisphere": hemi,
            "Dominant_ROI": dominant_roi,
            "COM_x": float(com_mni[0]),
            "COM_y": float(com_mni[1]),
            "COM_z": float(com_mni[2]),

        })

        cluster_mask[labeled == cid] = cluster_id
        cluster_id += 1
    

    return clusters, cluster_mask
