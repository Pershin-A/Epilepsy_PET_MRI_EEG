import subprocess
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import csv
import xml.etree.ElementTree as ET

from .io_dicom import dicom_to_nifti
from .spm_utils import check_spm_outputs

from .masks import create_gm_mask
from .scaling import proportional_scaling
from .zmap import compute_zmap
from .lateralization import lateralize_from_roi_stats
from .clusters import find_z_clusters
from .visualization import plot_pet_qc
from .percentile_maps import compute_percentile_map


# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
NIFTI_DIR = PROJECT_ROOT / "data" / "nifti"

SPM_DIR = PROJECT_ROOT / "spm"
SPM_OUTPUTS_DIR = SPM_DIR / "outputs"
SPM_ATLAS_DIR = SPM_DIR / "atlas"

ATLAS_NII = SPM_ATLAS_DIR / "labels_Neuromorphometrics.nii"
ATLAS_XML = SPM_ATLAS_DIR / "labels_Neuromorphometrics.xml"
ATLAS_RESLICED = SPM_ATLAS_DIR / "rlabels_Neuromorphometrics.nii"

RESULTS_DIR = PROJECT_ROOT / "results"

# --- DICOM input ---
MRI_DICOM_DIR = (
    RAW_DATA_DIR / "MRI" / "20250922_HEAD_EPI_3429" / "4_3D_Ax_T1_MP-RAGE"
)

PET_DICOM_DIR = (
    RAW_DATA_DIR / "PET" / "20250919_BRAIN_18F-FDG_PET_1642" / "12_Static_Brain_3D_MAC"
)

# --- NIfTI output ---
T1_NIFTI = NIFTI_DIR / "T1.nii.gz"
PET_NIFTI = NIFTI_DIR / "PET.nii.gz"


# --------------------------------------------------
# UTILS
# --------------------------------------------------
def check_dir(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Required directory not found: {name} → {path}")


def run_matlab_job(job_name: str):
    print(f"Running MATLAB job: {job_name}")
    subprocess.run(
        ["matlab", "-batch", job_name],
        cwd=str(SPM_DIR),
        check=True
    )


def load_neuromorphometrics_labels(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = {}
    for label in root.findall(".//label"):
        idx_el = label.find("index")
        name_el = label.find("name")

        if idx_el is None or name_el is None:
            continue

        try:
            idx = int(idx_el.text)
        except (TypeError, ValueError):
            continue

        labels[idx] = name_el.text.strip()

    if not labels:
        raise RuntimeError("No valid labels parsed from Neuromorphometrics XML")

    return labels

def load_nifti_3d(path: Path):
    """
    Load NIfTI and гарантировать 3D массив.
    SPM иногда сохраняет (X,Y,Z,1).
    """
    img = nib.load(path)
    data = img.get_fdata()

    if data.ndim == 4:
        if data.shape[3] != 1:
            raise RuntimeError(
                f"NIfTI has unexpected 4D shape: {data.shape} ({path})"
            )
        data = data[..., 0]

    return data, img.affine, img.header



# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():

    print("=== Epilepsy PET–MRI pipeline started ===")

    # ---- sanity checks
    check_dir(RAW_DATA_DIR, "raw data")
    check_dir(MRI_DICOM_DIR, "MRI DICOM")
    check_dir(PET_DICOM_DIR, "PET DICOM")
    check_dir(SPM_ATLAS_DIR, "SPM atlas directory")

    if not ATLAS_NII.exists():
        raise RuntimeError("labels_Neuromorphometrics.nii not found in spm/atlas")
    if not ATLAS_XML.exists():
        raise RuntimeError("labels_Neuromorphometrics.xml not found in spm/atlas")

    NIFTI_DIR.mkdir(exist_ok=True, parents=True)
    SPM_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------
    # DICOM → NIfTI
    # --------------------------------------------------
    if not T1_NIFTI.exists():
        dicom_to_nifti(MRI_DICOM_DIR, NIFTI_DIR, "T1")
    else:
        print("T1 NIfTI already exists, skipping conversion.")

    if not PET_NIFTI.exists():
        dicom_to_nifti(PET_DICOM_DIR, NIFTI_DIR, "PET")
    else:
        print("PET NIfTI already exists, skipping conversion.")

    # --------------------------------------------------
    # SPM PREPROCESSING
    # --------------------------------------------------
    need_preprocess = not all(
        (SPM_OUTPUTS_DIR / f).exists()
        for f in ["wrPET.nii", "wT1.nii", "c1T1.nii", "y_T1.nii"]
    )

    if need_preprocess:
        run_matlab_job("spm_preprocess")
    else:
        print("SPM preprocessing outputs exist — skipping spm_preprocess.")

    spm_outputs = check_spm_outputs(SPM_OUTPUTS_DIR)
    for k, v in spm_outputs.items():
        print(f"  {k}: {v.name}")

    wc1 = SPM_OUTPUTS_DIR / "wc1T1.nii"
    if not wc1.exists():
        run_matlab_job("spm_warp_gm")
    else:
        print("Warped GM already exists — skipping spm_warp_gm.")

    # --------------------------------------------------
    # LOAD FINAL DATA
    # --------------------------------------------------
    pet, pet_affine, pet_header = load_nifti_3d(
        SPM_OUTPUTS_DIR / "wrPET.nii"
    )
    gm, _, _ = load_nifti_3d(
        SPM_OUTPUTS_DIR / "wc1T1.nii"
    )


    # --------------------------------------------------
    # MASK / SCALING / Z
    # --------------------------------------------------
    mask = create_gm_mask(pet, gm)
    pet_scaled = proportional_scaling(pet, mask)
    zmap = compute_zmap(pet_scaled, mask)

    zmap_path = RESULTS_DIR / "zmap.nii"
    nib.save(
        nib.Nifti1Image(zmap, pet_affine, pet_header),
        zmap_path
    )
    print(f"Voxelwise Z-map saved to: {zmap_path}")


    # --------------------------------------------------
    # ASYMMETRY MAP
    # --------------------------------------------------
    nx = pet.shape[0]
    mid = nx // 2

    left = pet_scaled[:mid]
    right = pet_scaled[-mid:][::-1]

    m_left = mask[:mid]
    m_right = mask[-mid:][::-1]

    valid = m_left & m_right
    asym_vals = (left[valid] - right[valid]) / ((left[valid] + right[valid]) / 2)

    asym = np.zeros_like(pet_scaled, dtype=np.float32)
    asym[:mid][valid] = asym_vals
    asym[-mid:][::-1][valid] = -asym_vals

    # --------------------------------------------------
    # NEUROMORPHOMETRICS ROI ANALYSIS
    # --------------------------------------------------
    if not ATLAS_RESLICED.exists():
        run_matlab_job("spm_reslice_atlas")
    else:
        print("Resliced Neuromorphometrics atlas exists — skipping.")

    atlas, atlas_affine, _ = load_nifti_3d(ATLAS_RESLICED)
    atlas = atlas.astype(int)

    labels = load_neuromorphometrics_labels(ATLAS_XML)

    # --------------------------------------------------
    # ATLAS QC
    # --------------------------------------------------
    atlas_data, atlas_affine, _ = load_nifti_3d(ATLAS_RESLICED)
    atlas_data = atlas_data.astype(int)


    # --- 1. Affine check
    aff_diff = np.abs(pet_affine - atlas_affine).max()
    print(f"Atlas affine max diff: {aff_diff:.3e}")


    if aff_diff > 1e-4:
        raise RuntimeError("Atlas affine does not match PET affine")

    # --- 2. Label sanity
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels > 0]

    missing = sorted(set(unique_labels) - set(labels.keys()))
    if missing:
        raise RuntimeError(f"Atlas contains labels not in XML: {missing[:10]}")

    print(f"Atlas contains {len(unique_labels)} non-zero labels")

    # --- 3. Hemisphere voxel balance check (CORRECT for Neuromorphometrics)
    nx = atlas_data.shape[0]
    mid = nx // 2

    left_mask = atlas_data[:mid] > 0
    right_mask = atlas_data[mid:] > 0

    left_vox = int(left_mask.sum())
    right_vox = int(right_mask.sum())

    ratio = left_vox / right_vox if right_vox > 0 else np.inf

    print(f"Atlas hemisphere voxel count: L={left_vox}, R={right_vox}, ratio={ratio:.3f}")

    if ratio < 0.8 or ratio > 1.25:
        raise RuntimeError("Severe hemispheric imbalance in atlas")
    
    # --- 4. Label naming sanity
    left_named = [name for name in labels.values() if "Left" in name]
    right_named = [name for name in labels.values() if "Right" in name]

    print(f"Named ROIs: Left={len(left_named)}, Right={len(right_named)}")

    if abs(len(left_named) - len(right_named)) > 2:
        raise RuntimeError("Left/Right label naming imbalance")


    print("Atlas QC PASSED")



    if atlas.shape != mask.shape:
        raise RuntimeError(
            f"Atlas shape {atlas.shape} does not match PET shape {mask.shape}"
        )

    print(f"Parsed {len(labels)} Neuromorphometrics regions")

    roi_stats = []
    roi_asym = []

    for idx, name in labels.items():
        roi_mask = (atlas == idx) & mask
        nvox = int(roi_mask.sum())
        if nvox < 50:
            continue

        roi_stats.append({
            "ROI": name,
            "Index": idx,
            "Voxels": nvox,
            "PET_mean": float(pet_scaled[roi_mask].mean()),
            "Z_mean": float(zmap[roi_mask].mean()),
        })

        roi_asym.append({
            "ROI": name,
            "Index": idx,
            "Asymmetry_mean": float(asym[roi_mask].mean()),
        })
    lat = lateralize_from_roi_stats(roi_stats)

    print("\n=== Automatic Lateralization ===")
    print(f"Laterality   : {lat['laterality']}")
    print(f"Localization : {lat['localization']}")
    print(f"Confidence   : {lat['confidence']:.2f}")

    # --------------------------------------------------
    # Z < -2 CLUSTER ANALYSIS
    # --------------------------------------------------
    for min_vox in (50, 100):

        clusters, cl_mask = find_z_clusters(
            zmap=zmap,
            mask=mask,
            affine=pet_affine,
            atlas=atlas,
            labels=labels,
            z_thresh=-2.0,
            min_size=min_vox,
        )


        print(f"\nZ-clusters (min {min_vox} voxels): {len(clusters)}")

        # --- save CSV
        csv_path = RESULTS_DIR / f"z_clusters_min{min_vox}.csv"
        if clusters:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=clusters[0].keys())
                writer.writeheader()
                writer.writerows(clusters)

            print(f"Cluster table saved to: {csv_path}")
        else:
            print("No clusters detected.")

        # --- save NIfTI mask
        nii_path = RESULTS_DIR / f"z_clusters_min{min_vox}.nii"
        cl_img = nib.Nifti1Image(cl_mask, pet_affine, pet_header)
        nib.save(cl_img, nii_path)

        print(f"Cluster mask saved to: {nii_path}")

    # --------------------------------------------------
    # Z > +2 CLUSTER ANALYSIS (HYPERMETABOLISM)
    # --------------------------------------------------
    for min_vox in (50, 100):

        clusters_h, cl_mask_h = find_z_clusters(
            zmap=-zmap,              # инверсия знака → reuse функции
            mask=mask,
            affine=pet_affine,
            atlas=atlas,
            labels=labels,
            z_thresh=-2.0,           # фактически Z > +2
            min_size=min_vox,
        )

        print(f"\nHypermetabolic clusters (min {min_vox} voxels): {len(clusters_h)}")

        csv_path = RESULTS_DIR / f"z_clusters_hyper_min{min_vox}.csv"
        if clusters_h:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=clusters_h[0].keys())
                writer.writeheader()
                writer.writerows(clusters_h)

        nii_path = RESULTS_DIR / f"z_clusters_hyper_min{min_vox}.nii"
        nib.save(
            nib.Nifti1Image(cl_mask_h, pet_affine, pet_header),
            nii_path
        )

    #(Опционально) Процентильный порог
    p = 99  # топ 1%
    thr = np.percentile(pet_scaled[mask], p)
    hyper_mask = (pet_scaled > thr) & mask

    #визуализация QC
    plot_pet_qc(
        t1_path=SPM_OUTPUTS_DIR / "wT1.nii",
        pet_path=SPM_OUTPUTS_DIR / "wrPET.nii",
        zmap=zmap,
        clusters_hypo=cl_mask,
        clusters_hyper=cl_mask_h,
        out_png=RESULTS_DIR / "QC_PET_MRI.png"
    )

    hyper_pct, thr_h = compute_percentile_map(
        pet_scaled, mask, percentile=99, mode="high"
    )

    hypo_pct, thr_l = compute_percentile_map(
        pet_scaled, mask, percentile=1, mode="low"
    )

    nib.save(
        nib.Nifti1Image(hyper_pct, pet_affine, pet_header),
        RESULTS_DIR / "hyper_percentile_99.nii"
    )

    nib.save(
        nib.Nifti1Image(hypo_pct, pet_affine, pet_header),
        RESULTS_DIR / "hypo_percentile_1.nii"
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------
    stats_csv = RESULTS_DIR / "roi_pet_stats.csv"
    asym_csv = RESULTS_DIR / "roi_pet_asymmetry.csv"

    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=roi_stats[0].keys())
        writer.writeheader()
        writer.writerows(roi_stats)

    with open(asym_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=roi_asym[0].keys())
        writer.writeheader()
        writer.writerows(roi_asym)

    print(f"ROI stats saved to: {stats_csv}")
    print(f"ROI asymmetry saved to: {asym_csv}")
    print("=== Pipeline finished successfully ===")


# --------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nPIPELINE FAILED:")
        print(e)
        sys.exit(1)
