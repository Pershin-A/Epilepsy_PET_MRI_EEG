from pathlib import Path

SPM_OUTPUTS = {
    "t1_native": "T1.nii",
    "t1_mni": "wT1.nii",
    "gm_native": "c1T1.nii",
    "wm_native": "c2T1.nii",
    "csf_native": "c3T1.nii",
    "deformation": "y_T1.nii",
    "pet_mni": "wrPET.nii",
}


def check_spm_outputs(spm_dir: Path) -> dict:
    if not spm_dir.exists():
        raise FileNotFoundError(f"SPM output directory not found: {spm_dir}")

    outputs = {}

    # --- обязательные файлы ---
    for key, fname in SPM_OUTPUTS.items():
        f = spm_dir / fname

        if not f.exists():
            # fallback для PET
            if key == "pet_mni":
                candidates = list(spm_dir.glob("w*rPET*.nii"))
                if len(candidates) == 1:
                    outputs[key] = candidates[0]
                    continue
            raise FileNotFoundError(f"Missing required SPM output: {fname}")

        outputs[key] = f

    return outputs