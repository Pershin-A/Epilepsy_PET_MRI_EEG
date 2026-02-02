from pathlib import Path
import nibabel as nib

#ШАГ 1. Проверка, что атлас реально есть
spm_root = Path("C:/Users/andrew/Documents/MATLAB/spm")

atlas_nii = spm_root / "tpm" / "labels_Neuromorphometrics.nii"
atlas_txt = spm_root / "tpm" / "labels_Neuromorphometrics.xml"

print(atlas_nii.exists(), atlas_txt.exists())


#ШАГ 2. ROI-анализ с Neuromorphometrics
atlas_img = nib.load(atlas_nii)
atlas = atlas_img.get_fdata().astype(int)

labels = {}
with open(atlas_txt, encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(maxsplit=1)
        labels[int(idx)] = name
