from pathlib import Path
import shutil

ROOT = Path("data/normative")
OLD_DICOM = ROOT / "ADNI_FDG"
OLD_XML = ROOT / "ADNI_FDG_METADATA"
NEW_RAW = ROOT / "raw"

def find_all_dicoms(root):
    return list(root.rglob("*.dcm"))

def reorganize():
    NEW_RAW.mkdir(exist_ok=True)

    for dcm in find_all_dicoms(OLD_DICOM):
        # Subject ID берём из пути
        parts = dcm.parts
        subj = [p for p in parts if "_S_" in p][0]

        new_dir = NEW_RAW / f"sub-{subj}" / "pet_fdg"
        new_dir.mkdir(parents=True, exist_ok=True)

        # Новое имя
        idx = len(list(new_dir.glob("*.dcm"))) + 1
        new_name = new_dir / f"{subj}_fdg_{idx:03d}.dcm"

        shutil.copy2(dcm, new_name)

    print("DICOM reorganization complete.")

if __name__ == "__main__":
    reorganize()