import shutil
import subprocess
from pathlib import Path

def dicom_to_nifti(dicom_dir: Path, out_dir: Path, out_name: str):
    if shutil.which("dcm2niix") is None:
        raise RuntimeError("dcm2niix not found in PATH")

    out_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", out_name,
        "-o", str(out_dir),
        str(dicom_dir)
    ]

    subprocess.run(cmd, check=True)
