from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SPM_DIR = PROJECT_ROOT / "spm"
RESULTS_DIR = PROJECT_ROOT / "results"

FWHM_MM = 8.0
GM_THRESHOLD = 0.3
PERCENTILES = [95, 99]
MIN_CLUSTER_SIZE = 100
