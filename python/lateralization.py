import math


TEMPORAL_KEYWORDS = [
    "Hippocampus",
    "Temporal",
    "Fusiform",
    "Parahippocampal",
    "Amygdala",
]


def is_temporal(roi_name: str) -> bool:
    return any(k in roi_name for k in TEMPORAL_KEYWORDS)


def lateralize_from_roi_stats(roi_stats, z_thresh=-2.0):
    """
    roi_stats: list of dicts with keys:
        ROI, Z_mean
    """

    sides = {
        "Left": {"temporal": [], "extra": []},
        "Right": {"temporal": [], "extra": []},
    }

    for r in roi_stats:
        z = r["Z_mean"]
        if z >= z_thresh:
            continue

        name = r["ROI"]

        if "Left" in name:
            side = "Left"
        elif "Right" in name:
            side = "Right"
        else:
            continue

        if is_temporal(name):
            sides[side]["temporal"].append(z)
        else:
            sides[side]["extra"].append(z)

    def burden(zvals):
        return sum(abs(z) for z in zvals)

    summary = {}
    for side in ["Left", "Right"]:
        t = sides[side]["temporal"]
        e = sides[side]["extra"]

        summary[side] = {
            "temporal_burden": burden(t),
            "extratemporal_burden": burden(e),
            "total_burden": burden(t) + burden(e),
            "n_temporal": len(t),
            "n_extratemporal": len(e),
        }

    # --- lateralization decision
    L = summary["Left"]["total_burden"]
    R = summary["Right"]["total_burden"]

    if L > 1.5 * R:
        laterality = "Left"
        ratio = L / max(R, 1e-6)
    elif R > 1.5 * L:
        laterality = "Right"
        ratio = R / max(L, 1e-6)
    else:
        laterality = "Bilateral"
        ratio = 1.0

    # --- localization
    t_total = (
        summary["Left"]["temporal_burden"]
        + summary["Right"]["temporal_burden"]
    )
    total = L + R

    if total == 0:
        localization = "Indeterminate"
    elif t_total / total >= 0.7:
        localization = "Temporal"
    elif t_total / total <= 0.3:
        localization = "Extratemporal"
    else:
        localization = "Multifocal"

    confidence = min(1.0, math.log10(1 + ratio))

    return {
        "laterality": laterality,
        "localization": localization,
        "confidence": confidence,
        "details": summary,
    }
