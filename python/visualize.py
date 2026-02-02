import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage import measure

# ============================================================
# PARAMETERS
# ============================================================

Z_THR_HYPO = -2.0
Z_THR_HYPER = 2.0
ROI_FRACTION_THR = 0.10
MAX_AUTO_SLICES = 5
BRAIN_ALPHA_3D = 0.03
MAX_3D_POINTS = 20000

# ============================================================
# COLORMAP
# ============================================================

CMAP = LinearSegmentedColormap.from_list(
    "hypo_hyper",
    [(1.0, 0.9, 0.0), (0, 0, 0), (0.0, 0.8, 0.0)],
    N=256
)

# ============================================================
# UTILS
# ============================================================

def safe_parse_slices(s):
    s = s.strip()
    if not s:
        return []
    return [int(v) for v in s.split(",") if v.strip().isdigit()]

def get_slice(data, axis, idx):
    if axis == "x":
        return data[idx, :, :]
    if axis == "y":
        return data[:, idx, :]
    if axis == "z":
        return data[:, :, idx]

def axis_size(data, axis):
    return data.shape["xyz".index(axis)]

def plot_slice(ax, base, overlay, title, norm=None):
    ax.imshow(base, cmap="gray", origin="lower")
    if overlay is not None:
        ax.imshow(
            overlay,
            cmap=CMAP,
            origin="lower",
            alpha=0.9,
            norm=norm
        )
    ax.set_title(title, fontsize=9, pad=2)
    ax.axis("off")

# ============================================================
# CLUSTER CONTOUR (STABLE VERSION)
# ============================================================

def draw_cluster_contour(ax, mask2d):
    if mask2d is None:
        return

    contours = measure.find_contours(mask2d.astype(float), 0.5)
    for c in contours:
        ax.plot(
            c[:, 1],
            c[:, 0],
            color="red",
            linewidth=0.6,
            alpha=0.9
        )

# ============================================================
# CLUSTER ANALYSIS
# ============================================================

def summarize_clusters(cl_mask, zmap, atlas, labels):
    clusters = {}
    for cid in np.unique(cl_mask):
        if cid == 0:
            continue

        vox = np.where(cl_mask == cid)
        zvals = zmap[vox]

        roi_ids, counts = np.unique(atlas[vox], return_counts=True)
        roi_ids = roi_ids[roi_ids > 0]
        counts = counts[:len(roi_ids)]

        rois = []
        if counts.sum() > 0:
            for r, c in zip(roi_ids, counts):
                frac = c / counts.sum()
                if frac >= ROI_FRACTION_THR:
                    rois.append(f"{labels.get(int(r),'Unknown')} ({int(frac*100)}%)")

        clusters[int(cid)] = {
            "voxels": len(zvals),
            "mean_z": float(zvals.mean()),
            "peak_z": float(
                zvals.min() if abs(zvals.min()) > abs(zvals.max()) else zvals.max()
            ),
            "rois": rois or ["Outside atlas"],
            "coords": vox
        }
    return clusters

def print_cluster_table(title, clusters):
    print(f"\n{title}")
    print(f"{'ID':<4} {'Vox':<6} {'MeanZ':<7} {'PeakZ':<7} ROI(s)")
    print("-" * 100)
    for cid, c in clusters.items():
        print(
            f"{cid:<4} {c['voxels']:<6} "
            f"{c['mean_z']:<7.2f} {c['peak_z']:<7.2f} "
            f"{'; '.join(c['rois'])}"
        )

# ============================================================
# MANUAL SLICE VIEWER
# ============================================================

def manual_slice_viewer(t1, zmap, cl_mask, mode, norm, cluster_info=None):
    print(f"Activity map size: X={t1.shape[0]}, Y={t1.shape[1]}, Z={t1.shape[2]}")

    xs = safe_parse_slices(input("X slices: "))
    ys = safe_parse_slices(input("Y slices: "))
    zs = safe_parse_slices(input("Z slices: "))

    rows = [(a, s) for a, s in [("x", xs), ("y", ys), ("z", zs)] if s]
    if not rows:
        return

    rows_per_axis = 2 if mode == "both" else 1
    nrows = rows_per_axis * len(rows)
    ncols = max(len(s) for _, s in rows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.6*nrows))
    axes = np.atleast_2d(axes)

    if cluster_info:
        fig.suptitle(cluster_info, fontsize=14)

    r = 0
    for axis, idxs in rows:
        size = axis_size(t1, axis)
        pad = (ncols - len(idxs)) // 2

        for c in range(ncols):
            axes[r, c].axis("off")
            if mode == "both":
                axes[r+1, c].axis("off")

        for i, idx in enumerate(idxs):
            base = get_slice(t1, axis, idx).T
            z = get_slice(zmap, axis, idx).T
            cl = get_slice(cl_mask, axis, idx).T if cl_mask is not None else None

            if mode in {"all", "both"}:
                overlay = np.ma.masked_where(
                    (z > Z_THR_HYPO) & (z < Z_THR_HYPER), z
                )
                plot_slice(
                    axes[r, i+pad], base, overlay,
                    f"{axis.upper()} = {idx}/{size} all",
                    norm
                )

            if mode in {"clusters", "both"}:
                rr = r if mode == "clusters" else r + 1
                overlay = np.ma.masked_where(cl == 0, z)
                plot_slice(
                    axes[rr, i+pad], base, overlay,
                    f"{axis.upper()} = {idx}/{size} clusters",
                    norm
                )
                draw_cluster_contour(axes[rr, i+pad], cl)

        r += rows_per_axis

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
        cax=cax,
        label="Z-score"
    )

    plt.subplots_adjust(
        left=0.04, right=0.88, top=0.90, bottom=0.05,
        wspace=0.10, hspace=0.18
    )
    plt.show()

# ============================================================
# AUTO CLUSTER VIEW
# ============================================================

def auto_cluster_view(t1, zmap, cl_mask, cluster, cid, title, norm):
    xs, ys, zs = cluster["coords"]

    fig, axes = plt.subplots(3, MAX_AUTO_SLICES, figsize=(15, 7))
    fig.suptitle(
        f"{title}\n"
        f"Voxels={cluster['voxels']} | "
        f"MeanZ={cluster['mean_z']:.2f} | "
        f"PeakZ={cluster['peak_z']:.2f}",
        fontsize=14
    )

    for r, (axis, arr) in enumerate([("x", xs), ("y", ys), ("z", zs)]):
        uniq, counts = np.unique(arr, return_counts=True)
        idxs = sorted(uniq[np.argsort(counts)[::-1][:MAX_AUTO_SLICES]])
        size = axis_size(t1, axis)

        for c in range(MAX_AUTO_SLICES):
            axes[r, c].axis("off")

        for c, idx in enumerate(idxs):
            base = get_slice(t1, axis, idx).T
            z = get_slice(zmap, axis, idx).T
            cl = (get_slice(cl_mask, axis, idx) == cid).T

            overlay = np.ma.masked_where(cl == 0, z)
            plot_slice(
                axes[r, c], base, overlay,
                f"{axis.upper()} = {idx}/{size}",
                norm
            )
            draw_cluster_contour(axes[r, c], cl)

    plt.subplots_adjust(
        left=0.04, right=0.88, top=0.88, bottom=0.05,
        wspace=0.10, hspace=0.18
    )
    plt.show()

# ============================================================
# 3D VISUALIZATION
# ============================================================

def plot_3d(t1, zmap):
    ans = input("Show 3D visualization? [y/n]: ").lower()
    if ans != "y":
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    brain = t1 > np.percentile(t1, 70)
    pts = np.array(np.where(brain))
    if pts.shape[1] > MAX_3D_POINTS:
        pts = pts[:, np.random.choice(pts.shape[1], MAX_3D_POINTS, replace=False)]
    ax.scatter(*pts, s=1, c="gray", alpha=BRAIN_ALPHA_3D)

    hypo = np.where(zmap <= Z_THR_HYPO)
    hyper = np.where(zmap >= Z_THR_HYPER)

    ax.scatter(*hypo, s=4, c="gold", alpha=0.7, label="Hypo")
    ax.scatter(*hyper, s=4, c="green", alpha=0.7, label="Hyper")

    ax.legend()
    ax.set_title("3D PET abnormalities with brain outline")
    plt.show()

# ============================================================
# MAIN
# ============================================================

def visualize(results_dir):
    root = results_dir.parent

    t1 = nib.load(root / "spm/outputs/wT1.nii").get_fdata()
    zmap = nib.load(results_dir / "zmap.nii").get_fdata()

    GLOBAL_NORM = Normalize(vmin=np.nanmin(zmap), vmax=np.nanmax(zmap))

    cl_hypo = nib.load(results_dir / "z_clusters_min50.nii").get_fdata()
    cl_hyper = nib.load(results_dir / "z_clusters_hyper_min50.nii").get_fdata()

    atlas = nib.load(root / "spm/atlas/rlabels_Neuromorphometrics.nii").get_fdata().astype(int)
    roi = pd.read_csv(results_dir / "roi_pet_stats.csv")
    labels = dict(zip(roi["Index"], roi["ROI"]))

    total_vox = np.prod(zmap.shape)
    hypo_mask = zmap <= Z_THR_HYPO
    hyper_mask = zmap >= Z_THR_HYPER

    print("\nVoxel statistics:")
    print(f"Hypo voxels        : {hypo_mask.sum()} ({100*hypo_mask.sum()/total_vox:.2f}%)")
    print(f"Hyper voxels       : {hyper_mask.sum()} ({100*hyper_mask.sum()/total_vox:.2f}%)")
    print(f"Hypo cluster vox   : {(cl_hypo > 0).sum()}")
    print(f"Hyper cluster vox  : {(cl_hyper > 0).sum()}")

    clusters_hypo = summarize_clusters(cl_hypo, zmap, atlas, labels)
    clusters_hyper = summarize_clusters(cl_hyper, zmap, atlas, labels)

    print_cluster_table("Hypometabolic clusters", clusters_hypo)
    print_cluster_table("Hypermetabolic clusters", clusters_hyper)

    mask_all = (cl_hypo > 0) | (cl_hyper > 0)

    while True:
        print("\nSelect visualization mode:")
        print("  1 - all suprathreshold voxels")
        print("  2 - cluster voxels only")
        print("  3 - all + clusters")
        print("  4 - single hypometabolic cluster")
        print("  5 - single hypermetabolic cluster")
        print("  6 - 3D visualization")
        print("  q - quit")

        c = input("Choice: ").strip()
        if c == "q":
            break

        if c in {"1", "2", "3"}:
            mode = {"1": "all", "2": "clusters", "3": "both"}[c]
            manual_slice_viewer(t1, zmap, mask_all, mode, GLOBAL_NORM)

        elif c in {"4", "5"}:
            clusters = clusters_hypo if c == "4" else clusters_hyper
            cl_mask = cl_hypo if c == "4" else cl_hyper
            name = "Hypometabolic" if c == "4" else "Hypermetabolic"

            cid = int(input(f"{name} cluster ID: "))
            if cid not in clusters:
                continue

            cluster = clusters[cid]
            title = (
                f"{name} cluster {cid}\n"
                f"Voxels={cluster['voxels']} | "
                f"MeanZ={cluster['mean_z']:.2f} | "
                f"PeakZ={cluster['peak_z']:.2f}"
            )

            print("\nSelect slice mode:")
            print("  1 - automatic")
            print("  2 - manual")

            m = input("Choice: ").strip()
            if m == "1":
                auto_cluster_view(
                    t1, zmap, cl_mask, cluster, cid, name, GLOBAL_NORM
                )
            elif m == "2":
                manual_slice_viewer(
                    t1, zmap,
                    (cl_mask == cid),
                    "clusters",
                    GLOBAL_NORM,
                    cluster_info=title
                )

        elif c == "6":
            plot_3d(t1, zmap)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    visualize(Path(args.results))

if __name__ == "__main__":
    main()
