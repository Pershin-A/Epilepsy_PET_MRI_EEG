import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def plot_pet_qc(
    t1_path,
    pet_path,
    zmap,
    clusters_hypo,
    clusters_hyper,
    out_png
):
    t1 = nib.load(t1_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()

    z = zmap
    mid = t1.shape[0] // 2

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for ax in axes.ravel():
        ax.axis("off")

    slices = [mid-10, mid, mid+10]

    for i, s in enumerate(slices):
        axes[i, 0].imshow(t1[s].T, cmap="gray", origin="lower")
        axes[i, 0].set_title("T1")

        axes[i, 1].imshow(pet[s].T, cmap="hot", origin="lower")
        axes[i, 1].set_title("PET scaled")

        axes[i, 2].imshow(z[s].T, cmap="coolwarm", origin="lower", vmin=-4, vmax=4)
        axes[i, 2].set_title("Z-map")

        axes[i, 3].imshow(clusters_hypo[s].T, cmap="Blues", origin="lower")
        axes[i, 3].set_title("Hypometabolism")

        axes[i, 4].imshow(clusters_hyper[s].T, cmap="Reds", origin="lower")
        axes[i, 4].set_title("Hypermetabolism")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
