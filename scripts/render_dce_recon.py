"""Report version of uuv-rl-env encoder/render_dce.py (Figure: DCE reconstructions).
Changes from the original: no segmentation column, no legend, no debug title, report column
titles. Run inside the Isaac Sim container with the uuv-rl-env repo bound to /work, see
scripts/render_figures.slurm.

Original docstring:
Render a trained DCE on held-out images: depth in, collision image target, decoded output, error.

Rebuilds the run's train/val split (same seed, groups and drop filter as encoder/train_dce.py),
so the images shown are from validation episodes the model never trained on. Picks one image per
kind (obstacle, maze wall, procedural structure, close anchor, toward view, open water) unless
sample numbers are given. Runs in the container (torch):
    /isaac-sim/python.sh encoder/render_dce.py <run dir> <out.png> [val sample numbers...]
On Idun via a short CPUQ job (see encoder/datasets.md, D001 encoder runs).
"""

import os
import sys

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, "/work/encoder")  # report copy: import the encoder from the repo
from dce import DCE  # noqa: E402
from train_dce import cache_path, load_dataset, robot_of  # noqa: E402
import glob  # noqa: E402

SEG_NAMES = ["no hit", "floor", "tank", "maze wall", "obstacle", "vehicle", "structure"]
SEG_CMAP = ListedColormap(["#000000", "#8c6d31", "#6b6ecf", "#e6550d", "#31a354", "#bdbdbd", "#d62728"])


def main():
    run_dir, out = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    torch.manual_seed(cfg.seed)  # train_dce.py seeds right before loading; load_dataset uses no torch RNG
    drop, sat = cfg.data.get("drop_saturated"), cfg.data.get("saturated_range", 0.05)
    robot = robot_of(cfg.data)
    x, y, group = load_dataset(cfg.data.path, cfg.data.max_range, robot, 1, drop, sat)
    n_groups = int(group.max()) + 1
    val_groups = torch.randperm(n_groups)[: max(1, int(cfg.data.val_fraction * n_groups))]
    val_idx = torch.isin(group, val_groups).nonzero().flatten().numpy()

    # Per-image metadata in load_dataset's order (same files, same keep mask).
    seg, pert, reason, keys = [], [], [], []
    for f in sorted(glob.glob(os.path.join(cfg.data.path, "**", "shard_*.npz"), recursive=True)):
        z = np.load(f)
        keep = np.ones(len(z["depth"]), bool)
        if drop is not None and "perturbed" in z:
            keep = (z["perturbed"] == 0) | ((np.load(cache_path(f, robot)) < sat).mean((1, 2)) <= drop)
        keys += [f"{f}:{i}" for i in np.nonzero(keep)[0]]
        seg.append(z["segmentation"][keep]); pert.append(z["perturbed"][keep]); reason.append(z["reason"][keep])
    seg, pert, reason = np.concatenate(seg), np.concatenate(pert), np.concatenate(reason)
    assert len(seg) == len(x)

    pos = {int(i): k for k, i in enumerate(val_idx)}
    row = {k: i for i, k in enumerate(keys)}
    # Picks: val sample numbers, or sample keys "<shard>.npz:<index>" (printed below, stable
    # across collision sets, whose drop filters shift the numbers).
    picks = [pos[row[a]] if ":" in a else int(a) for a in args]
    if not picks:
        rng = np.random.default_rng(1)
        R = x[val_idx, 0].float().numpy() * cfg.data.max_range
        s = seg[val_idx]
        kinds = [("obstacle", (s == 4).mean((1, 2)) > 0.08),
                 ("maze wall", (s == 3).mean((1, 2)) > 0.25),
                 ("structure", (s == 6).mean((1, 2)) > 0.15),
                 ("close anchor", (pert[val_idx] == 0) & (R.min((1, 2)) < 0.6) & (R.min((1, 2)) > 0.3)),
                 ("toward view", pert[val_idx] == 2),
                 ("open water", (s == 0).mean((1, 2)) > 0.5)]
        picks = [int(rng.choice(np.nonzero(m)[0])) for _, m in kinds if m.any()]
    idx = val_idx[picks]

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    model = DCE(ckpt["image_hw"], **ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        recon = model(x[idx].float())[0]   # clean input (no augmentation); eval uses z = mu

    mr = cfg.data.max_range
    cols = ["Depth image (input)", "Collision image (target)", "Reconstruction", "Absolute error"]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(3.1 * len(cols), 2.0 * len(idx) + 0.8), squeeze=False)
    kw = dict(cmap="viridis_r", vmin=0, vmax=mr)
    kind = {0: "anchor", 1: "gauss view", 2: "toward view"}
    for row, i, p, rc in zip(axes, idx, picks, recon):
        R, T, D = x[i, 0].float().numpy() * mr, y[i, 0].float().numpy() * mr, rc[0].numpy() * mr
        row[0].imshow(R, **kw)
        row[1].imshow(T, **kw)
        im = row[2].imshow(D, **kw)
        er = row[3].imshow(np.abs(D - T), cmap="magma", vmin=0, vmax=2.0)
        rmse = float(np.sqrt(((D - T) ** 2).mean()))
        row[0].set_ylabel(f"{kind[int(pert[i])]}\nmin. range {R.min():.2f} m", fontsize=8)
        row[3].set_ylabel(f"RMSE {rmse:.2f} m", fontsize=8)
        row[3].yaxis.set_label_position("right")
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    for a, t in zip(axes[0], cols):
        a.set_title(t, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.10, wspace=0.08, hspace=0.12)
    fig.colorbar(im, cax=fig.add_axes([0.06, 0.05, 0.64, 0.015]), orientation="horizontal", label="Range (m)")
    fig.colorbar(er, cax=fig.add_axes([0.73, 0.05, 0.21, 0.015]), orientation="horizontal", label="Error (m)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out, "val samples", picks)
    print("keys:", " ".join(keys[i] for i in idx))


if __name__ == "__main__":
    main()
