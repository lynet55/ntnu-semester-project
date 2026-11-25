"""Figure: the steps that turn a depth image into the collision image (Method, depth encoder).

Uses uuv-rl-env encoder/collision_image.py unchanged (edges, edge sampling, box ray casting,
offset, minimum) with the box robot ROBOT_BOX of the final encoder (D001-C030b), on the same
validation images as the reconstruction figure. Run inside the Isaac Sim container with the
uuv-rl-env repo bound to /work (numpy, matplotlib, PyYAML), see scripts/render_figures.slurm:

    /isaac-sim/python.sh /scripts/render_collision_pipeline.py <out.png>

The edge sampling is random (seed 0), so the sampled points can differ from the cached training
targets; the steps and the robot are the same.
"""
import os
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/work/encoder")
import collision_image as ci  # noqa: E402

DATASET = "/work/experiments/datasets/D001_mix_r3x6_k2"
# (task folder, shard, index): validation samples #3307, #2723 and #8261 of the reconstruction
# figure (keys printed by render_dce_recon.py): an obstacle, a maze wall and a structure.
SAMPLES = [("task_12", "shard_00004.npz", 289), ("task_11", "shard_00003.npz", 461), ("task_7", "shard_00001.npz", 362)]


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(0)
    cols = ["1. Depth image $R$", "2. Edges and sampled points", "3. Vehicle boxes, ray cast ($D_M$)",
            "4. Depth moved closer ($D_{\\mathrm{offset}}$)", "5. Collision image $\\min(D_M, D_{\\mathrm{offset}})$"]
    fig, axes = plt.subplots(len(SAMPLES), len(cols), figsize=(3.1 * len(cols), 2.0 * len(SAMPLES) + 0.8), squeeze=False)
    kw = dict(cmap="viridis_r", vmin=0, vmax=ci.MAX_RANGE)
    for row, (task, shard, i) in zip(axes, SAMPLES):
        cam = yaml.safe_load(open(os.path.join(DATASET, task, "camera.yaml")))
        rays = ci.camera_rays(cam)
        R = np.load(os.path.join(DATASET, task, shard))["depth"][i] / 1000.0   # uint16 mm -> m
        e, (iy, ix), D_M, D_off, D_coll = ci.collision_image(R, rays, rng, ci.ROBOT_BOX)
        row[0].imshow(R, **kw)
        row[1].imshow(R, cmap="gray_r", vmin=0, vmax=ci.MAX_RANGE)
        row[1].scatter(ix, iy, s=1.5, c="#d62728", linewidths=0)
        row[2].imshow(D_M, **kw)
        row[3].imshow(D_off, **kw)
        im = row[4].imshow(D_coll, **kw)
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    for a, t in zip(axes[0], cols):
        a.set_title(t, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12, wspace=0.06, hspace=0.10)
    fig.colorbar(im, cax=fig.add_axes([0.25, 0.06, 0.5, 0.015]), orientation="horizontal", label="Range (m)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    w, h, front, back = ci.ROBOT_BOX
    print(out, "robot box (half w, half h, front, back):", ci.ROBOT_BOX, "edge jump", ci.JUMP,
          "slope ratio", ci.SLOPE_RATIO, "edge fraction", ci.EDGE_FRACTION)


if __name__ == "__main__":
    main()
