#!/bin/bash
# Render the depth-encoder figures of the report from the trained encoder and the dataset.
# Runs both scripts in the Isaac Sim container (torch, numpy, matplotlib), with the uuv-rl-env
# repo bound to /work and this scripts/ folder to /scripts. Writes into ../figures/.
#
#   sbatch scripts/render_figures.slurm          (from the report repo root)
#
# Inputs (uuv-rl-env, worktree "environment"):
#   experiments/datasets/D001_mix_r3x6_k2           dataset and collision-image caches
#   experiments/encoders/dce/dce_D001_box_25903576  final encoder (best.pt)
#SBATCH --job-name=report_figures
#SBATCH --partition=CPUQ
#SBATCH --account=share-iv-mtp
#SBATCH --time=00:20:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/work/%u/report_render/%x_%j.out
set -euo pipefail
REPORT=${REPORT:-${SLURM_SUBMIT_DIR}}
REPO=${REPO:-/cluster/home/${USER}/uuv-sim/uuv-rl-env-environment}
SIF=${SIF:-/cluster/work/${USER}/containers/isaac-sim-5.1-marinegym.sif}
RUN=experiments/encoders/dce/dce_D001_box_25903576
SAMPLES="3307 2723 8261 9409 317 849"   # validation samples shown in the reconstruction figure
mkdir -p /cluster/work/${USER}/report_render

run() {
  apptainer exec --cleanenv --writable-tmpfs --home "/cluster/work/${USER}/isaac_cache/home" \
    --bind "${REPO}:/work" --bind "${REPORT}/scripts:/scripts" --bind "${REPORT}/figures:/figures" \
    --pwd /work "${SIF}" bash -c 'exec "${ISAAC_PY}" "$@"' _ "$@"
}

run /scripts/render_dce_recon.py "${RUN}" /figures/dce_recon.png ${SAMPLES}
run /scripts/render_collision_pipeline.py /figures/collision_pipeline.png
echo "Done: ${REPORT}/figures/dce_recon.png, ${REPORT}/figures/collision_pipeline.png"
