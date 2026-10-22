#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
CONFIG_DIR="${2:-${REPO}/beta_vae_experiments/new_results_2026-10-22/annealing_observed_design/configs}"
MODE="${1:---preflight}"
if [[ "${MODE}" != --preflight && "${MODE}" != --run ]]; then
    printf 'Usage: bash %s [--preflight|--run] [config_directory]\n' "$0" >&2
    exit 2
fi
export PYTHONHASHSEED=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/beta-vae-matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${REPO}"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA required for the large-scale suite"'
shopt -s nullglob
CONFIGS=("${CONFIG_DIR}"/*.json)
if (( ${#CONFIGS[@]} == 0 )); then
    printf 'No experiment configs found: %s\n' "${CONFIG_DIR}" >&2
    exit 1
fi
for CONFIG in "${CONFIGS[@]}"; do
    "${PYTHON}" -m beta_vae_experiments.new_beta_varix_experiments "${CONFIG}" --preflight
done
if [[ "${MODE}" == --preflight ]]; then
    exit 0
fi
for CONFIG in "${CONFIGS[@]}"; do
    "${PYTHON}" -m beta_vae_experiments.new_beta_varix_experiments "${CONFIG}"
done
