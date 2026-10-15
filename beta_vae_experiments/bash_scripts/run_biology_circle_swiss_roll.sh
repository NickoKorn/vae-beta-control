#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
BIOLOGY_CONFIG="${PROJECT_ROOT}/beta_vae_experiments/user_interface/experiments.json#training.tcga"
CIRCLE_CONFIG="${PROJECT_ROOT}/beta_vae_experiments/user_interface/experiments.json#training.circle"
SWISS_ROLL_CONFIG="${PROJECT_ROOT}/beta_vae_experiments/user_interface/experiments.json#training.swiss_roll"

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2
export MPLCONFIGDIR=/tmp/beta-vae-matplotlib
export XDG_CACHE_HOME=/tmp/beta-vae-xdg

if (( $# > 0 )); then
    CONFIGS=("$@")
else
    CONFIGS=("${BIOLOGY_CONFIG}" "${CIRCLE_CONFIG}" "${SWISS_ROLL_CONFIG}")
fi

on_interrupt() {
    printf '\nTraining sequence interrupted. The completed runs remain resumable.\n' >&2
    exit 130
}
trap on_interrupt INT TERM

if [[ ! -x "${PYTHON}" ]]; then
    printf 'Project Python environment not found: %s\n' "${PYTHON}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

"${PYTHON}" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this shell.")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
'

printf '\nValidating %d experiment JSON file(s) before training...\n' "${#CONFIGS[@]}"
for config in "${CONFIGS[@]}"; do
    "${PYTHON}" -m beta_vae_experiments.new_beta_varix_experiments \
        "${config}" --preflight
done

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    printf '\n===== %d/%d %s =====\n' \
        "$((index + 1))" "${#CONFIGS[@]}" "$(basename "${config}")"
    "${PYTHON}" -m beta_vae_experiments.new_beta_varix_experiments "${config}"
done

printf '\nAll experiments and metric analyses completed successfully.\n'
