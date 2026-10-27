#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
CONFIG_DIR="${2:-${REPO}/beta_vae_experiments/large_scale_results/ready_2026_10_27}"
MODE="${1:---preflight}"
if [[ "${MODE}" != --preflight && "${MODE}" != --run ]]; then
    printf 'Usage: bash %s [--preflight|--run] [config_directory_or_suite_root]\n' "$0" >&2
    exit 2
fi
export PYTHONHASHSEED=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/beta-vae-matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${REPO}"
shopt -s nullglob
if [[ -d "${CONFIG_DIR}/trustworthiness/configs" && -d "${CONFIG_DIR}/cohesiveness/configs" ]]; then
    CONFIGS=("${CONFIG_DIR}/trustworthiness/configs/"*.json "${CONFIG_DIR}/cohesiveness/configs/"*.json)
elif [[ -d "${CONFIG_DIR}/configs" ]]; then
    CONFIGS=("${CONFIG_DIR}/configs/"*.json)
else
    CONFIGS=("${CONFIG_DIR}/"*.json)
fi
if (( ${#CONFIGS[@]} == 0 )); then
    printf 'No experiment configs found: %s\n' "${CONFIG_DIR}" >&2
    exit 1
fi
# Check both TCGA strategies first; keep every remaining dataset in the queue.
ORDERED=()
for CONFIG in "${CONFIGS[@]}"; do
    if [[ "${CONFIG##*/}" == *tcga_pancancer* ]]; then ORDERED+=("${CONFIG}"); fi
done
for CONFIG in "${CONFIGS[@]}"; do
    if [[ "${CONFIG##*/}" != *tcga_pancancer* ]]; then ORDERED+=("${CONFIG}"); fi
done
"${PYTHON}" - "${ORDERED[@]}" <<'PY'
import sys
from pathlib import Path
from beta_vae_experiments.new_beta_varix_experiments import preflight_experiment
for config in sys.argv[1:]:
    preflight_experiment(Path(config))
PY
if [[ "${MODE}" == --preflight ]]; then
    exit 0
fi
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA required for the large-scale suite"'
FAILURES=0
for CONFIG in "${ORDERED[@]}"; do
    if [[ -f "${CONFIG_DIR}/preflight_report.json" ]]; then
        "${PYTHON}" - "${REPO}" "${CONFIG_DIR}/preflight_report.json" <<'PYCODE'
import hashlib, json, sys
from pathlib import Path
repo, report = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
for key in ('source_hashes', 'config_hashes'):
    changed = [name for name, digest in report[key].items()
               if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise SystemExit('Checked code/config changed; recheck before continuing: ' + ', '.join(changed))
PYCODE
    fi
    if ! "${PYTHON}" -u -m beta_vae_experiments.new_beta_varix_experiments "${CONFIG}"; then
        printf 'Configuration failed; continuing suite: %s\n' "${CONFIG}" >&2
        FAILURES=$((FAILURES + 1))
    fi
done
if (( FAILURES > 0 )); then
    printf '%s configurations contain failures; inspect experiment_metrics.csv and the suite log.\n' "${FAILURES}" >&2
    exit 1
fi
