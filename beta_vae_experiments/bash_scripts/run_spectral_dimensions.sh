#!/usr/bin/env bash
# Full notebook G analysis, without training any VAE or controller.
set -euo pipefail
SPECTRAL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SPECTRAL_REPO"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vae_spectral_matplotlib"
.venv/bin/python -m beta_vae_experiments.result_analysis.spectral_dimension_analysis --z-draws 3
.venv/bin/python -m beta_vae_experiments.result_analysis.spectral_dimension_followup
