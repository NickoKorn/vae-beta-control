#!/bin/zsh
set -e

export UV_CACHE_DIR=/tmp/autoencodix-uv-cache
export MPLCONFIGDIR=/tmp/autoencodix-matplotlib
export XDG_CACHE_HOME=/tmp/autoencodix-xdg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2

uv run python beta_vae_experiments/beta_varix_experiments.py \
  --skip-smoke-test \
  --epochs 50 \
  --loss-reduction sum \
  --yes
