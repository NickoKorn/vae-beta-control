#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${1:-${project_root}/Real_Data_Latent_d8_FixedBeta_AbsoluteCoreMetrics_v4_MPS/epochs_50/plots}"
target_root="${2:-${project_root}/Real_Data_Latent_d8_FixedBeta_AbsoluteCoreMetrics_v4_MPS/epochs_50/comparison_plots}"

if [[ ! -d "${source_root}" ]]; then
    echo "Plot source directory does not exist: ${source_root}" >&2
    exit 1
fi

mkdir -p "${target_root}"

manifest="${target_root}/plot_manifest.tsv"
temporary_manifest="${manifest}.tmp"
printf 'run_config\tplot_type\tsource_path\tcomparison_path\n' > "${temporary_manifest}"

collected=0
while IFS= read -r -d '' source_path; do
    run_config="$(basename "$(dirname "${source_path}")")"
    original_name="$(basename "${source_path}")"
    plot_type="${original_name%.png}"
    comparison_name="${run_config}__${original_name}"
    comparison_path="${target_root}/${comparison_name}"

    ln -sfn "${source_path}" "${comparison_path}"
    printf '%s\t%s\t%s\t%s\n' \
        "${run_config}" \
        "${plot_type}" \
        "${source_path}" \
        "${comparison_path}" >> "${temporary_manifest}"
    collected=$((collected + 1))
done < <(
    find "${source_root}" -mindepth 2 -maxdepth 2 -type f \
        \( \
            -name '2D-scatter_*.png' \
            -o -name 'Ridgeline_*.png' \
            -o -name 'loss_absolute.png' \
            -o -name 'loss_relative.png' \
        \) \
        -print0 \
        | sort -z
)

mv "${temporary_manifest}" "${manifest}"

echo "Collected ${collected} comparison plots."
echo "Target: ${target_root}"
echo "Manifest: ${manifest}"
