"""Read-only cohort audit of official AUTOENCODIX Parquet files; no training.

Run directly to avoid importing the training package:
    python beta_vae_experiments/datasets/audit_tcga_pancancer.py --data-root data/raw/TCGAPanCancerAutoencodix
Only IDs/clinical annotations are loaded, not the entire omics matrices.
Full value-level preprocessing and split audits still belong to AUTOENCODIX.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

FILES = {'clinical': 'data_clinical_formatted.parquet',
         'rna': 'data_mrna_seq_v2_rsem_formatted.parquet',
         'methylation': 'data_methylation_per_gene_formatted.parquet'}


def audit(data_root, output):
    root = Path(data_root)
    report = {'source': 'https://doi.org/10.5281/zenodo.15518831',
              'data_root': str(root.resolve()), 'files': {},
              'scope': 'Raw cohort/ID audit; not a preprocessing or train/test split validation'}
    ids, clinical = {}, None
    for label, name in FILES.items():
        path = root / name
        if not path.is_file():
            report['files'][label] = {'file': name, 'status': 'missing'}
            continue
        parquet = pq.ParquetFile(path)
        frame = pd.read_parquet(path) if label == 'clinical' else pd.read_parquet(path, columns=[])
        # Official AUTOENCODIX omics files retain an unnamed sample index;
        # clinical uses SAMPLE_ID. Validate actual IDs, do not rewrite raw data.
        valid_index_name = frame.index.name == 'SAMPLE_ID' or (
            label != 'clinical' and frame.index.name is None
            and frame.index.astype(str).str.startswith('TCGA-').all())
        if not valid_index_name or not frame.index.is_unique or frame.index.hasnans:
            raise ValueError(f'{name}: expected unique nonmissing TCGA sample IDs')
        ids[label] = frame.index.astype(str)
        report['files'][label] = {'file': name, 'status': 'present',
            'n_samples': len(frame), 'file_bytes': path.stat().st_size,
            'parquet_rows': parquet.metadata.num_rows, 'index_name': frame.index.name,
            'unique_sample_ids': True}
        if label == 'clinical':
            clinical = frame
            counts = frame['CANCER_TYPE_ACRONYM'].value_counts(dropna=False)
            report['clinical_class_counts'] = {str(k): int(v) for k, v in counts.items()}
            report['clinical_n_classes'] = int(frame['CANCER_TYPE_ACRONYM'].nunique())
            report['clinical_n_patients'] = int(frame['PATIENT_ID'].nunique())
    if len(ids) == len(FILES):
        common = ids['clinical'].intersection(ids['rna'], sort=False).intersection(ids['methylation'], sort=False)
        if len(common) == 0:
            raise ValueError('No common samples across clinical/RNA/methylation')
        report['n_common_samples'] = len(common)
        report['common_class_counts'] = {str(k): int(v) for k, v in clinical.loc[common, 'CANCER_TYPE_ACRONYM'].value_counts(dropna=False).items()}
        report['classes_lost_after_join'] = sorted(set(clinical['CANCER_TYPE_ACRONYM'].dropna()) - set(clinical.loc[common, 'CANCER_TYPE_ACRONYM'].dropna()))
        report['status'] = 'raw_cohort_audited'
    else:
        report['status'] = 'incomplete_download'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path,
        default=Path('beta_vae_experiments/outputs/tcga_pancancer_audit.json'))
    args = parser.parse_args()
    result = audit(args.data_root, args.output)
    raise SystemExit(0 if result['status'] == 'raw_cohort_audited' else 2)
