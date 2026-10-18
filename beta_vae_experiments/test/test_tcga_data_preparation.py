"""Standalone data preparation checks without importing AUTOENCODIX."""
import importlib.util
from pathlib import Path
import json
from types import SimpleNamespace
import pandas as pd
import pytest


def load(name):
    path = Path('beta_vae_experiments/datasets') / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_checks_capacity_before_network(tmp_path, monkeypatch):
    module = load('fetch_tcga_pancancer')
    manifest = tmp_path/'source.json'
    manifest.write_text(json.dumps({'members': [dict(name='rna.parquet', size=100,
        crc=0, offset=0, end_offset=99)]}))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(module, 'urlopen', lambda *_a, **_k: pytest.fail('network before capacity check'))
    with pytest.raises(OSError, match='Insufficient disk space'):
        module.fetch(manifest, tmp_path)


def test_audit_reports_classes_lost_in_modality_intersection(tmp_path):
    module = load('audit_tcga_pancancer')
    clinical = pd.DataFrame({'CANCER_TYPE_ACRONYM': ['A','B','C'],
        'PATIENT_ID':['p1','p2','p3']}, index=pd.Index(['s1','s2','s3'], name='SAMPLE_ID'))
    clinical.to_parquet(tmp_path/module.FILES['clinical'])
    for label in ('rna','methylation'):
        pd.DataFrame({'gene':[1.,2.]}, index=pd.Index(['s1','s2'],name='SAMPLE_ID')).to_parquet(tmp_path/module.FILES[label])
    result = module.audit(tmp_path, tmp_path/'audit.json')
    assert result['n_common_samples'] == 2
    assert result['classes_lost_after_join'] == ['C']
    assert result['common_class_counts'] == {'A':1, 'B':1}
