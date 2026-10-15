"""Central profiles preserve complete settings, isolation and path semantics."""
import json
from pathlib import Path
import pytest
from beta_vae_experiments.user_interface.config import read_config, load_config, list_profiles, merge_settings, CATALOG


def test_nested_defaults_replace_arrays_without_mutating_inputs():
    defaults={'training':{'seeds':[1,2],'epochs':100},'metrics':{'enabled':True}}
    overrides={'training':{'seeds':[3]}}
    result=merge_settings(defaults,overrides)
    assert result=={'training':{'seeds':[3],'epochs':100},'metrics':{'enabled':True}}
    result['training']['seeds'].append(4)
    assert defaults['training']['seeds']==[1,2] and overrides['training']['seeds']==[3]


def test_profile_reads_are_independent_and_unknown_profile_is_clear():
    reference=f'{CATALOG}#training.circle'
    original=load_config(reference)
    changed=load_config(reference);changed['training']['seeds'].clear()
    assert load_config(reference)==original
    with pytest.raises(ValueError,match='Unknown profile'):
        read_config(f'{CATALOG}#training.missing')
    assert len(list_profiles())==23
    assert load_config(CATALOG)==load_config(f'{CATALOG}#training.tcga')


def test_plain_json_still_supported_and_fragments_are_not_ignored(tmp_path):
    file=tmp_path/'custom.json';file.write_text('{"training":{"epochs":12}}')
    settings,directory=read_config(file)
    assert settings=={'training':{'epochs':12}} and directory==tmp_path
    with pytest.raises(ValueError,match='Named profiles'):
        read_config(f'{file}#not_a_profile')


def test_catalog_retains_smoke_relative_directory():
    _,directory=read_config(f'{CATALOG}#smoke.tcga_graph')
    assert directory==CATALOG.parent/'smoke_downstream_graph'
