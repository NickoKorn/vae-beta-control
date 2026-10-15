"""Read a plain JSON or a named profile in experiments.json; never start training.

Example: experiments.json#training.tcga . Dictionaries merge recursively; arrays
and scalar values replace defaults. Each read returns an independent dictionary.
"""
import argparse
import copy
import json
from pathlib import Path

CATALOG = Path(__file__).with_name('experiments.json')


def merge_settings(defaults, overrides):
    result=copy.deepcopy(defaults)
    for key,value in overrides.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=merge_settings(result[key],value)
        else:
            result[key]=copy.deepcopy(value)
    return result


def read_config(reference):
    """Return (expanded settings, original base directory for relative paths)."""
    filename,separator,profile=str(reference).partition('#')
    path=Path(filename).expanduser().resolve()
    with path.open(encoding='utf-8') as handle:
        document=json.load(handle)
    if not isinstance(document,dict):
        raise ValueError(f'Configuration must be a JSON object: {path}')
    if 'catalog_schema_version' not in document:
        if separator:
            raise ValueError('Named profiles require an experiment catalog')
        return document,path.parent
    if document['catalog_schema_version']!=1:
        raise ValueError('Unsupported experiment catalog schema')
    profile=profile if separator else document['default_profile']
    try:
        group_name,name=profile.split('.',1)
        if name not in document['groups'].get(group_name,{}).get('profiles',{}):
            archive=document.get('archive')
            if archive is None:
                raise KeyError(profile)
            path=(path.parent/archive).resolve()
            document=json.loads(path.read_text())
            if document.get('catalog_schema_version')!=1:
                raise ValueError('Unsupported archive schema')
        group=document['groups'][group_name]
        settings=merge_settings(group.get('defaults',{}),group['profiles'][name])
    except (KeyError,ValueError) as error:
        raise ValueError(f'Unknown profile {profile!r}; use --list to see profiles') from error
    directory=(path.parent/group.get('profile_directories',{}).get(name,'.')).resolve()
    return settings,directory


def load_config(reference):
    return read_config(reference)[0]


def list_profiles(path=CATALOG):
    path=Path(path)
    document=json.loads(path.read_text())
    names=[f'{group}.{name}' for group,entry in document['groups'].items() for name in entry['profiles']]
    if document.get('archive'):
        archive=json.loads((path.parent/document['archive']).read_text())
        names.extend(f'{group}.{name}' for group,entry in archive['groups'].items()
                     for name in entry['profiles'] if f'{group}.{name}' not in names)
    return names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',nargs='?',default=str(CATALOG))
    parser.add_argument('--list',action='store_true',help='List profiles without starting any work')
    parser.add_argument('--output',type=Path,help='Export the expanded settings as plain JSON')
    args=parser.parse_args()
    if args.list:
        if args.output:
            parser.error('--list and --output cannot be combined')
        print('\n'.join(list_profiles(str(args.config).partition('#')[0])))
        return
    settings,directory=read_config(args.config)
    if args.output and isinstance(settings.get('training'),dict):
        # A plain exported training config must keep the original output paths.
        for key in ('output_dir','results_root'):
            if settings.get(key) and not Path(settings[key]).is_absolute():
                settings[key]=str((directory/settings[key]).resolve())
    content=json.dumps(settings,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content)
    else:
        print(content,end='')


if __name__=='__main__':
    main()
