"""Copy the prepared experiment bundle and source snapshot to the user's drive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def publish(destination, design, smoke=None):
    destination, design = Path(destination), Path(design)
    if not destination.parent.is_mount():
        raise FileNotFoundError(f'External drive/path not mounted: {destination.parent}')
    destination.mkdir(exist_ok=True)
    bundle = destination / 'annealing_recovery_2026_10_14'
    bundle.mkdir(exist_ok=True)
    shutil.copytree(design, bundle / 'design', dirs_exist_ok=True)
    if smoke and Path(smoke).is_dir():
        shutil.copytree(smoke, bundle / 'cpu_smoke_test', dirs_exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    source_roots = [('vae-beta-control', project / 'beta_vae_experiments'),
                    ('autoencodix_package', project.parent / 'autoencodix_package/src')]
    manifest = []
    with zipfile.ZipFile(bundle / 'source_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, root in source_roots:
            for path in root.rglob('*'):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                if any(part.startswith(('results', 'new_results', '__pycache__', '.')) for part in relative.parts):
                    continue
                if path.suffix not in {'.py', '.json', '.md', '.sh', '.txt', '.bib', '.ipynb'}:
                    continue
                content = path.read_bytes()
                target = f'{name}/{root.name}/{relative}'
                archive.writestr(target, content)
                manifest.append(dict(file=target, sha256=hashlib.sha256(content).hexdigest(), size=len(content)))
        for filename in ['pyproject.toml', 'uv.lock', 'README.md']:
            path = project / filename
            if path.exists():
                archive.write(path, f'vae-beta-control/{filename}')
    (bundle / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Saved experiment bundle:', bundle)
    print('Future run outputs:', destination / 'results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    publish(args.destination, args.design, args.smoke)
