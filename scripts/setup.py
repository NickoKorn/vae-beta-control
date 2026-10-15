"""Create a local Python 3.11 environment and register the notebook kernel."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=ROOT / '.venv')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error('Use Python 3.11: python3.11 scripts/setup.py')
    environment = args.env.expanduser().resolve()
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    run(python, '-m', 'pip', 'install', '--upgrade', 'pip')
    run(python, '-m', 'pip', 'install', '-r', ROOT / 'beta_vae_experiments/requirements.txt')
    # This package declares an older SciPy ceiling. Our Forman path is tested
    # with the project's newer SciPy; dependencies are explicitly listed above.
    run(python, '-m', 'pip', 'install', '--no-deps', '-r', ROOT / 'beta_vae_experiments/requirements-graph.txt')
    run(python, '-m', 'ipykernel', 'install', '--sys-prefix',
        '--name', 'vae-beta-control', '--display-name', 'VAE beta control (Python 3.11)')
    run(python, '-c', 'from beta_vae_experiments.new_beta_varix_experiments import run_experiment; print("Training imports OK")')
    print(f'\nReady. Launch: "{python}" -m jupyterlab START_HERE.ipynb')


if __name__ == '__main__':
    main()
