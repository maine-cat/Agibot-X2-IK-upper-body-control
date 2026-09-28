#!/usr/bin/env python3
"""Build wheel and standalone runtime from the same x2ik sources."""
import argparse
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

SOURCE = Path(__file__).resolve().parents[1]


def package_files():
    return sorted(p for p in (SOURCE / 'x2ik').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')


def version():
    tree = ast.parse((SOURCE / 'x2ik/__init__.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing x2ik.__version__')


def build(output):
    # setuptools/wheel are build dependencies only; NumPy remains the runtime dependency.
    import setuptools.build_meta as backend

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '1704067200'))
    previous_epoch = os.environ.get('SOURCE_DATE_EPOCH')
    os.environ['SOURCE_DATE_EPOCH'] = str(epoch)
    old_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix='x2ik-build-') as directory:
            work = Path(directory)
            runtime = work / 'runtime'
            runtime.mkdir()
            for src in package_files():
                dest = runtime / src.relative_to(SOURCE)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            shutil.copyfile(SOURCE / 'pyproject.toml', runtime / 'pyproject.toml')
            manifest = {p.relative_to(SOURCE / 'x2ik').as_posix():
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in package_files()}
            (runtime / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
            # Build in a temporary checkout; never leave build/egg-info in maintained sources.
            wheel_source = work / 'wheel-source'
            shutil.copytree(runtime, wheel_source)
            wheel_dir = work / 'wheel'
            wheel_dir.mkdir()
            os.chdir(wheel_source)
            wheel_name = backend.build_wheel(str(wheel_dir))
            os.chdir(old_cwd)
            runtime_name = f'x2ik-runtime-{version()}.tar.gz'
            archive = work / runtime_name
            with archive.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=epoch) as gz:
                with tarfile.open(fileobj=gz, mode='w') as tar:
                    for path in sorted(runtime.rglob('*')):
                        if not path.is_file():
                            continue
                        data = path.read_bytes()
                        item = tarfile.TarInfo(path.relative_to(work).as_posix())
                        item.size, item.mtime, item.mode = len(data), epoch, 0o644
                        tar.addfile(item, io.BytesIO(data))
            for src in (wheel_dir / wheel_name, archive):
                staged = output / (src.name + '.new')
                shutil.copyfile(src, staged)
                os.replace(staged, output / src.name)
            print(f'Built {wheel_name} and {runtime_name}')
    finally:
        os.chdir(old_cwd)
        if previous_epoch is None:
            os.environ.pop('SOURCE_DATE_EPOCH', None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = previous_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SOURCE.parent / 'module')
    args = parser.parse_args()
    build(args.output)


if __name__ == '__main__':
    main()
