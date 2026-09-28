#!/usr/bin/env python3
"""Verify package contents and run offline tests from installed wheel and runtime."""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
from zipfile import ZipFile

from build_module import SOURCE, package_files, version


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(command, *, cwd, env, input=None):
    result = subprocess.run(command, cwd=cwd, env=env, input=input,
                            text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'{command}\n{result.stdout}\n{result.stderr}')
    return result.stdout + result.stderr


def verify(module_dir):
    import numpy

    release = version()
    wheel = module_dir / f'x2ik-{release}-py3-none-any.whl'
    runtime = module_dir / f'x2ik-runtime-{release}.tar.gz'
    expected = {p.relative_to(SOURCE).as_posix(): p.read_bytes() for p in package_files()}
    report = dict(version=release, python=platform.python_version(),
                  scope='Offline mathematics, application logic and packaging; no ROS or hardware connection',
                  artifacts={p.name: digest(p.read_bytes()) for p in (wheel, runtime)},
                  source_sha256={name: digest(data) for name, data in expected.items()}, checks={})
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        actual = {n: archive.read(n) for n in names if n.startswith('x2ik/')}
        if actual != expected:
            raise ValueError('Wheel package differs from maintained source')
        record = f'x2ik-{release}.dist-info/RECORD'
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        if sorted(row[0] for row in rows) != sorted(names):
            raise ValueError('Wheel RECORD is incomplete')
        for name, checksum, size in rows:
            if name == record:
                if checksum or size:
                    raise ValueError('RECORD must not hash itself')
                continue
            data = archive.read(name)
            value = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            if checksum != 'sha256=' + value or int(size) != len(data):
                raise ValueError(f'Invalid wheel RECORD entry: {name}')
        metadata = archive.read(f'x2ik-{release}.dist-info/METADATA').decode()
        if f'Version: {release}\n' not in metadata:
            raise ValueError('Wheel version mismatch')
    report['checks']['wheel_record_and_source'] = True

    with tempfile.TemporaryDirectory(prefix='x2ik-release-check-') as directory:
        work = Path(directory)
        with tarfile.open(runtime) as archive:
            members = archive.getmembers()
            expected_names = {'runtime/' + n for n in expected} | {'runtime/pyproject.toml', 'runtime/MANIFEST.json'}
            if {m.name for m in members} != expected_names or len(members) != len(expected_names):
                raise ValueError('Unexpected runtime members')
            for member in members:
                if not member.isfile():
                    raise ValueError('Runtime must contain regular files only')
                dest = work / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(member).read())
        for name, data in expected.items():
            if (work / 'runtime' / name).read_bytes() != data:
                raise ValueError(f'Runtime differs from source: {name}')
        manifest = json.loads((work / 'runtime/MANIFEST.json').read_text())
        if manifest != {name.removeprefix('x2ik/'): digest(data) for name, data in expected.items()}:
            raise ValueError('Runtime MANIFEST mismatch')
        if (work / 'runtime/pyproject.toml').read_bytes() != (SOURCE / 'pyproject.toml').read_bytes():
            raise ValueError('Runtime build metadata differs from source')
        report['checks']['runtime_manifest_and_source'] = True
        env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        env.pop('PYTHONPATH', None)
        installed = work / 'installed'
        run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
             '--target', str(installed), str(wheel)], cwd=work, env=env)
        for label, location in (('wheel', installed), ('runtime', work / 'runtime')):
            # Embedded Python may supply NumPy through a separate site directory.
            numpy_site = str(Path(numpy.__file__).resolve().parent.parent)
            test_env = {**env, 'PYTHONPATH': str(location) + os.pathsep + numpy_site}
            probe = run([sys.executable, '-c',
                         'import json, x2ik, numpy; print(json.dumps(dict(version=x2ik.__version__, file=x2ik.__file__, numpy=numpy.__version__)))'],
                        cwd=work, env=test_env)
            info = json.loads(probe)
            if info['version'] != release or not Path(info['file']).is_relative_to(location):
                raise ValueError('Tests imported another package')
            report.setdefault('numpy', info['numpy'])
            output = run([sys.executable, '-m', 'unittest', 'discover', '-s', str(SOURCE / 'tests'), '-v'],
                         cwd=work, env=test_env)
            report['checks'][label + '_tests'] = output
            for args in (['--help'], ['movej'], ['movej', '--tcp-mode', 'gripper'],
                         ['mdi', '--stdio', '--demo']):
                run([sys.executable, '-m', 'x2ik', *args], cwd=work, env=test_env, input='')
            report['checks'][label + '_offline_cli'] = True
            print(f'{label}: source hashes, tests and offline CLI passed', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-dir', type=Path, default=SOURCE.parent / 'module')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file to preserve previous results')
    report = verify(args.module_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
