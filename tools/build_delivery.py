#!/usr/bin/env python3
"""Assemble an explicit customer handover; never copy site config, SDK or results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'x2ik-0.2.0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    desktop = ROOT / 'dist/X2-MDI-0.2.0-x86_64.AppImage'
    info = json.loads((ROOT / 'dist/APPIMAGE_BUILD_INFO.json').read_text())
    assert digest(desktop) == info['sha256'], 'AppImage hash differs from build record'
    assert digest(ROOT / 'desktop/x2_mdi_desktop.py') == info['desktop_source_sha256'], 'Rebuild desktop after source changes'
    sources = {p.name: digest(p) for p in (ROOT/'desktop').glob('*.py')}
    assert sources == info['desktop_sources_sha256'], 'Rebuild desktop after a view module changes'
    selected = {
        'docs/HANDOVER.md': 'README.md',
        'dist/x2ik-0.2.0-py3-none-any.whl': 'module/x2ik-0.2.0-py3-none-any.whl',
        'dist/x2ik-runtime-0.2.0.tar.gz': 'module/x2ik-runtime-0.2.0.tar.gz',
        'dist/X2-MDI-0.2.0-x86_64.AppImage': 'desktop/X2-MDI-0.2.0-x86_64.AppImage',
        'dist/APPIMAGE_BUILD_INFO.json': 'desktop/APPIMAGE_BUILD_INFO.json',
        'examples/movej_minimal.py': 'examples/movej_minimal.py',
        'examples/mdi_minimal.sh': 'examples/mdi_minimal.sh',
        'desktop/x2_mdi_desktop.py': 'source/desktop/x2_mdi_desktop.py',
        'tools/build_appimage.py': 'source/tools/build_appimage.py',
    }
    for path in sorted((ROOT/'desktop').glob('*.py')):
        selected[str(path.relative_to(ROOT))] = 'source/desktop/' + path.name
    for name in ('PROJECT_OVERVIEW.md', 'MODULE_GUIDE.md', 'DESKTOP_MDI_GUIDE.md', 'COORDINATES.md'):
        selected['packaging/share_docs/' + name] = 'docs/' + name
    with tempfile.TemporaryDirectory(prefix='x2-delivery-') as temporary:
        stage = Path(temporary) / NAME
        stage.mkdir()
        for source, destination in selected.items():
            target = stage/destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/source, target)
            if source == 'docs/HANDOVER.md':
                target.write_text(target.read_text().replace(
                    '(../packaging/share_docs/PROJECT_OVERVIEW.md)',
                    '(docs/PROJECT_OVERVIEW.md)'))
        (stage/'THIRD_PARTY_NOTICES.txt').write_text(
            'Desktop: PyQt5/Qt, Python and OpenSSH; distribution notices are included below\n'
            'and inside the AppImage usr/licenses directory. Desktop source and build script\n'
            'are provided in source/. Robot module requires an existing ROS/AimDK installation;\n'
            'ROS/AimDK SDK binaries, credentials, local configuration, calibration and logs\n'
            'are not included. URDF model upstream declares Mulan PSL v2.\n')
        for name, source in {
            'PyQt5': '/usr/share/doc/python3-pyqt5/copyright',
            'Qt': '/usr/share/doc/libqt5core5a/copyright',
            'OpenSSH': '/usr/share/doc/openssh-client/copyright',
            'Python': '/usr/share/doc/python3.10/copyright',
            'Model': str(ROOT/'x2_urdf_upstream_README.md'),
        }.items():
            if Path(source).is_file():
                target = stage/'licenses'/name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source, target)
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        (stage/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(stage)}\n' for p in files))
        destination = ROOT/'delivery'/NAME
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            # Only replace files belonging to an earlier generated delivery.
            previous = destination/'SHA256SUMS'
            if not previous.is_file():
                raise RuntimeError('Existing delivery has no manifest; refusing to replace it')
            for line in previous.read_text().splitlines():
                sha, relative = line.split(maxsplit=1)
                path = destination/relative
                if path.is_file() and digest(path) != sha:
                    raise RuntimeError(f'Locally edited handover file: {path}')
            for p in destination.rglob('*'):
                if p.is_file() and p.relative_to(destination).as_posix() not in {
                    line.split(maxsplit=1)[1] for line in previous.read_text().splitlines()
                } | {'SHA256SUMS'}:
                    raise RuntimeError(f'Untracked handover file: {p}')
            shutil.rmtree(destination)
        shutil.copytree(stage, destination)
        archive = ROOT/'delivery'/f'{NAME}.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            stream.add(stage, arcname=NAME)
        (ROOT/'delivery/SHA256SUMS').write_text(f'{digest(archive)}  {archive.name}\n')
        print(f'{archive}\nSHA256 {digest(archive)}\n{len(files)} files plus SHA256SUMS')


if __name__ == '__main__':
    main()
