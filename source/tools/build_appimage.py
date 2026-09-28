#!/usr/bin/env python3
"""Build a native desktop AppImage, with Python/Qt and an SSH executable.

Prerequisites: PyInstaller, PyQt5, mksquashfs and official appimagetool AppImage.
For reproducible releases record the tool hashes printed into BUILD_INFO.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    subprocess.run([str(v) for v in args], check=True, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--appimagetool', required=True, type=Path)
    ap.add_argument('--pyinstaller-path', type=Path, help='Optional pip --target directory')
    args = ap.parse_args()
    if platform.machine() != 'x86_64':
        ap.error('Build this Linux x86_64 artifact on x86_64')
    tool = args.appimagetool.resolve()
    env = dict(os.environ)
    # Keep unrelated user-site backports out of PyInstaller's analysis.
    env['PYTHONNOUSERSITE'] = '1'
    if args.pyinstaller_path:
        env['PYTHONPATH'] = str(args.pyinstaller_path.resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='x2-mdi-build-') as work:
        work = Path(work)
        env['PYINSTALLER_CONFIG_DIR'] = str(work/'pyinstaller-cache')
        run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
             '--name', 'x2-mdi-desktop', '--distpath', work/'frozen', '--workpath', work/'work',
             '--specpath', work, '--add-binary', '/usr/bin/ssh:bin',
             '--paths', ROOT, '--add-data', str(ROOT/'assets') + ':assets',
             '--add-data', str(ROOT/'desktop/assets') + ':assets',
             ROOT/'desktop/x2_mdi_desktop.py'], env=env)
        appdir = work/'X2-MDI.AppDir'
        shutil.copytree(work/'frozen/x2-mdi-desktop', appdir/'usr')
        (appdir/'AppRun').write_text('''#!/bin/sh
APPDIR_LOCAL=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="$APPDIR_LOCAL/usr/_internal/bin:$PATH"
exec "$APPDIR_LOCAL/usr/x2-mdi-desktop" "$@"
''')
        (appdir/'AppRun').chmod(0o755)
        (appdir/'x2-mdi.desktop').write_text('''[Desktop Entry]
Type=Application
Name=X2 MDI Desktop
Comment=SSH desktop console for X2 arms
Exec=x2-mdi-desktop
Icon=x2-mdi
Terminal=false
Categories=Development;Engineering;
''')
        (appdir/'x2-mdi.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" rx="52" fill="#132338"/><path d="M72 62L40 132L92 194M184 62L216 132L164 194" fill="none" stroke="#44cbb2" stroke-width="18" stroke-linecap="round"/><path d="M103 69H153V151H103Z" fill="#dde9f5"/><circle cx="92" cy="194" r="15" fill="#ffcb70"/><circle cx="164" cy="194" r="15" fill="#ffcb70"/></svg>''')
        (appdir/'usr/licenses').mkdir()
        for name, path in {'Qt-PyQt5': '/usr/share/doc/python3-pyqt5/copyright',
                           'OpenSSH': '/usr/share/doc/openssh-client/copyright',
                           'NumPy': '/usr/share/doc/python3-numpy/copyright',
                           'Model-provenance.md': ROOT/'desktop/assets/README.md',
                           'Third-party-notices.md': ROOT/'THIRD_PARTY_NOTICES.md'}.items():
            if Path(path).is_file():
                shutil.copy2(path, appdir/'usr/licenses'/name)
        # Reuse the utility's official type-2 runtime; no hidden network downloads.
        offset = int(subprocess.check_output([str(tool), '--appimage-offset'], text=True).strip())
        runtime = work/'runtime-x86_64'
        runtime.write_bytes(tool.read_bytes()[:offset])
        run([tool, '--appimage-extract'], cwd=work, stdout=subprocess.DEVNULL)
        target = output/'X2-MDI-2.1.0-x86_64.AppImage'
        staged_target = work/target.name
        build_env = {**env, 'ARCH':'x86_64'}
        run([work/'squashfs-root/AppRun', '--no-appstream', '--runtime-file', runtime,
             '--mksquashfs-opt', '-processors', '--mksquashfs-opt', '2', appdir, staged_target], env=build_env)
        staged_target.chmod(0o755)
        # Install atomically even when the preceding desktop is still running.
        staged_output = output/(target.name + '.new')
        shutil.copy2(staged_target, staged_output)
        os.replace(staged_output, target)
        info = dict(artifact=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    architecture=platform.machine(), libc=platform.libc_ver(),
                    appimagetool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
                    desktop_source_sha256=hashlib.sha256((ROOT/'desktop/x2_mdi_desktop.py').read_bytes()).hexdigest())
        info['desktop_sources_sha256'] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'desktop').glob('*.py'))
        }
        info['support_files_sha256'] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'x2_tcp.py', ROOT/'assets/tcp_presets.json', ROOT/'THIRD_PARTY_NOTICES.md',
                      *sorted((ROOT/'desktop/assets').glob('*'))] if p.is_file()
        }
        (output/'APPIMAGE_BUILD_INFO.json').write_text(json.dumps(info, indent=2)+'\n')
        print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
