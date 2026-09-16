#!/usr/bin/env python3
"""Export reviewed source only. No SDK, site config, results, keys or git push."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'x2_api.py', 'x2_arm_bridge.py', 'x2_arm_dynamics.py', 'x2_arm_model.py',
    'x2_converge_test.py', 'x2_frames.py', 'x2_mdi.py', 'x2_mdi_bridge.py',
    'x2_movej.py', 'x2_random_path_audit.py', 'x2_random_points.py',
    'x2_random_reach_plan.py', 'x2_random_reach_test.py', 'x2_record.py',
    'x2_sim_ros.py', 'x2_srs_batch.py', 'x2_srs_ik.py', 'x2_upper_raw.py',
    'x2ik.py', 'upper_body_control.py', 'verify_x2_arm.py',
)
# The two model assets already exist in the repository's v1.0 branch.
ASSETS = ('x2_ultra.urdf', 'x2_ultra.xml', 'x2_urdf_upstream_README.md',
          '.gitignore', '.gitattributes', 'config/x2ik.conf.example', 'THIRD_PARTY_NOTICES.md')
FORBIDDEN_PARTS = {'.git', '.ssh', '.codex', '.agents', 'aimdk', 'runtime',
                   'dist', 'delivery', 'results', 'calibration', 'backups',
                   'artifacts', '__pycache__', 'archive'}
SECRET_PATTERNS = (
    re.compile(rb'-----BEGIN (?:OPENSSH|RSA|DSA|EC|ENCRYPTED) PRIVATE KEY-----'),
    re.compile(rb'\b(?:glpat-|gh[pousr]_)[A-Za-z0-9_\-]{20,}\b'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
)


def source_files():
    selected = {name: ROOT/name for name in MODULES + ASSETS}
    for directory, suffixes in {
        'desktop': {'.py', '.md'}, 'examples': {'.py', '.sh'},
        'tests': {'.py'}, 'tools': {'.py'}, 'packaging': {'.py', '.md'},
    }.items():
        for path in sorted((ROOT/directory).rglob('*')):
            relative = path.relative_to(ROOT)
            if (path.is_file() and path.suffix in suffixes
                    and not set(relative.parts) & FORBIDDEN_PARTS):
                selected[relative.as_posix()] = path
    selected['README.md'] = ROOT/'packaging/repository/README.md'
    selected['CONTRIBUTING.md'] = ROOT/'packaging/repository/CONTRIBUTING.md'
    selected['docs/HANDOVER.md'] = ROOT/'docs/HANDOVER.md'
    for p in (ROOT/'packaging/share_docs').glob('*.md'):
        selected['docs/'+p.name] = p
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        ap.error('output must be absent or empty; existing files are never deleted')
    selected = source_files()
    payload = {}
    for name, path in selected.items():
        if path.is_symlink() or set(Path(name).parts) & FORBIDDEN_PARTS:
            raise RuntimeError('Disallowed export path: '+name)
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise RuntimeError('Credential pattern detected in '+name)
        payload[name] = (data, path.stat().st_mode & 0o777)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (data, mode) in sorted(payload.items()):
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
        manifest[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    print(json.dumps(dict(output=str(output), files=manifest), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
