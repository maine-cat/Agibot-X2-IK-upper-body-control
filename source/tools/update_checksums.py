#!/usr/bin/env python3
"""Refresh SHA256SUMS after rebuilding a delivery."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = {'__pycache__', '.git', '.venv', 'build', 'dist'}
    rows = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or path.name == 'SHA256SUMS'
                or path.suffix in ('.pyc', '.pyo') or excluded.intersection(relative.parts)
                or any(part.endswith('.egg-info') for part in relative.parts)):
            continue
        rows.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative.as_posix()}\n')
    (root / 'SHA256SUMS').write_text(''.join(rows))
    print(f'Updated SHA256SUMS: {len(rows)} files')


if __name__ == '__main__':
    main()
