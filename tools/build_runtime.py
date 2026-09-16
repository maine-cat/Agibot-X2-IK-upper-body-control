#!/usr/bin/env python3
"""Build a small, dependency-explicit robot module from reviewed source files."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["x2_movej", "x2_api", "x2_mdi", "x2_mdi_bridge", "x2_sim_ros", "x2_arm_model",
           "x2_arm_dynamics", "x2_frames", "x2_srs_ik", "x2_srs_batch", "x2_record"]


def package_imports(source):
    # Change actual imports only, preserving docstrings and the original source otherwise.
    lines = source.splitlines(keepends=True)
    edits = []
    modules = set(MODULES) | {"x2ik"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in modules:
            name = "_bootstrap" if node.module == "x2ik" else node.module
            old = "from " + node.module
            line = lines[node.lineno - 1]
            pos = line.index(old, node.col_offset)
            edits.append((node.lineno - 1, pos, pos + len(old), "from ." + name))
        elif isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name in modules:
            alias = node.names[0]
            name = "_bootstrap" if alias.name == "x2ik" else alias.name
            replacement = "from . import " + name
            if alias.asname or name != alias.name:
                replacement += " as " + (alias.asname or alias.name)
            edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, replacement))
    for line, start, end, replacement in sorted(edits, reverse=True):
        lines[line] = lines[line][:start] + replacement + lines[line][end:]
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    dest = ROOT / "runtime" / "x2ik"
    dest.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        (dest / (name + ".py")).write_text(package_imports((ROOT / (name + ".py")).read_text()))
    (dest / "_bootstrap.py").write_text(package_imports((ROOT / "x2ik.py").read_text()))
    shutil.copy2(ROOT / "packaging/runtime_init.py", dest / "__init__.py")
    shutil.copy2(ROOT / "packaging/runtime_main.py", dest / "__main__.py")
    for name in ("x2_ultra.urdf", "x2_ultra.xml"):
        shutil.copy2(ROOT / name, dest / name)
    files = sorted(p for p in dest.iterdir() if p.is_file())
    checks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / "runtime/MANIFEST.json").write_text(json.dumps(checks, indent=2) + "\n")
    (ROOT / "runtime/pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "x2ik"
version = "0.2.0"
description = "X2 MDI and MoveJ robot module"
requires-python = ">=3.10"
dependencies = ["numpy>=1.21"]
[project.scripts]
x2ik = "x2ik.__main__:main"
[tool.setuptools.package-data]
x2ik = ["*.urdf", "*.xml"]
''')
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / "x2ik-runtime-0.2.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted((ROOT / "runtime").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not any(x.endswith(".egg-info") for x in path.parts) and "build" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    print(archive, archive.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
