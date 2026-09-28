"""Validated, local wrist-to-tool transforms for MDI and MoveJ.

All transforms map TCP coordinates into ``{side}_wrist_roll_link``. A TCP
changes Cartesian pose interpretation; it never changes a MoveJ joint target.
This module has no robot, DDS, network or motion dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

TCP_MODES = ("none", "hand", "gripper", "custom")
SIDES = ("left", "right")
PRESETS_PATH = Path(__file__).resolve().parent / "assets" / "tcp_presets.json"
MAX_CONFIG_BYTES = 64 * 1024


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_tcp_json(path: str | Path) -> dict:
    """Read a bounded JSON object; reject duplicates and non-JSON numbers."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("TCP configuration exceeds 64 KiB")
    result = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                        parse_constant=lambda value: _reject_constant(value))
    if not isinstance(result, dict):
        raise ValueError("TCP configuration must be a JSON object")
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Validate numeric JSON/NumPy data without accepting strings or booleans."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have shape {shape}") from exc
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain numbers with shape {shape}")
    # NumPy promotes mixed bool/float lists to float; reject bools before casting.
    def contains_bool(item):
        if isinstance(item, (bool, np.bool_)):
            return True
        if isinstance(item, (list, tuple, np.ndarray)):
            return any(contains_bool(part) for part in item)
        return False
    if contains_bool(value):
        raise ValueError(f"{label} must not contain booleans")
    result = raw.astype(float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite numbers")
    return result


def validate_rotation(value: Any, label: str = "rotation_matrix") -> np.ndarray:
    """Reject reflections, scaled/sheared matrices and invalid rotations."""
    result = finite_array(value, (3, 3), label)
    if not np.allclose(result.T @ result, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError(f"{label} must be orthonormal (R.T @ R = I)")
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"{label} must have determinant +1")
    return result


def _keys(value: dict, required: set[str], optional: set[str], label: str):
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{label}: missing fields {sorted(missing)}; unknown fields {sorted(unknown)}")


def validate_tcp_config(config: dict, mode: str = "custom") -> dict[str, dict]:
    """Return independent, JSON-serializable left/right tool dictionaries.

    Schema: ``{schema_version: 1, units: 'm', tools: {left: ..., right: ...}}``.
    Each tool requires ``name``, ``frame``, ``translation_m``, ``rotation_matrix``;
    Optional ``source``, ``estimated`` and ``description`` describe provenance
    and whether the point is nominal/estimated. Full proper SE(3) is mandatory.
    """
    if mode not in TCP_MODES:
        raise ValueError(f"TCP mode must be one of {', '.join(TCP_MODES)}")
    if not isinstance(config, dict):
        raise ValueError("TCP configuration must be an object")
    _keys(config, {"schema_version", "units", "tools"}, set(), "TCP configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("TCP schema_version must be integer 1")
    if config["units"] != "m":
        raise ValueError("TCP units must be 'm'; translations are metres")
    tools = config["tools"]
    if not isinstance(tools, dict) or set(tools) != set(SIDES):
        raise ValueError("TCP tools must contain exactly left and right")
    result = {}
    for side in SIDES:
        tool = tools[side]
        if not isinstance(tool, dict):
            raise ValueError(f"TCP {side} must be an object")
        _keys(tool, {"name", "frame", "translation_m", "rotation_matrix"}, {"source", "estimated", "description"}, side)
        if not isinstance(tool["name"], str) or not tool["name"].strip() or len(tool["name"]) > 200:
            raise ValueError(f"TCP {side}.name must be a nonempty string up to 200 characters")
        if tool["frame"] != f"{side}_wrist_roll_link":
            raise ValueError(f"TCP {side}.frame must be {side}_wrist_roll_link")
        source = tool.get("source", "custom measured wrist-to-tool transform")
        if not isinstance(source, str) or not source.strip() or len(source) > 2000:
            raise ValueError(f"TCP {side}.source must be a nonempty string up to 2000 characters")
        estimated = tool.get("estimated", False)
        if type(estimated) is not bool:
            raise ValueError(f"TCP {side}.estimated must be boolean")
        description = tool.get("description", "Custom wrist-to-tool transform; verify independently before motion.")
        if not isinstance(description, str) or not description.strip() or len(description) > 2000:
            raise ValueError(f"TCP {side}.description must be a nonempty string up to 2000 characters")
        result[side] = {
            "mode": mode, "name": tool["name"], "frame": tool["frame"],
            "translation_m": finite_array(tool["translation_m"], (3,), f"{side}.translation_m").tolist(),
            "rotation_matrix": validate_rotation(tool["rotation_matrix"], f"{side}.rotation_matrix").tolist(),
            "source": source, "estimated": estimated, "description": description,
        }
    return result


def load_tcp_tools(mode: str = "none", tcp_file: str | Path | None = None,
                   tcp_config: dict | None = None) -> dict[str, dict]:
    """Load one of four modes, or validate an in-process custom JSON object.

    ``tcp_file`` is local to the calling process. The desktop MDI selects an
    absolute robot-side path and passes it to the SSH bridge at startup; it does
    not upload a desktop file or send JSON over the control protocol. ``tcp_config``
    is only an offline/in-process validation helper, not a bridge operation.
    """
    if mode not in TCP_MODES:
        raise ValueError(f"TCP mode must be one of {', '.join(TCP_MODES)}")
    if tcp_file is not None and tcp_config is not None:
        raise ValueError("tcp_file and tcp_config are mutually exclusive")
    if mode == "custom":
        if tcp_file is None and tcp_config is None:
            raise ValueError("custom TCP requires tcp_file or tcp_config")
        return validate_tcp_config(read_tcp_json(tcp_file) if tcp_file is not None else tcp_config)
    if tcp_file is not None or tcp_config is not None:
        raise ValueError("tcp_file/tcp_config is only accepted with custom TCP mode")
    presets = read_tcp_json(PRESETS_PATH)
    return validate_tcp_config({"schema_version": presets["schema_version"], "units": presets["units"],
                                "tools": presets["presets"][mode]}, mode=mode)


def wrist_to_tcp(position_m, rotation_matrix, tool: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compose a base-to-wrist pose with an already validated tool transform."""
    p = np.asarray(position_m, dtype=float)
    R = np.asarray(rotation_matrix, dtype=float)
    return p + R @ np.asarray(tool["translation_m"]), R @ np.asarray(tool["rotation_matrix"])


def tcp_to_wrist(position_m, rotation_matrix, tool: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert a base-to-TCP target to the wrist pose expected by arm IK."""
    R = np.asarray(rotation_matrix, dtype=float) @ np.asarray(tool["rotation_matrix"]).T
    return np.asarray(position_m, dtype=float) - R @ np.asarray(tool["translation_m"]), R
