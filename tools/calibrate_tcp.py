#!/usr/bin/env python3
"""Offline pivot calibration: fit a rigid TCP from recorded wrist poses.

No robot connection or motion is performed. Pivot data identifies translation
only; the TCP orientation must be supplied independently for each sampled arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x2_tcp import finite_array, validate_rotation, validate_tcp_config, read_tcp_json


def fit_pivot(samples: list[dict], *, max_rms_mm: float = 1.0,
              max_residual_mm: float = 2.0, max_condition: float = 1000.0) -> dict:
    """Fit ``R_i @ tcp_offset + wrist_position_i = fixed_pivot``.

    Require at least six samples, full rank and adequate orientation diversity;
    reject noisy/inconsistent data instead of silently writing a tool offset.
    """
    for name, value in (("max_rms_mm", max_rms_mm), ("max_residual_mm", max_residual_mm),
                        ("max_condition", max_condition)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(samples, list) or len(samples) < 6:
        raise ValueError("pivot calibration needs at least 6 wrist poses per arm")
    rotations, positions = [], []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {"position_m", "rotation_matrix"}:
            raise ValueError(f"sample {index} requires only position_m and rotation_matrix")
        positions.append(finite_array(sample["position_m"], (3,), f"sample {index}.position_m"))
        rotations.append(validate_rotation(sample["rotation_matrix"], f"sample {index}.rotation_matrix"))
    A = np.vstack([np.hstack((R, -np.eye(3))) for R in rotations])
    b = -np.concatenate(positions)
    solution, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    if rank != 6:
        raise ValueError("pivot samples are rank deficient: vary wrist orientation about multiple axes")
    condition = float(singular[0] / singular[-1])
    if condition > max_condition:
        raise ValueError(f"pivot samples are ill-conditioned ({condition:.3g}); increase orientation diversity")
    residuals = np.linalg.norm((A @ solution - b).reshape(-1, 3), axis=1) * 1000
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    worst = float(np.max(residuals))
    if rms > max_rms_mm or worst > max_residual_mm:
        raise ValueError(f"pivot residual too high: RMS={rms:.4f} mm, max={worst:.4f} mm "
                         f"(limits {max_rms_mm:g}/{max_residual_mm:g} mm)")
    return {"translation_m": solution[:3].tolist(), "pivot_position_m": solution[3:].tolist(),
            "rms_mm": rms, "max_residual_mm": worst, "condition_number": condition,
            "sample_count": len(samples)}


def calibrate_config(data: dict, *, base_config: dict | None = None,
                     max_rms_mm: float = 1.0, max_residual_mm: float = 2.0,
                     max_condition: float = 1000.0) -> tuple[dict, dict]:
    """Produce a loadable custom config and a separate diagnostic report.

    Input has schema_version, units, samples and tcp_rotation_matrix. samples
    and tcp_rotation_matrix map side names to measured poses and independent
    tool orientations. A one-arm calibration requires an existing base config
    for the untouched arm; no unmeasured zero transform is invented.
    """
    required = {"schema_version", "units", "samples", "tcp_rotation_matrix"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("samples file requires schema_version, units, samples and tcp_rotation_matrix")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["units"] != "m":
        raise ValueError("samples schema_version must be 1 and units must be 'm'")
    samples, orientations = data["samples"], data["tcp_rotation_matrix"]
    if not isinstance(samples, dict) or not samples or not set(samples) <= {"left", "right"}:
        raise ValueError("samples must contain left and/or right")
    if not isinstance(orientations, dict) or set(orientations) != set(samples):
        raise ValueError("tcp_rotation_matrix must explicitly specify each sampled arm; pivot cannot measure orientation")
    tools = {}
    if base_config is not None:
        tools = {side: {k: v for k, v in tool.items() if k != "mode"}
                 for side, tool in validate_tcp_config(base_config).items()}
    if set(samples) != {"left", "right"} and not tools:
        raise ValueError("one-arm calibration requires --base-file with the other arm's existing transform")
    report = {}
    for side, poses in samples.items():
        R = validate_rotation(orientations[side], f"{side}.tcp_rotation_matrix")
        fit = fit_pivot(poses, max_rms_mm=max_rms_mm, max_residual_mm=max_residual_mm,
                        max_condition=max_condition)
        tools[side] = {"name": f"{side}_pivot_calibrated_tool", "frame": f"{side}_wrist_roll_link",
                       "translation_m": fit["translation_m"], "rotation_matrix": R.tolist(),
                       "source": "offline pivot fit of measured wrist poses; rotation supplied independently"}
        report[side] = fit
    result = {"schema_version": 1, "units": "m", "tools": tools}
    validate_tcp_config(result)
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="recorded wrist pose JSON (metres, rotation matrices)")
    parser.add_argument("--output", required=True, help="new custom TCP JSON file (must not already exist)")
    parser.add_argument("--base-file", help="existing custom TCP config when calibrating one arm")
    parser.add_argument("--max-rms-mm", type=float, default=1.0)
    parser.add_argument("--max-residual-mm", type=float, default=2.0)
    parser.add_argument("--max-condition", type=float, default=1000.0)
    args = parser.parse_args(argv)
    try:
        config, report = calibrate_config(read_tcp_json(args.samples),
            base_config=read_tcp_json(args.base_file) if args.base_file else None,
            max_rms_mm=args.max_rms_mm, max_residual_mm=args.max_residual_mm,
            max_condition=args.max_condition)
        output = Path(args.output)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"TCP calibration rejected: {exc}\n")
    print(json.dumps({"output": str(output), "fit": report,
                      "note": "Translation fitted offline; orientation supplied separately. No robot motion performed."},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
