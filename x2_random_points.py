#!/usr/bin/env python3
"""纯离线、固定 HOME 腕部姿态的前方空间随机 IK 覆盖；不连接 ROS 或发送运动。

位置从 torso 空间网格独立均匀采样，不用 FK 生成已知可达目标，不重抽失败点。
左右使用相同随机位置的 Y 镜像，各自使用本侧模型和 HOME wrist_roll_link 旋转。
该旋转只是当前手系假设：没有手掌外参，不能据此声称真实手掌朝向或外部 TCP 精度。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from x2_arm_model import ArmModel, DEFAULT_URDF, log3, matrix_to_rpy
from x2_frames import HOME_Q
from x2_srs_ik import SrsArmIK

DEFAULT_BOUNDS = ((.08, .30), (.18, .36), (-.10, .10))  # x, abs(y), z；m
DEFAULT_CELLS = (2, 2, 2)
DEFAULT_SEED = 20260915
DEFAULT_SAMPLES_PER_CELL = 32
DEFAULT_POS_TOL = .0005
DEFAULT_ROT_TOL = math.radians(.5)
DEFAULT_MIN_MARGIN = math.radians(3.)
HERE = Path(__file__).resolve().parent
ORIENTATION_ASSUMPTION = (
    "固定各侧 HOME 的 wrist_roll_link 旋转，relax=0；"
    "暂无手掌外参，此约定不代表已确认的真实手掌朝向。"
)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return int(value)


def _sampling_config(samples_per_cell, seed, bounds, cells):
    count = _positive_int(samples_per_cell, "samples_per_cell")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    box = np.asarray(bounds, float)
    if box.shape != (3, 2) or not np.all(np.isfinite(box)) or np.any(box[:, 0] >= box[:, 1]):
        raise ValueError("bounds 必须为 x / abs(y) / z 的三组有限递增区间")
    if box[0, 0] <= 0 or box[1, 0] <= 0:
        raise ValueError("前方区域要求 x 和 abs(y) 的下限都大于 0")
    if len(cells) != 3:
        raise ValueError("cells 必须为 x / abs(y) / z 三个正整数")
    grid = tuple(_positive_int(value, "cells") for value in cells)
    return count, int(seed), box, grid


def generate_samples(side, samples_per_cell=DEFAULT_SAMPLES_PER_CELL, seed=DEFAULT_SEED,
                     bounds=DEFAULT_BOUNDS, cells=DEFAULT_CELLS):
    """返回 JSON 可序列化的全部原始目标；同 seed、cell、index 的左右位置严格镜像。

    每个 cell 使用 SeedSequence([seed, ix, iy, iz]) 独立随机流。改变每格样本数
    不改变该格已有前缀；side 不参与随机流，方便比较同一镜像位置的左右求解结果。
    唯一使用 FK 生成的量是固定 HOME 旋转，随机目标位置不由 FK 生成。
    """
    if side not in ("left", "right"):
        raise ValueError("side 必须为 left 或 right")
    count, seed, box, grid = _sampling_config(samples_per_cell, seed, bounds, cells)
    _, rotation = ArmModel(side).forward_kinematics(HOME_Q)
    edges = [np.linspace(*box[axis], grid[axis] + 1) for axis in range(3)]
    samples = []
    for cell in product(*(range(n) for n in grid)):
        low = np.array([edges[axis][index] for axis, index in enumerate(cell)])
        high = np.array([edges[axis][index + 1] for axis, index in enumerate(cell)])
        rng = np.random.default_rng(np.random.SeedSequence([seed, *cell]))
        positions = rng.uniform(low, high, size=(count, 3))
        if side == "right":
            positions[:, 1] *= -1
        for index, pos in enumerate(positions):
            samples.append(dict(id=f"{side}-{'-'.join(map(str, cell))}-{index:04d}",
                                side=side, cell=list(cell), sample_index=index, seed=seed,
                                pos=pos.tolist(), rot=rotation.tolist()))
    return samples


def _timings(records):
    values = np.array([row["solve_ms"] for row in records], float)
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))} if len(values) else {
                "p50": None, "p95": None, "max": None}


def _summary(records):
    total = len(records)
    counts = {key: sum(bool(row[key]) for row in records)
              for key in ("ik_solved", "pose_pass", "margin_pass", "accepted")}
    return dict(total=total, **counts,
                success_rate=counts["accepted"] / total if total else None,
                pose_success_rate=counts["pose_pass"] / total if total else None,
                ik_solution_rate=counts["ik_solved"] / total if total else None,
                reason_counts=dict(Counter(row["reason"] for row in records)),
                solve_ms=_timings(records))


def _validate_sample(sample, models):
    side = sample["side"]
    if side not in models:
        raise ValueError(f"样本 side={side!r} 无效")
    pos, rot = np.asarray(sample["pos"], float), np.asarray(sample["rot"], float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        raise ValueError(f"样本 {sample.get('id')} 的 pos 必须为三个有限数")
    _, fixed = models[side].forward_kinematics(HOME_Q)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)) or not np.allclose(rot, fixed, atol=1e-12, rtol=0):
        raise ValueError(f"样本 {sample.get('id')} 的 rot 必须为本侧固定 HOME 旋转")
    cell = sample["cell"]
    if len(cell) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in cell):
        raise ValueError("样本 cell 必须为三个非负整数")
    return pos.copy(), rot.copy()


def evaluate_samples(samples, *, pos_tol=DEFAULT_POS_TOL, rot_tol=DEFAULT_ROT_TOL,
                     min_margin=DEFAULT_MIN_MARGIN, progress=None):
    """逐个独立求解，返回含所有成功/失败目标的报告，不做工作空间投影或姿态放宽。

    每次 solve 都传入新的 HOME_Q 副本。误差重新通过本侧 FK 计算，不直接相信
    solver 的误差字段；3° 限位余量是单独的执行候选门槛，不叫做 IK 无解。
    progress(done, total, last_record) 可用于 CLI 提示；本函数完全离线。
    """
    for name, value, allow_zero in (("pos_tol", pos_tol, False), ("rot_tol", rot_tol, False),
                                    ("min_margin", min_margin, True)):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} 必须为有限{'非负' if allow_zero else '正'}数")
    source = list(samples)
    if not source:
        raise ValueError("samples 不能为空")
    sides = sorted({sample["side"] for sample in source})
    models = {side: ArmModel(side) for side in sides}
    solvers = {side: SrsArmIK(model) for side, model in models.items()}
    targets = [_validate_sample(sample, models) for sample in source]
    ids = [sample["id"] for sample in source]
    if len(ids) != len(set(ids)):
        raise ValueError("样本 id 必须唯一")
    records = []
    for sample, (pos, rot) in zip(source, targets):
        side = sample["side"]
        model = models[side]
        row = dict(sample, pos=pos.tolist(), rot=rot.tolist(), q=None,
                   pos_err_m=None, rot_err_rad=None, min_limit_margin_rad=None,
                   ik_solved=False, pose_pass=False, margin_pass=False, accepted=False,
                   reason="no_solution", reasons=["no_solution"], solve_ms=0.)
        started = time.perf_counter()
        try:
            solution = solvers[side].solve(pos, rot, q_seed=HOME_Q.copy())
        except Exception as exc:
            row.update(reason="solver_error", reasons=["solver_error"],
                       error=f"{type(exc).__name__}: {exc}")
            solution = None
        finally:
            row["solve_ms"] = (time.perf_counter() - started) * 1000.
        if solution is not None:
            q = np.asarray(solution.q, float)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                row.update(reason="invalid_solution", reasons=["invalid_solution"])
            else:
                row["q"] = q.tolist()
                measured_pos, measured_rot = model.forward_kinematics(q)
                position_error = float(np.linalg.norm(measured_pos - pos))
                rotation_error = float(np.linalg.norm(log3(rot.T @ measured_rot)))
                margin = float(model.limit_margin(q))
                finite = all(math.isfinite(v) for v in (position_error, rotation_error, margin))
                in_limits = finite and model.within_limits(q)
                row.update(pos_err_m=position_error if math.isfinite(position_error) else None,
                           rot_err_rad=rotation_error if math.isfinite(rotation_error) else None,
                           min_limit_margin_rad=margin if math.isfinite(margin) else None,
                           ik_solved=bool(in_limits),
                           pose_pass=bool(in_limits and position_error <= pos_tol and rotation_error <= rot_tol),
                           margin_pass=bool(in_limits and margin >= min_margin))
                reasons = []
                if not finite:
                    reasons.append("invalid_solution")
                elif not in_limits:
                    reasons.append("joint_limits")
                else:
                    if not row["pose_pass"]:
                        reasons.append("pose_tolerance")
                    if not row["margin_pass"]:
                        reasons.append("margin_gate")
                row.update(accepted=not reasons, reason=reasons[0] if reasons else "accepted",
                           reasons=reasons or ["accepted"])
        records.append(row)
        if progress is not None:
            progress(len(records), len(source), row)

    by_side = {side: _summary([row for row in records if row["side"] == side]) for side in sides}
    by_cell = {}
    for side in sides:
        by_cell[side] = {"-".join(map(str, cell)): _summary([
            row for row in records if row["side"] == side and tuple(row["cell"]) == cell])
            for cell in sorted({tuple(row["cell"]) for row in records if row["side"] == side})}
    model_info = {}
    for side, model in models.items():
        home_pos, home_rot = model.forward_kinematics(HOME_Q)
        model_info[side] = dict(ee_link=model.ee_link, tcp_offset_m=model.tcp_offset.tolist(),
                                joint_names=list(model.joint_names),
                                home_q_rad=HOME_Q.tolist(), home_pos_m=home_pos.tolist(),
                                fixed_rot=home_rot.tolist(),
                                fixed_rpy_rad=matrix_to_rpy(home_rot).tolist())
    return dict(schema="x2ik.random_points.v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                offline_only=True, frame="torso_link", position_unit="m", angle_unit="rad",
                orientation_assumption=ORIENTATION_ASSUMPTION, relax_deg=0.,
                sampling_seeds=sorted({sample["seed"] for sample in source if "seed" in sample}),
                independent_seed_q_rad=HOME_Q.tolist(),
                thresholds=dict(pos_tol_m=pos_tol, rot_tol_rad=rot_tol, min_limit_margin_rad=min_margin),
                definitions=dict(ik_solved="有限且在关节限位内的返回解，不代表通过位姿误差门槛",
                                 pose_pass="返回解通过位置和姿态误差门槛",
                                 margin_pass="返回解至少保留指定关节限位余量",
                                 accepted="pose_pass 与 margin_pass 同时通过；不是运动或碰撞预检"),
                model=dict(urdf=DEFAULT_URDF.name,
                           urdf_sha256=hashlib.sha256(DEFAULT_URDF.read_bytes()).hexdigest(), sides=model_info),
                code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                             for name in ("x2_arm_model.py", "x2_srs_ik.py", "x2_srs_batch.py", "x2_random_points.py")},
                summary=dict(overall=_summary(records), by_side=by_side, by_cell=by_cell), samples=records)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=("both", "left", "right"), default="both")
    ap.add_argument("--samples-per-cell", type=int, default=DEFAULT_SAMPLES_PER_CELL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cells", type=int, nargs=3, default=DEFAULT_CELLS, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--x-range", type=float, nargs=2, default=DEFAULT_BOUNDS[0], metavar=("MIN", "MAX"))
    ap.add_argument("--abs-y-range", type=float, nargs=2, default=DEFAULT_BOUNDS[1], metavar=("MIN", "MAX"))
    ap.add_argument("--z-range", type=float, nargs=2, default=DEFAULT_BOUNDS[2], metavar=("MIN", "MAX"))
    ap.add_argument("--output", type=Path, default=Path("random_points_report.json"))
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    bounds = (args.x_range, args.abs_y_range, args.z_range)
    try:
        _sampling_config(args.samples_per_cell, args.seed, bounds, args.cells)
    except (ValueError, TypeError) as exc:
        ap.error(str(exc))
    if args.output.exists():
        ap.error(f"输出已存在，拒绝覆盖：{args.output}")
    sides = ("left", "right") if args.side == "both" else (args.side,)

    def progress(done, total, row):
        if done % args.samples_per_cell == 0:
            print(f"  {done}/{total}：{row['side']} cell={row['cell']}", flush=True)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            samples = [sample for side in sides for sample in generate_samples(
                side, args.samples_per_cell, args.seed, bounds, args.cells)]
            print(ORIENTATION_ASSUMPTION, flush=True)
            print(f"纯离线：seed={args.seed}，cells={list(args.cells)}，每格 {args.samples_per_cell} 点，"
                  f"总计 {len(samples)} 点；失败样本保留，不重抽。", flush=True)
            report = evaluate_samples(samples, progress=progress)
            report["sampling"] = dict(method="uniform_cartesian_stratified", seed=args.seed,
                                      bounds_x_abs_y_z_m=np.asarray(bounds).tolist(), cells=list(args.cells),
                                      samples_per_cell=args.samples_per_cell, sides=list(sides),
                                      mirrored_positions=True, resample_failures=False)
            text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            handle.write(text)
    except KeyboardInterrupt:
        print("离线覆盖已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"离线覆盖失败：{exc}", file=sys.stderr)
        return 1
    for side, summary in report["summary"]["by_side"].items():
        timing = summary["solve_ms"]
        print(f"{side}: 位姿通过 {summary['pose_pass']}/{summary['total']}；"
              f"余量门槛后通过 {summary['accepted']}/{summary['total']} "
              f"({summary['success_rate']:.1%})；"
              f"solve ms p50/p95/max={timing['p50']:.2f}/{timing['p95']:.2f}/{timing['max']:.2f}")
    print(f"完整原始记录：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
