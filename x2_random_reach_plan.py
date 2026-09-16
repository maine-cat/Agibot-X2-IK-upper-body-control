#!/usr/bin/env python3
"""随机点位测试的纯离线计划层，不创建 ROS 节点或发送运动。

首轮默认范围只是前方小区间，不宣称碰撞安全。位置从 Cartesian 网格直接采样，
固定本侧 HOME 的 wrist_roll_link 旋转；没有手掌外参，且不放宽姿态或重抽失败点。
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import product
import json
import math
import os
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from x2_arm_model import ArmModel, log3
from x2_frames import HOME_Q
from x2_srs_ik import SrsArmIK
from x2_random_path_audit import audit_segment

DEFAULT_BOUNDS = ((.08, .16), (.23, .30), (-.09, -.03))  # x, abs(y), z；m
BROAD_BOUNDS = ((.08, .30), (.18, .36), (-.10, .10))
DEFAULT_SEED = 20260915
DEFAULT_COUNT = 8
DEFAULT_DURATION = 8.
CELLS = (2, 2, 2)
POS_TOL = .0005
ROT_TOL = math.radians(.5)
MIN_MARGIN = math.radians(3.)


def _config(side, count, seed, bounds):
    if side not in ("left", "right"):
        raise ValueError("side 必须为 left 或 right")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 32:
        raise ValueError("count 必须是 1 到 32 之间的整数")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed 必须是非负整数")
    box = np.asarray(bounds, float)
    broad = np.asarray(BROAD_BOUNDS)
    if (box.shape != (3, 2) or not np.all(np.isfinite(box)) or np.any(box[:, 0] >= box[:, 1])
            or np.any(box[:, 0] < broad[:, 0]) or np.any(box[:, 1] > broad[:, 1])):
        raise ValueError("bounds 必须为 x/abs(y)/z 的三组有限递增区间，且位于已有广域内："
                         "x [0.08,0.30], abs(y) [0.18,0.36], z [-0.10,0.10] m")
    return int(count), int(seed), box


def _args_config(args):
    side = getattr(args, "side", "right")
    count = getattr(args, "count", DEFAULT_COUNT)
    seed = getattr(args, "seed", DEFAULT_SEED)
    bounds = getattr(args, "bounds", None)
    if bounds is None:
        bounds = (getattr(args, "x_range", DEFAULT_BOUNDS[0]),
                  getattr(args, "abs_y_range", DEFAULT_BOUNDS[1]),
                  getattr(args, "z_range", DEFAULT_BOUNDS[2]))
    count, seed, box = _config(side, count, seed, bounds)
    duration = getattr(args, "duration", DEFAULT_DURATION)
    if not math.isfinite(duration) or duration < DEFAULT_DURATION:
        raise ValueError("duration 必须为有限数且至少 8 秒")
    return side, count, seed, box, float(duration)


def validate_args(args):
    """供运行器在 ROS 初始化或输出预留前使用；只检查离线计划参数。"""
    _args_config(args)


def random_candidates(side, count=DEFAULT_COUNT, seed=DEFAULT_SEED, bounds=DEFAULT_BOUNDS):
    """按 2×2×2 格轮流取点，保留相同 count 前缀与左右镜像位置。

    每格 SeedSequence([seed, ix, iy, iz]) 独立随机流；count=3 取前三格，count=8
    每格一个，count=32 每格四个。不通过拒绝采样筛选，不由关节 FK 制造目标位置。
    """
    count, seed, box = _config(side, count, seed, bounds)
    _, rotation = ArmModel(side).forward_kinematics(HOME_Q)
    edges = [np.linspace(*limits, 3) for limits in box]
    cells = list(product(range(2), repeat=3))
    streams = {cell: np.random.default_rng(np.random.SeedSequence([seed, *cell])) for cell in cells}
    candidates = []
    for index in range(count):
        cell = cells[index % len(cells)]
        sample_index = index // len(cells)
        low = [edges[axis][number] for axis, number in enumerate(cell)]
        high = [edges[axis][number + 1] for axis, number in enumerate(cell)]
        pos = streams[cell].uniform(low, high)
        if side == "right":
            pos[1] *= -1.
        candidates.append(dict(id=f"{side}-{'-'.join(map(str, cell))}-{sample_index:04d}",
                               side=side, cell=list(cell), sample_index=sample_index, seed=seed,
                               pos=pos.tolist(), rot=rotation.tolist()))
    return candidates


def endpoint_check(model, solver, position, rotation):
    """独立 HOME 初值求解，并通过 FK 复验返回解；不投影、不放宽姿态。"""
    pos, rot = np.asarray(position, float), np.asarray(rotation, float)
    result = dict(q=None, pos_err_m=None, rot_err_rad=None, min_limit_margin_rad=None,
                  accepted=False, reason="no_solution", reasons=["no_solution"], solve_ms=0.)
    started = time.perf_counter()
    try:
        solution = solver.solve(pos.copy(), rot.copy(), q_seed=HOME_Q.copy())
    except Exception as exc:
        result.update(reason="solver_error", reasons=["solver_error"],
                      error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result["solve_ms"] = (time.perf_counter() - started) * 1000.
    if solution is None:
        return result
    q = np.asarray(solution.q, float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        result.update(reason="invalid_solution", reasons=["invalid_solution"])
        return result
    measured_pos, measured_rot = model.forward_kinematics(q)
    pos_err = float(np.linalg.norm(measured_pos - pos))
    rot_err = float(np.linalg.norm(log3(rot.T @ measured_rot)))
    margin = float(model.limit_margin(q))
    finite = all(math.isfinite(value) for value in (pos_err, rot_err, margin))
    reasons = []
    if not finite:
        reasons.append("invalid_solution")
    elif not model.within_limits(q):
        reasons.append("joint_limits")
    else:
        if pos_err > POS_TOL or rot_err > ROT_TOL:
            reasons.append("pose_tolerance")
        if margin < MIN_MARGIN:
            reasons.append("margin_gate")
    result.update(q=q.tolist(), pos_err_m=pos_err if math.isfinite(pos_err) else None,
                  rot_err_rad=rot_err if math.isfinite(rot_err) else None,
                  min_limit_margin_rad=margin if math.isfinite(margin) else None,
                  accepted=not reasons, reason=reasons[0] if reasons else "accepted",
                  reasons=reasons or ["accepted"])
    return result


def segment_summary(model, solver, q_start, position, rotation, duration):
    """调用既有逐帧路径审查，移除 trace；保留失败帧、余量、峰值及成功终点。"""
    try:
        result = audit_segment(model, solver, np.asarray(q_start).copy(), np.asarray(position).copy(),
                               np.asarray(rotation).copy(), duration=duration)
    except Exception as exc:
        return dict(accepted=False, reason="path_audit_error", error=f"{type(exc).__name__}: {exc}",
                    q_end_rad=None)
    return {key: value for key, value in result.items() if key != "trace"}


def plan_digest(metadata):
    """计划输入的规范 JSON 哈希；调用方可按返回的 hash_input 重新验证。"""
    return hashlib.sha256(json.dumps(metadata, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def make_plan(args, ros=None):
    """完整离线审查：端点、HOME→点、从去程关节终点重新规划→HOME TCP。

    ros 参数供运行器保持统一接口，本模块不调用其 ROS 客户端或运动函数。
    回程只保证回到 HOME TCP 位姿，不保证与 HOME_Q 相同；运行时须按实际反馈重审路径。
    """
    side, count, seed, box, duration = _args_config(args)
    model = ArmModel(side)
    solver = SrsArmIK(model)
    home_pos, fixed_rot = model.forward_kinematics(HOME_Q)
    candidates = random_candidates(side, count, seed, box)
    hash_input = dict(schema="x2ik.random_reach_plan.v1", side=side, seed=seed, count=count,
                      bounds=box.tolist(), cells=list(CELLS), duration_s=duration,
                      fixed_rot=fixed_rot.tolist(), home_pos=home_pos.tolist(), home_q=HOME_Q.tolist(),
                      thresholds=dict(pos_tol_m=POS_TOL, rot_tol_rad=ROT_TOL,
                                      min_limit_margin_rad=MIN_MARGIN, max_joint_step_rad=.05,
                                      tolerate_failed_frames=0, relax_deg=0.),
                      candidates=[dict(row) for row in candidates])
    results = []
    for candidate in candidates:
        endpoint = endpoint_check(model, solver, candidate["pos"], candidate["rot"])
        row = dict(candidate, endpoint=endpoint, outward=None, return_path=None,
                   accepted=False, reason=f"endpoint:{endpoint['reason']}")
        if endpoint["accepted"]:
            outward = segment_summary(model, solver, HOME_Q, candidate["pos"], fixed_rot, duration)
            row.update(outward=outward, reason=f"outward:{outward['reason']}")
            if outward["accepted"]:
                returning = segment_summary(model, solver, outward["q_end_rad"], home_pos, fixed_rot, duration)
                row.update(return_path=returning, reason=f"return:{returning['reason']}")
                if returning["accepted"]:
                    row.update(accepted=True, reason="accepted")
        results.append(row)
    accepted_ids = [row["id"] for row in results if row["accepted"]]
    summary = dict(total=count, endpoint_accepted=sum(row["endpoint"]["accepted"] for row in results),
                   outward_evaluated=sum(row["outward"] is not None for row in results),
                   return_evaluated=sum(row["return_path"] is not None for row in results),
                   accepted=len(accepted_ids), rejected=count - len(accepted_ids),
                   success_rate=len(accepted_ids) / count,
                   reason_counts=dict(Counter(row["reason"] for row in results)))
    return dict(schema=hash_input["schema"], offline_only=True, side=side, seed=seed, count=count,
                bounds=box.tolist(), frame="torso_link", position_unit="m", angle_unit="rad",
                duration_s=duration, fixed_rot=fixed_rot.tolist(), home_pos=home_pos.tolist(),
                home_q=HOME_Q.tolist(), orientation_assumption="各侧 HOME wrist_roll_link 固定旋转；暂无手掌外参",
                thresholds=hash_input["thresholds"], hash_input=hash_input,
                plan_sha256=plan_digest(hash_input), candidates=results, accepted_ids=accepted_ids,
                summary=summary, limitations=[
                    "前方小区间只是首轮采样范围，未做碰撞检测，不证明环境或双臂碰撞安全",
                    "纯离线计划不是机器人执行记录，不证明实机精度",
                    "所有候选保留，拒绝不重抽，端点无解不代表物理不可达",
                    "每次真实运动须以当前完整反馈重审路径，不能直接重放离线关节轨迹",
                    "回程重新规划到 HOME TCP，不保证回到相同 HOME 关节角或沿原关节轨迹"])
