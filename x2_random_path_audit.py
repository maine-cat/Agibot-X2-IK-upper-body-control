#!/usr/bin/env python3
"""随机覆盖报告的离线 MoveL 路径复核；不会连接 ROS，也不授权机器人运动。

每个种子、每臂、每个格子只选原始顺序中首个通过端点门槛的样本，不因路径失败
更换目标。独立复核 HOME→目标，再重新规划回 HOME TCP 位姿；拒绝去程不规划回程。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from x2_arm_model import ArmModel, log3
from x2_frames import HOME_Q
from x2_srs_ik import SrsArmIK
import x2_sim_ros as ros


def audit_segment(model, ik, q_start, position, rotation, duration=8.):
    """使用生产插值与 track，再加整条路径零失败和余量门槛；不包括 ROS/补偿。"""
    if not math.isfinite(duration) or duration < 8.:
        raise ValueError("duration 必须为有限数且至少 8 秒")
    q = np.asarray(q_start, float).copy()
    pos, rot = np.asarray(position, float), np.asarray(rotation, float)
    if (q.shape != (7,) or not np.all(np.isfinite(q)) or not model.within_limits(q)
            or pos.shape != (3,) or not np.all(np.isfinite(pos))
            or rot.shape != (3, 3) or not np.all(np.isfinite(rot))):
        raise ValueError("路径起点/目标形状、数值或限位无效")
    p0, r0 = model.forward_kinematics(q)
    if np.linalg.norm(log3(r0.T @ rot)) > math.radians(.5):
        raise ValueError("路径起点不在固定参考姿态的 0.5 度范围内")
    psi, branch = ik.sew_angle(q), None
    rotation_delta = log3(r0.T @ rot)
    frames = int(round(duration * ros.IK_RATE))
    trace, timings = [], []
    peak_step, peak_pos, peak_rot = 0., 0., 0.
    min_margin = float(model.limit_margin(q))
    reason, failure_frame = None, None
    for index in range(frames):
        alpha, _ = ros.quintic((index + 1) / frames)
        target = p0 + (pos - p0) * alpha
        target_rot = r0 @ ros.expm3(rotation_delta * alpha)
        _, clipped = ik.project_to_workspace(target, target_rot)
        if clipped:
            reason, failure_frame = "workspace_projection", index + 1
            break
        started = time.perf_counter()
        solution = ik.track(target, target_rot, q_prev=q, psi_prev=psi,
                            branch_prev=branch, fallback=False, clamp_reach=False)
        timings.append((time.perf_counter() - started) * 1000.)
        if solution is None:
            reason, failure_frame = "tracking_no_solution", index + 1
            break
        if np.shape(solution.q) != (7,) or not np.all(np.isfinite(solution.q)):
            reason, failure_frame = "invalid_solution", index + 1
            break
        delta = ros.angle_delta(solution.q, q)
        step = float(np.max(np.abs(delta)))
        peak_step = max(peak_step, step)
        next_q = q + delta
        if step > .05 or not model.within_limits(next_q):
            reason, failure_frame = "joint_step_or_limit", index + 1
            break
        next_branch = ros.branch_tuple(solution)
        if branch is not None and next_branch != branch:
            reason, failure_frame = "branch_changed", index + 1
            break
        measured, measured_rot = model.forward_kinematics(next_q)
        pos_err = float(np.linalg.norm(measured - target))
        rot_err = float(np.linalg.norm(log3(target_rot.T @ measured_rot)))
        reference_rot_err = float(np.linalg.norm(log3(rot.T @ measured_rot)))
        margin = float(model.limit_margin(next_q))
        if not all(math.isfinite(v) for v in (pos_err, rot_err, reference_rot_err, margin)):
            reason, failure_frame = "invalid_fk_error", index + 1
            break
        peak_pos, peak_rot = max(peak_pos, pos_err), max(peak_rot, reference_rot_err)
        min_margin = min(min_margin, margin)
        trace.append(dict(frame=index + 1, q_rad=next_q.tolist(), target_m=target.tolist(),
                          pos_err_m=pos_err, rot_err_rad=rot_err, margin_rad=margin,
                          reference_rot_err_rad=reference_rot_err,
                          branch=list(next_branch), step_rad=step))
        if pos_err > .0005 or max(rot_err, reference_rot_err) > math.radians(.5):
            reason, failure_frame = "path_pose_tolerance", index + 1
            break
        if margin < math.radians(3.):
            reason, failure_frame = "path_margin_gate", index + 1
            break
        q, psi, branch = next_q, solution.psi, next_branch
    accepted = reason is None and len(trace) == frames
    return dict(accepted=accepted, reason=reason or "accepted", failure_frame=failure_frame,
                planned_frames=frames, evaluated_frames=len(timings), q_start_rad=list(q_start),
                q_end_rad=q.tolist() if accepted else None, target_m=pos.tolist(), fixed_rot=rot.tolist(),
                peak_joint_step_rad=peak_step, peak_position_error_m=peak_pos,
                peak_rotation_error_rad=peak_rot, min_limit_margin_rad=min_margin,
                track_ms={key: float(value) for key, value in zip(("p50", "p95", "max"),
                    (np.percentile(timings, 50), np.percentile(timings, 95), max(timings)))}
                    if timings else None, trace=trace)


def audit_reports(paths, duration=8.):
    routes, sources, selection = [], [], []
    for path in paths:
        data = path.read_bytes()
        report = json.loads(data)
        if report.get("schema") != "x2ik.random_points.v1" or report.get("offline_only") is not True:
            raise ValueError(f"不是离线随机覆盖报告：{path}")
        seeds = {row.get("seed") for row in report["samples"]}
        if len(seeds) != 1 or None in seeds:
            raise ValueError("每个输入报告必须只含一个明确的随机 seed")
        seed = next(iter(seeds))
        sources.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                            endpoint_summary=report["summary"]))
        for side in ("left", "right"):
            model, ik = ArmModel(side), SrsArmIK(ArmModel(side))
            home_pos, home_rot = model.forward_kinematics(HOME_Q)
            rows = [row for row in report["samples"] if row["side"] == side]
            cells = sorted({tuple(row["cell"]) for row in rows})
            for cell in cells:
                pool = [row for row in rows if tuple(row["cell"]) == cell]
                point = next((row for row in pool if row["accepted"]), None)
                item = dict(source=str(path), seed=seed, side=side, cell=list(cell), cell_candidates=len(pool),
                            cell_accepted=sum(row["accepted"] for row in pool),
                            selected_id=None if point is None else point["id"])
                selection.append(item)
                if point is None:
                    continue
                if not np.allclose(point["rot"], home_rot, atol=1e-12, rtol=0):
                    raise ValueError("样本参考姿态与当前 HOME 模型不一致")
                outward = audit_segment(model, ik, HOME_Q, point["pos"], home_rot, duration)
                returning = (audit_segment(model, ik, outward["q_end_rad"], home_pos, home_rot, duration)
                             if outward["accepted"] else None)
                passed = outward["accepted"] and returning["accepted"]
                routes.append(dict(**item, accepted=bool(passed), outward=outward, returning=returning))
                print(f"{path.name} {side} {point['id']}: "
                      f"out={outward['reason']} back={returning['reason'] if returning else 'not_planned'}",
                      flush=True)
    return dict(schema="x2ik.random_paths.v1", offline_only=True,
                created_at_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                method="first accepted endpoint per seed/side/cell; HOME-to-target then replan to HOME TCP",
                path_gates=dict(position_tolerance_m=.0005, rotation_tolerance_rad=math.radians(.5),
                                min_margin_rad=math.radians(3.), max_joint_step_rad=.05,
                                tolerate_failed_frames=0, pose_relaxation=False),
                duration_s=duration, selection=selection, routes=routes,
                summary=dict(selected_routes=len(routes), passed=sum(r["accepted"] for r in routes),
                    rejected=sum(not r["accepted"] for r in routes),
                    cells_without_candidate=sum(row["selected_id"] is None for row in selection),
                    outward_reasons=dict(Counter(r["outward"]["reason"] for r in routes)),
                    return_not_planned=sum(r["returning"] is None for r in routes),
                    return_evaluated=sum(r["returning"] is not None for r in routes),
                    return_passed=sum(bool(r["returning"] and r["returning"]["accepted"]) for r in routes),
                    return_reasons=dict(Counter(r["returning"]["reason"] for r in routes if r["returning"]))),
                code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("x2_random_path_audit.py", "x2_arm_model.py", "x2_srs_ik.py",
                                          "x2_srs_batch.py", "x2_sim_ros.py", "x2_frames.py", "x2_ultra.urdf")},
                limitations=["仅离线模型路径检查，不是机器人执行记录或物理精度验证",
                             "未做碰撞检测，不证明前方空间没有障碍物",
                             "拟人手系尚未定义，暂按各侧 HOME 腕坐标系固定旋转",
                             "跟踪器与插值沿用生产代码，但零失败帧和 3 度余量门槛更严格",
                             "回程重新规划到 HOME 位姿，不保证回到相同 HOME 关节角或沿原关节轨迹",
                             "只选每格一个端点通过样本，路径通过率不代表全部随机候选"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 8.:
        parser.error("--duration 必须至少 8 秒")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"initializing","offline_only":true}\n')
    report = audit_reports(args.reports, args.duration)
    from x2_converge_test import save_report
    save_report(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
