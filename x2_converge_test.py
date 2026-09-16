#!/usr/bin/env python3
"""URS 到点对照测试：默认只读预检，只有 --run 才发布运动指令。

每点先 MoveJ 接近已知关节姿态，再用同一 FK 位姿 MoveL 精定位；本工具
检验关节反馈 FK 到点精度，不代表长距离直线轨迹或外部 TCP 精度验收。
可直接 python3 x2_converge_test.py --preflight，自动沿用 x2ik.ros_env。
在线检查前须显式设置 X2_TEST_EXPECTED_SN，与本机 X2_ROBOT_SN 和标定 sn 一致。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from datetime import datetime

HERE = Path(__file__).resolve().parent


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="显式允许 URS 点位运动")
    mode.add_argument("--preflight", action="store_true", help="只读真机预检（默认）")
    mode.add_argument("--offline", action="store_true", help="只做模型和路径预检，不创建 ROS 节点")
    ap.add_argument("--side", choices=("left", "right"), default="right")
    ap.add_argument("--point", choices=("all", "home", "lateral", "forward"), default="all")
    ap.add_argument("--repeats", type=int, default=5, help="每点每组有效次数，另加 1 次丢弃热身")
    ap.add_argument("--groups", choices=("both", "off", "on"), default="both")
    ap.add_argument("--converge", type=int, default=3, help="on 组最多修正轮数；0 只跑 off")
    ap.add_argument("--converge-tol", type=float, default=1., help="收敛阈值 mm，最大 1")
    ap.add_argument("--converge-step", type=float, default=.5, help="单轮关节修正上限 deg，最大 0.5")
    ap.add_argument("--converge-total", type=float, default=3., help="累计关节修正上限 deg，最大 3")
    ap.add_argument("--duration", type=float, default=8., help="MoveJ 接近时长 s，最少 8")
    ap.add_argument("--cartesian-duration", type=float, default=2., help="近距离 MoveL 时长 s，最少 2")
    ap.add_argument("--settle", type=float, default=2., help="每轮保持 s，最少 2")
    ap.add_argument("--output", type=Path, default=None, help="完整 JSON 原始记录文件")
    return ap


def validate_args(args):
    if args.repeats < 1 or args.converge < 0:
        raise ValueError("repeats 必须 >=1，converge 必须 >=0")
    for name, minimum in (("duration", 8.), ("cartesian_duration", 2.), ("settle", 2.)):
        value = getattr(args, name)
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{name} 必须为有限数且 >= {minimum}s")
    if args.groups == "on" and args.converge == 0:
        raise ValueError("--groups on 要求 --converge > 0")
    for name, maximum in (("converge_tol", 1.), ("converge_step", .5), ("converge_total", 3.)):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value > maximum:
            raise ValueError(f"{name} 要求 0 < 值 <= {maximum}")
    if args.converge_step > args.converge_total:
        raise ValueError("单轮修正上限不能大于累计修正上限")


def is_urs(action):
    return isinstance(action, str) and re.fullmatch(
        r"(?:URS|UPPERBODY_REMOTE_SPLIT)(?:\(\d+\))?", action.strip()) is not None


def forbid_action(*args, **kwargs):
    raise RuntimeError("本测试禁止所有状态切换；只允许当前 URS 下做点位")


def bootstrap(argv):
    import x2ik
    x2ik.load_conf()
    if os.environ.get("X2_CONVERGE_ENV_READY") == "1":
        return None
    py = x2ik.ros_python()
    if py is None:
        raise RuntimeError("找不到 ROS Python，先在机器人端运行 ./x2ik.py doctor")
    env = x2ik.ros_env(x2ik.find_sim_home(), x2ik.find_msgs_prefix())
    env["X2_CONVERGE_ENV_READY"] = "1"
    return x2ik.run_with_ros([py, str(Path(__file__).resolve()), *argv], env)


def calibration(ros):
    expected_sn = os.environ.get("X2_TEST_EXPECTED_SN")
    if not expected_sn or not expected_sn.strip():
        raise RuntimeError("必须显式设置 X2_TEST_EXPECTED_SN 为本次已核对的机器人序列号")
    sn = os.environ.get("X2_ROBOT_SN")
    if not sn or not sn.strip() or sn != expected_sn:
        raise RuntimeError(f"X2_ROBOT_SN={sn!r}，与 X2_TEST_EXPECTED_SN={expected_sn!r} 不一致；拒绝加载其它机器参数")
    data = ros.load_calibration(sn)
    if not data or data.get("sn") != sn:
        raise RuntimeError("当前机器标定文件缺失或文件内 sn 不匹配")
    if (data.get("stiffness") != 40 or data.get("bias_limit_deg") != 12
            or data.get("gravity_source") != "pelvis"):
        raise RuntimeError("本轮测试仅接受 stiffness=40 / bias_limit_deg=12 / pelvis")
    return data


def make_points(ros, model, side):
    qs = {"home": ros.HOME_Q.copy(),
          "lateral": ros.lateral_raise_q(side, math.radians(-20.), model),
          "forward": ros.HOME_Q.copy()}
    qs["forward"][0] = math.radians(55.)
    return {name: dict(q=q, pos=model.forward_kinematics(q)[0],
                       rot=model.forward_kinematics(q)[1]) for name, q in qs.items()}


def validate_path(ros, model, ik, q_start, point, duration):
    """逐帧复现 MoveL 解支/步长检查，不发布；发现投影、无解或跳支立即拒绝。"""
    import numpy as np
    q = np.array(q_start, float)
    if not np.all(np.isfinite(q)) or np.max(np.abs(model.clamp(q) - q)) > 1e-9:
        raise RuntimeError("预检起始关节超限或非有限数")
    p0, r0 = model.forward_kinematics(q)
    dr = ros.log3(r0.T @ point["rot"])
    psi, branch, peak = ik.sew_angle(q), None, 0.
    frames = max(1, int(round(duration * ros.IK_RATE)))
    for k in range(frames):
        a, _ = ros.quintic((k + 1) / frames)
        pos = p0 + (point["pos"] - p0) * a
        rot = r0 @ ros.expm3(dr * a)
        projected, clipped = ik.project_to_workspace(pos, rot)
        sol = ik.track(projected, rot, q_prev=q, psi_prev=psi,
                       branch_prev=branch, fallback=False)
        if clipped or sol is None:
            raise RuntimeError(f"MoveL 预检第 {k + 1}/{frames} 帧裁剪或无 IK 解")
        delta = ros.angle_delta(sol.q, q)
        peak = max(peak, float(np.max(np.abs(delta))))
        if peak > .05 or np.max(np.abs(model.clamp(q + delta) - q - delta)) > 1e-9:
            raise RuntimeError(f"MoveL 预检第 {k + 1}/{frames} 帧关节跳变/限位")
        q += delta
        psi, branch = sol.psi, ros.branch_tuple(sol)
    reached, rotation = model.forward_kinematics(q)
    if np.linalg.norm(reached - point["pos"]) > .0005:
        raise RuntimeError("MoveL 预检终点误差超过 0.5 mm")
    return {"frames": frames, "peak_joint_step_rad": peak,
            "position_error_m": float(np.linalg.norm(reached - point["pos"])),
            "rotation_error_rad": float(np.linalg.norm(ros.log3(point["rot"].T @ rotation)))}


def plan(args):
    names = ("home", "lateral", "forward") if args.point == "all" else (args.point,)
    groups = [("off", 0), ("on", args.converge)]
    groups = [(name, count) for name, count in groups
              if (args.groups == "both" or args.groups == name)
              and (name == "off" or count > 0)]
    return [dict(group=group, converge=count, point=name, repeat=repeat,
                 discarded=repeat == 0)
            for group, count in groups for repeat in range(args.repeats + 1)
            for name in names]


def check_feedback(cli, ros):
    import numpy as np
    expected = {f"{side}_{suffix}" for side in ("left", "right")
                for suffix in ros.ARM_JOINT_SUFFIX}
    missing = expected - set(cli.state_names)
    if missing:
        raise RuntimeError(f"最新反馈缺少关节：{sorted(missing)}")
    for side in ("left", "right"):
        q = cli.q(side)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(cli.dq(side))):
            raise RuntimeError(f"{side} 关节反馈非有限数")
        if np.max(np.abs(cli.models[side].clamp(q) - q)) > math.radians(.5):
            raise RuntimeError(f"{side} 反馈已在关节限位外")
    if cli.imu_count["pelvis"] == 0 or cli.imu_has_orientation["pelvis"] is not True:
        raise RuntimeError("没有 pelvis IMU 有效姿态，拒绝静默使用默认重力")


def check_graph(cli, ros):
    pubs = cli.node.count_publishers(ros.UPPER_BODY_TOPIC)
    subs = cli.node.count_subscribers(ros.UPPER_BODY_TOPIC)
    if pubs != 1 or subs < 1:
        raise RuntimeError(f"upper_body 发布者={pubs}（必须只有本节点 1 个），订阅者={subs}")
    return dict(publishers=pubs, subscribers=subs)


def preflight(cli, ros):
    if cli.mode != "upper_body":
        raise RuntimeError("仅允许 upper_body 接口")
    if not cli.wait_state(5.):
        raise RuntimeError("5s 内无关节反馈")
    # 留出独立 DDS 端点发现窗口；随后连续 1s 检查独占状态。
    cli.spin(3.)
    action = cli.get_action()
    if not is_urs(action):
        raise RuntimeError(f"当前 action={action!r}，仅接受已在 URS")
    count = cli.state_count
    imu_count = cli.imu_count["pelvis"]
    for _ in range(10):
        check_graph(cli, ros)
        cli.spin(.1)
    if cli.state_count <= count or cli.imu_count["pelvis"] <= imu_count:
        raise RuntimeError("反馈/IMU 未推进，拒绝使用旧帧")
    if not cli.fresh_state(.5):
        raise RuntimeError("预检未获得新关节反馈")
    check_feedback(cli, ros)
    return dict(action=action, state_count=cli.state_count,
                imu_count=dict(cli.imu_count), **check_graph(cli, ros))


class FrameBuffer:
    """动作期间只写预分配数值内存，停止发送后才展开原 JSON 字典结构。"""
    PHASES = ("idle", "excitation", "approach", "cartesian")
    JOINT_KEYS = ("q_input_left", "q_input_right", "q_sent_left", "q_sent_right",
                  "q_meas_left", "q_meas_right")

    def __init__(self, capacity):
        import numpy as np
        self.data = np.empty((capacity, 48), dtype=np.float64)
        self.data.fill(0.)         # 首条指令之前触碰页面，避免运动中首次分配页面。
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, stamp, phase, count, joints, guard_s=0., send_s=0.):
        if self.size >= len(self.data):
            raise RuntimeError("预分配记录缓冲已满，停止序列")
        row = self.data[self.size]
        row[0:3] = stamp, self.PHASES.index(phase), count
        for index, values in enumerate(joints):
            row[3 + index * 7:10 + index * 7] = values
        row[45:47] = guard_s, send_s
        self.size += 1

    def __getitem__(self, index):
        row = self.data[:self.size][index]
        result = dict(t_s=float(row[0]), phase=self.PHASES[int(row[1])],
                      state_count=int(row[2]), guard_s=float(row[45]),
                      send_call_s=float(row[46]), record_s=float(row[47]))
        for joint, key in enumerate(self.JOINT_KEYS):
            result[key] = row[3 + joint * 7:10 + joint * 7].tolist()
        return result

    def tolist(self):
        return [self[index] for index in range(self.size)]


class GcTiming:
    """用固定数值缓冲旁听 GC 耗时；不禁用 ROS 运行时的循环垃圾回收。"""
    def __init__(self, origin, capacity=32768):
        import numpy as np
        self.origin = origin
        self.starts = [0., 0., 0.]
        self.events = np.empty((capacity, 5), dtype=np.float64)
        self.events.fill(0.)
        self.size = 0
        self.dropped = 0
        self.callback = self._on_gc

    def _on_gc(self, phase, info):
        now = time.monotonic()
        generation = info["generation"]
        if phase == "start":
            self.starts[generation] = now
        elif self.size < len(self.events):
            self.events[self.size] = (self.starts[generation] - self.origin,
                                      now - self.starts[generation], generation,
                                      info.get("collected", 0), info.get("uncollectable", 0))
            self.size += 1
        else:
            self.dropped += 1

    def start(self):
        gc.callbacks.append(self.callback)

    def stop(self):
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)

    def report(self):
        return dict(columns=["start_t_s", "duration_s", "generation", "collected", "uncollectable"],
                    events=self.events[:self.size], dropped=self.dropped,
                    gc_enabled=gc.isenabled(), thresholds=gc.get_threshold())


class MotionMonitor:
    """固定非测试臂输入，并记录每次真正 publish 后的双臂命令和反馈。"""
    def __init__(self, cli, ros, side):
        self.cli, self.ros, self.side = cli, ros, side
        self.other = "left" if side == "right" else "right"
        self.fixed_other = cli.q(self.other).copy()
        self.original_send = cli.send
        self.last_send = None
        self.last_state_count = cli.state_count
        self.last_fresh_at = time.monotonic()
        self.last_graph_at = 0.
        self.max_gap = 0.
        self.phase = "idle"
        self.rows = FrameBuffer(4096)
        self.started_at = time.monotonic()

    def send(self, left, right, dl=None, dr=None):
        import numpy as np
        now = time.monotonic()
        if len(self.rows) >= len(self.rows.data):
            raise RuntimeError("预分配记录缓冲已满，停止序列")
        if self.last_send is not None:
            gap = now - self.last_send
            self.max_gap = max(self.max_gap, gap)
            if gap >= .2:
                raise RuntimeError(f"发送间隔 {gap * 1000:.1f} ms 达到 mc 200 ms 过期线")
        if self.cli.state_count > self.last_state_count:
            self.last_state_count = self.cli.state_count
            self.last_fresh_at = now
        elif self.last_send is not None and now - self.last_fresh_at >= .2:
            raise RuntimeError("关节反馈超过 200 ms 未更新，停止点位序列")
        check_feedback(self.cli, self.ros)
        if now - self.last_graph_at >= .5:
            check_graph(self.cli, self.ros)
            self.last_graph_at = now
        if self.other == "left":
            left, dl = self.fixed_other, None
        else:
            right, dr = self.fixed_other, None
        checked_at = time.monotonic()
        self.original_send(left, right, dl, dr)
        stamp = time.monotonic()
        # 包含本帧 send() 的计算/发布耗时；以发布完成到完成间隔报告。
        completed_gap = 0. if self.last_send is None else stamp - self.last_send
        self.max_gap = max(self.max_gap, completed_gap)
        self.last_send = stamp
        self.rows.append(stamp - self.started_at, self.phase, self.cli.state_count,
                         (left, right, self.cli._last_sent["left"], self.cli._last_sent["right"],
                          self.cli.q("left"), self.cli.q("right")),
                         guard_s=checked_at - now, send_s=stamp - checked_at)
        self.rows.data[self.rows.size - 1, 47] = time.monotonic() - stamp
        if completed_gap >= .2:
            raise RuntimeError(f"实际发布完成间隔 {completed_gap * 1000:.1f} ms 达到 200 ms")


def result_error(result):
    if result.get("stale"):
        return "终点没有新反馈"
    if result.get("aborted") or not result.get("trajectory_valid", True):
        return "轨迹被中止"
    if any(result.get(key, 0) for key in ("ik_fails", "clipped", "step_rejects", "branch_rejects")):
        return "轨迹出现无解/裁剪/跳支"
    reason = result.get("converge_reason")
    if reason and reason not in ("tolerance", "max_iterations"):
        return f"闭环停止原因：{reason}"
    if not math.isfinite(result["pos_err"]):
        return "终点误差非有限数"
    return None


def summary(rows):
    import numpy as np
    out = {}
    for row in rows:
        if row["discarded"] or row.get("error") or "result" not in row:
            continue
        key = f"{row['group']}/{row['point']}"
        out.setdefault(key, []).append(row["result"]["pos_err"] * 1000.)
    return {key: dict(n=len(values), median_mm=float(np.median(values)),
                      range_mm=float(np.ptp(values)), min_mm=min(values), max_mm=max(values),
                      all_within_1mm=max(values) <= 1.) for key, values in out.items()}


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_safe(value):
    """异常反馈中的 NaN/Inf 也必须完整留档；JSON 中使用 null 标识。"""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_report(output, report):
    payload = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # 先完整序列化并写临时文件，避免异常/中断截断原报告。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=output.name + ".", suffix=".tmp", delete=False) as stream:
        stream.write(payload)
        temporary = stream.name
    os.replace(temporary, output)


def run(args, ros):
    import numpy as np
    model = ros.ArmModel(args.side)
    points = make_points(ros, model, args.side)
    ik = ros.SrsArmIK(model)
    checks = {name: validate_path(ros, model, ik, point["q"], point,
                                 args.cartesian_duration)
              for name, point in points.items()}
    tasks = plan(args)
    print(f"路径预检通过：{list(points)}；{len(tasks)} 次定位，"
          f"每点每组丢弃首轮、保留 {args.repeats} 次", flush=True)
    if args.offline:
        print(json.dumps(dict(points=points, checks=checks, plan=tasks),
                         default=json_default, ensure_ascii=False, indent=2), flush=True)
        return 0
    calib = calibration(ros)
    output = args.output or HERE / "results" / (
        "converge-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    # 独占保留报告路径先于创建 ROS 节点和全部运动；不能覆盖旧实测数据。
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write('{"status": "initializing"}\n')
    except FileExistsError as exc:
        raise RuntimeError(f"输出文件已存在，拒绝覆盖：{output}") from exc
    cli = ros.X2ArmClient("upper_body", joint_stiffness=np.full(7, 40.),
                          gravity_source="pelvis", bias_limit=math.radians(12.),
                          hand_mode=1)
    cli.set_action = forbid_action
    cli.enter_control = forbid_action
    report = dict(schema_version=1, started_at=datetime.now().astimezone().isoformat(),
                  metric="feedback_joint_fk_tcp_position_error_m; excludes assembly/external TCP error",
                  method="known-joint approach followed by short Cartesian positioning",
                  json_nonfinite_policy="NaN/Inf stored as null; inspect error fields",
                  sn=calib["sn"], calibration=calib, arguments=vars(args), points=points,
                  path_checks=checks, plan=tasks, trials=[], status="preflight")
    monitor = gc_timing = None
    code = 0
    try:
        report["preflight"] = preflight(cli, ros)
        print("只读预检通过：URS、upper_body 独占、双臂新反馈、pelvis IMU 有效", flush=True)
        if args.run:
            save_report(output, report)
            report["status"] = "running"
            monitor = MotionMonitor(cli, ros, args.side)
            # 包含单点重复时的额外消隙行程及每轮 fresh-hold 最坏 0.5s。
            # 全部提前分配；各 trial 仅保留数值缓冲引用，不在点间展开/写 JSON。
            trial_buffers = [FrameBuffer(int(math.ceil(cli.rate * (
                2 * args.duration + args.settle + args.cartesian_duration
                + (task["converge"] + 1) * (args.settle + .5)))) + 128) for task in tasks]
            gc_timing = GcTiming(monitor.started_at)
            gc_timing.start()
            cli.send = monitor.send
            report["fixed_other_input"] = monitor.fixed_other
            previous = None
            for task, buffer in zip(tasks, trial_buffers):
                row = dict(task, started_state_count=cli.state_count)
                report["trials"].append(row)
                monitor.rows = buffer
                point = points[task["point"]]
                # 单点小样本也要改变载荷方向，避免连续原地保持虚构重复性。
                if previous == task["point"]:
                    reset = points["forward" if task["point"] == "home" else "home"]
                    monitor.phase = "excitation"
                    ros.goto_joint(cli, {args.side: reset["q"],
                                        monitor.other: monitor.fixed_other}, args.duration, 0.)
                monitor.phase = "approach"
                ros.goto_joint(cli, {args.side: point["q"], monitor.other: monitor.fixed_other},
                               args.duration, args.settle)
                monitor.phase = "cartesian"
                result = ros.goto_cartesian(cli, args.side, point["pos"], point["rot"],
                                             args.cartesian_duration, args.settle,
                                             converge=task["converge"],
                                             converge_tol=args.converge_tol / 1000.,
                                             converge_step=math.radians(args.converge_step),
                                             converge_total=math.radians(args.converge_total))
                row.update(result=result, final_state_count=cli.state_count,
                           fresh_state_delta=cli.state_count - row["started_state_count"],
                           max_send_gap_s=monitor.max_gap, frames=monitor.rows,
                           other_drift_rad=ros.angle_delta(cli.q(monitor.other), monitor.fixed_other))
                error = result_error(result)
                curve = result.get("converge_hist", [result["pos_err"]])
                row["curve_mm"] = [float(value * 1000.) for value in curve]
                print(f"{task['group']}/{task['point']} #{task['repeat']} "
                      f"{'丢弃' if task['discarded'] else '有效'} "
                      f"err={result['pos_err'] * 1000:.3f}mm "
                      f"curve={row['curve_mm']} fresh_count={cli.state_count} "
                      f"max_gap={monitor.max_gap * 1000:.1f}ms", flush=True)
                if error:
                    row["error"] = error
                    raise RuntimeError(error)
                previous = task["point"]
            report["status"] = "completed"
            measurements = summary(report["trials"])
            if any(not stats["all_within_1mm"] for key, stats in measurements.items()
                   if key.startswith("on/")):
                report["status"] = "precision_not_met"
                code = 3
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        if monitor and report["trials"] and "frames" not in report["trials"][-1]:
            report["trials"][-1]["frames"] = monitor.rows
        print(f"[停止序列] {exc}；未调用任何状态切换", file=sys.stderr, flush=True)
    finally:
        if gc_timing:
            gc_timing.stop()
            report["gc_timing"] = gc_timing.report()
        # 不再下发运动；只查询结束状态，保持 URS，绝不调用 passive/reenter。
        try:
            action = cli.get_action()
            report["final_action"] = action
            if not is_urs(action):
                code = code or 1
                report["final_action_error"] = "结束时未确认 URS"
            print(f"结束 action={action!r}", flush=True)
        except Exception as exc:
            report["final_action_error"] = str(exc)
            code = code or 1
        cli.close()
        if cli.rclpy.ok():
            cli.rclpy.shutdown()
        report["summary"] = summary(report["trials"])
        report["finished_at"] = datetime.now().astimezone().isoformat()
        if monitor:
            report["max_send_gap_s"] = monitor.max_gap
        save_report(output, report)
        print(f"记录：{output}", flush=True)
        print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    args = ap.parse_args(argv)
    try:
        validate_args(args)
        if not args.offline:
            status = bootstrap(argv)
            if status is not None:
                return status
        # 限制小矩阵 BLAS 线程数；必须发生在第一次 import numpy 之前。
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(key, "1")
        import x2_sim_ros as ros
        return run(args, ros)
    except (RuntimeError, ValueError) as exc:
        print(f"[拒绝执行] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
