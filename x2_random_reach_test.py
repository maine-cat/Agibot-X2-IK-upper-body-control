#!/usr/bin/env python3
"""随机点真机到达测试：默认离线；--preflight 只读；--execute 才发送 URS 点位。

固定本侧 HOME 腕姿态，不是已校准的手掌/指尖系。测试臂须已在 HOME 附近。
逐点 MoveL 到达后按计划返回 HOME TCP；故障退出不追加回位，不请求任何状态切换。
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import x2_converge_test as base
from x2_mdi import ActionQuery
from x2_random_reach_plan import DEFAULT_BOUNDS, make_plan, random_candidates

PLANNING_TIMEOUT = 20.
SOURCE_FILES = ("x2_random_reach_test.py", "x2_random_reach_plan.py", "x2_random_path_audit.py",
                "x2_converge_test.py", "x2_mdi.py", "x2_sim_ros.py", "x2_arm_model.py",
                "x2_srs_ik.py", "x2_srs_batch.py", "x2_frames.py", "x2_arm_dynamics.py", "x2_ultra.urdf")


class PrecisionError(RuntimeError):
    """有效反馈未通过本轮位置/姿态精度门槛。"""


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="仅生成、审查随机计划（默认）")
    mode.add_argument("--preflight", action="store_true", help="只读检查当前机器人和起始姿态")
    mode.add_argument("--execute", action="store_true", help="执行随机点及每点的计划回程")
    ap.add_argument("--side", choices=("left", "right"), default="right")
    ap.add_argument("--count", type=int, default=8, help="原始候选点数 1..32；拒绝不补抽")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--repeats", type=int, default=1, help="每点有效往返次数 1..5；另加一次热身")
    ap.add_argument("--duration", type=float, default=8., help="每程 MoveL 秒数，至少 8")
    ap.add_argument("--settle", type=float, default=2., help="稳定与每轮闭环保持秒数，至少 2")
    ap.add_argument("--converge", type=int, default=8, help="最多闭环轮数 0..8；0 为关闭对照")
    ap.add_argument("--x-range", type=float, nargs=2, default=DEFAULT_BOUNDS[0], metavar=("MIN", "MAX"))
    ap.add_argument("--abs-y-range", type=float, nargs=2, default=DEFAULT_BOUNDS[1], metavar=("MIN", "MAX"))
    ap.add_argument("--z-range", type=float, nargs=2, default=DEFAULT_BOUNDS[2], metavar=("MIN", "MAX"))
    ap.add_argument("--plan-sha256", help="可选：核对离线计划输入标识；代码版本另用发布清单校验")
    ap.add_argument("--output", type=Path, help="新 JSON 报告路径，禁止覆盖已有文件")
    return ap


def validate_args(args):
    if not args.execute and not args.preflight:
        args.offline = True
    if not 1 <= args.repeats <= 5 or not 0 <= args.converge <= 8:
        raise ValueError("repeats 须为 1..5，converge 须为 0..8")
    for name, minimum, maximum in (("duration", 8., 30.), ("settle", 2., 5.)):
        value = getattr(args, name)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} 要求 {minimum}..{maximum} 秒的有限数")
    random_candidates(args.side, args.count, args.seed,
                      (args.x_range, args.abs_y_range, args.z_range))
    if args.plan_sha256 is not None:
        if len(args.plan_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.plan_sha256):
            raise ValueError("plan-sha256 必须为 64 位小写十六进制")


def bootstrap(argv):
    """仅在线模式准备 ROS；重启当前脚本，不能重启到旧三点测试器。"""
    import x2ik
    x2ik.load_conf()
    if os.environ.get("X2_RANDOM_REACH_ENV_READY") == "1":
        return None
    python = x2ik.ros_python()
    if python is None:
        raise RuntimeError("找不到 ROS Python；先在板卡目录运行 ./x2ik.py doctor")
    env = x2ik.ros_env(x2ik.find_sim_home(), x2ik.find_msgs_prefix())
    env["X2_RANDOM_REACH_ENV_READY"] = "1"
    return x2ik.run_with_ros([python, str(Path(__file__).resolve()), *argv], env)


def build_tasks(plan, args):
    """一个点的每次到达后都有明确的回程；热身也完整记录但不计入有效精度统计。"""
    return [dict(index=index, point=point["id"], repeat=repeat, discarded=repeat == 0,
                 leg=leg, pos=point["pos"] if leg == "target" else plan["home_pos"],
                 rot=plan["fixed_rot"], status="not_attempted")
            for index, (point, repeat, leg) in enumerate(
                (point, repeat, leg) for repeat in range(args.repeats + 1)
                for point in plan["candidates"] if point["accepted"]
                for leg in ("target", "return"))]


def summarize(plan, tasks, trials):
    target_tasks = [t for t in tasks if t["leg"] == "target" and not t["discarded"]]
    measured = [r for r in trials if r["leg"] == "target" and not r["discarded"]]
    errors = [r["result"]["pos_err"] * 1000. for r in measured if r.get("result")
              and math.isfinite(r["result"].get("pos_err", math.nan))]
    return dict(sampled_candidates=len(plan["candidates"]),
                candidate_accepted=sum(p["accepted"] for p in plan["candidates"]),
                candidate_rejected=sum(not p["accepted"] for p in plan["candidates"]),
                planned_legs=len(tasks), attempted_legs=len(trials),
                unattempted_legs=len(tasks) - len(trials),
                valid_target_planned=len(target_tasks), valid_target_attempted=len(measured),
                valid_target_passed=sum(r.get("precision_pass") is True for r in measured),
                valid_target_failed=sum(r.get("precision_pass") is not True for r in measured),
                valid_target_unattempted=len(target_tasks) - len(measured),
                valid_target_pass_rate=(sum(r.get("precision_pass") is True for r in measured)
                                        / len(target_tasks) if target_tasks else None),
                position_mm=dict(n=len(errors), median=float(np.median(errors)) if errors else None,
                                 maximum=max(errors) if errors else None))


def start_diagnostics(cli, ros, side):
    """Capture the complete start gate evidence, including when the gate rejects."""
    q = np.asarray(cli.q(side), float).copy()
    home_q = np.asarray(ros.HOME_Q, float).copy()
    home_pos, home_rot = cli.models[side].forward_kinematics(home_q)
    actual_pos, actual_rot = cli.models[side].forward_kinematics(q)
    rotvec = np.asarray(ros.log3(home_rot.T @ actual_rot), float)
    return {
        "side": side,
        "q_rad": q,
        "home_q_rad": home_q,
        "q_delta_deg": np.degrees(q - home_q),
        "q_max_abs_deg": float(np.degrees(np.max(np.abs(q - home_q)))),
        "home_pos_m": home_pos,
        "actual_pos_m": actual_pos,
        "position_delta_m": actual_pos - home_pos,
        "wrist_rotvec_rad": rotvec,
        "wrist_orientation_error_deg": float(np.degrees(np.linalg.norm(rotvec))),
        "state_count": int(cli.state_count),
        "imu_count": {name: int(value) for name, value in cli.imu_count.items()},
        "captured_at": datetime.now().astimezone().isoformat(),
    }


def check_start(cli, ros, side):
    base.check_feedback(cli, ros)
    check_gravity(cli)
    diagnostic = start_diagnostics(cli, ros, side)
    cli._random_reach_start_diagnostics = diagnostic
    q = np.asarray(diagnostic["q_rad"], float)
    if np.max(np.abs(q - ros.HOME_Q)) > math.radians(3.):
        raise RuntimeError(
            f"测试臂距 HOME 超过 3°（最大 {diagnostic['q_max_abs_deg']:.3f}°），拒绝自动回位；先用现有操作入口定位")
    if diagnostic["wrist_orientation_error_deg"] > .5:
        raise RuntimeError(
            f"起始腕姿态偏离 HOME {diagnostic['wrist_orientation_error_deg']:.3f}°（门槛 0.500°），不启动随机运动")


def check_gravity(cli):
    """验证实际 IMU/腰角数值，不能仅凭 has_orientation 布尔值证明重力有效。"""
    frame = cli._imu_last.get("pelvis")
    quat = np.asarray(frame[0], float) if frame is not None and frame[0] is not None else np.array([])
    waist = np.asarray(cli.q_waist, float)
    stiffness = np.asarray(cli.k_eff, float)
    if (quat.shape != (4,) or not np.all(np.isfinite(quat))
            or not math.isfinite(float(np.linalg.norm(quat))) or np.linalg.norm(quat) < 1e-6
            or waist.shape != (3,) or not np.all(np.isfinite(waist))):
        raise RuntimeError("pelvis 四元数或腰角非有限/无效，拒绝重力补偿发令")
    if (stiffness.shape != (7,) or not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0)
            or not math.isfinite(cli.bias_limit) or not 0 <= cli.bias_limit <= math.radians(12.)):
        raise RuntimeError("实际重力补偿参数无效")
    gravity = np.asarray(cli.refresh_gravity(), float)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)) or not 8. <= np.linalg.norm(gravity) <= 11.:
        raise RuntimeError("计算得到的重力向量无效，拒绝发送")
    if cli.grav.have_fix is not True or cli.grav.last_reject:
        raise RuntimeError("重力估计未采纳当前有效姿态，拒绝沿用默认值或旧值")


class PlanningJob:
    """工作线程仅持有输入副本及独立模型；卡住的计算不阻碍主程序退出。"""
    def __init__(self, side, q, pos, rot, duration):
        values = side, np.array(q, copy=True), np.array(pos, copy=True), np.array(rot, copy=True), duration
        self.future = Future()

        def calculate():
            if not self.future.set_running_or_notify_cancel():
                return
            try:
                from x2_random_path_audit import audit_segment
                from x2_arm_model import ArmModel
                from x2_srs_ik import SrsArmIK
                selected, start, position, rotation, seconds = values
                model = ArmModel(selected)
                result = audit_segment(model, SrsArmIK(model), start, position, rotation, seconds)
            except BaseException as exc:
                self.future.set_exception(exc)
            else:
                self.future.set_result(result)

        self.thread = threading.Thread(target=calculate, name="random-reach-path", daemon=True)
        self.thread.start()

    def close(self):
        self.future.cancel()


class ReachMonitor(base.MotionMonitor):
    def __init__(self, cli, ros, side):
        super().__init__(cli, ros, side)
        self.raw = {s: cli.q(s).copy() for s in ("left", "right")}
        self.failed = False
        self.query = None
        self.last_path_summary = None
        original_send = self.original_send

        def final_send(left, right, dl=None, dr=None):
            # 父监视器的反馈/图检查也可能耗时，底层发布前再检查一次。
            check_gravity(self.cli)
            if self.failed or (self.last_send is not None and time.monotonic() - self.last_send >= .2):
                raise RuntimeError("底层发布前检查已经超时或会话失败，拒绝发送")
            return original_send(left, right, dl, dr)

        self.original_send = final_send

    def send(self, left, right, dl=None, dr=None):
        if self.failed:
            raise RuntimeError("本会话此前发送/检查失败，拒绝继续发送")
        values = {"left": np.asarray(left, float), "right": np.asarray(right, float)}
        values[self.other] = self.fixed_other
        try:
            for side, q in values.items():
                if q.shape != (7,) or not np.all(np.isfinite(q)) or not self.cli.models[side].within_limits(q):
                    raise RuntimeError(f"{side} 原始发送关节非有限或超限")
            if np.max(np.abs(values[self.side] - self.raw[self.side])) > .05:
                raise RuntimeError("相邻原始指令关节变化超过 0.05 rad")
            if np.max(np.abs(self.cli.q(self.side) - self.raw[self.side])) > math.radians(5.):
                raise RuntimeError("测试臂反馈跟踪偏差超过 5°")
            # 所有预检查完后再次检查剩余指令间隔，避免慢检查后仍发布一帧。
            if self.last_send is not None and time.monotonic() - self.last_send >= .2:
                raise RuntimeError("发送前已经达到 200 ms 指令过期线")
            super().send(values["left"], values["right"], dl, dr)
            self.raw = {s: q.copy() for s, q in values.items()}
        except BaseException:
            self.failed = True
            raise

    def hold_tick(self):
        started = time.time()
        self.send(self.raw["left"], self.raw["right"])
        self.ros._pace(self.cli, started, 1. / self.cli.rate)

    def fresh_state(self, timeout=.5):
        return self.ros._fresh_hold(self, self.side, self.raw[self.side], self.fixed_other, timeout)

    def __getattr__(self, name):
        return getattr(self.cli, name)

    def confirm_action(self):
        if self.query is None:
            self.query = ActionQuery(self.cli, self.ros.GET_ACTION_SRV)
        self.query.start()
        deadline = time.monotonic() + 3.
        while True:
            done, action = self.query.poll()
            if done:
                if not base.is_urs(action):
                    raise RuntimeError(f"动作前未确认 URS：{action!r}")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("动作前读取 URS 超时")
            self.hold_tick()

    def plan_from_feedback(self, pos, rot, duration):
        self.last_path_summary = None
        if not self.fresh_state(.5):
            raise RuntimeError("路径规划前未获得完整新反馈")
        snapshot = self.cli.q(self.side).copy()
        job = PlanningJob(self.side, snapshot, pos, rot, duration)
        deadline = time.monotonic() + PLANNING_TIMEOUT
        try:
            while not job.future.done():
                if time.monotonic() >= deadline:
                    raise RuntimeError("实际起点路径规划超过 20 秒，停止序列")
                self.hold_tick()
            result = job.future.result()
            self.last_path_summary = {key: value for key, value in result.items() if key != "trace"}
        finally:
            job.close()
        if not result["accepted"]:
            raise RuntimeError(f"实际反馈起点的路径被拒绝：{result['reason']}")
        if np.max(np.abs(self.cli.q(self.side) - snapshot)) > .01:
            raise RuntimeError("规划期间起点反馈变化超过 0.01 rad，拒绝过时路径")
        return result

    def close_query(self):
        if self.query is not None:
            query, self.query = self.query, None
            query.close()


def execute_leg(monitor, task, args, ros):
    monitor.phase = "idle"
    monitor.last_path_summary = None
    monitor.confirm_action()
    path = monitor.plan_from_feedback(task["pos"], task["rot"], args.duration)
    # 模型审查与路径计算已完成。回放同一份关节序列，不在发送循环重复全局 IK。
    monitor.phase = "approach"
    for frame in path["trace"]:
        started = time.time()
        inputs = dict(monitor.raw)
        inputs[monitor.side] = np.asarray(frame["q_rad"], float)
        monitor.send(inputs["left"], inputs["right"])
        ros._pace(monitor.cli, started, 1. / monitor.cli.rate)
    monitor.phase = "cartesian"
    # 已到达目标附近；复用现有短 MoveL + 尾部闭环。代理 fresh_state 始终边发边等。
    result = ros.goto_cartesian(monitor, monitor.side, np.asarray(task["pos"]), np.asarray(task["rot"]),
                                2., args.settle, converge=args.converge,
                                converge_tol=.001, converge_step=math.radians(.5),
                                converge_total=math.radians(3.))
    error = base.result_error(result)
    if (result.get("stale") is not False or result.get("trajectory_valid") is not True
            or result.get("implicit_fallbacks", 0) or not math.isfinite(result.get("rot_err", math.nan))):
        error = error or "精定位反馈/轨迹无效"
    path_summary = {key: value for key, value in path.items() if key != "trace"}
    passed = (error is None and result["pos_err"] <= .001 and result["rot_err"] <= math.radians(.5)
              and (not args.converge or result.get("converged") is True))
    return result, path_summary, passed, error


def run(args, ros, plan=None):
    validate_args(args)
    output = args.output or HERE / "results" / ("random-reach-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"initializing"}\n')
    report = dict(schema="x2ik.random_reach.v1", started_at=datetime.now().astimezone().isoformat(),
                  mode="execute" if args.execute else "preflight" if args.preflight else "offline",
                  status="initializing", arguments=vars(args), trials=[], tasks=[],
                  metric="feedback_joint_fk; excludes external TCP/assembly error",
                  source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES})
    cli = monitor = gc_timing = None
    code = 0
    try:
        plan = make_plan(args, ros) if plan is None else plan
        report["plan"] = plan
        if args.plan_sha256 and args.plan_sha256 != plan["plan_sha256"]:
            raise ValueError("计划 SHA-256 与指定标识不一致，未连接机器人")
        tasks = build_tasks(plan, args)
        report["tasks"] = tasks
        print(f"计划 {plan['plan_sha256']}：候选 {len(plan['candidates'])}，"
              f"通过 {len(plan['accepted_ids'])}，去/回共 {len(tasks)} 程；"
              "固定本侧 HOME 腕姿态，不做姿态放宽。", flush=True)
        for point in plan["candidates"]:
            print(f"  {point['id']}: xyz={np.round(point['pos'], 4).tolist()} "
                  f"{point['reason']}", flush=True)
        if not tasks:
            raise ValueError("所有候选均被拒绝，没有可执行路径；保留原候选，不重抽")
        report["status"] = "offline_ready"
        if not args.offline:
            report["calibration"] = base.calibration(ros)
            cli = ros.X2ArmClient("upper_body", joint_stiffness=np.full(7, 40.),
                                  gravity_source="pelvis", bias_limit=math.radians(12.), hand_mode=1)
            cli.set_action = base.forbid_action
            cli.enter_control = base.forbid_action
            report["preflight"] = base.preflight(cli, ros)
            try:
                check_start(cli, ros, args.side)
            finally:
                if hasattr(cli, "_random_reach_start_diagnostics"):
                    report["start_diagnostics"] = cli._random_reach_start_diagnostics
            report["initial_q"] = {s: cli.q(s).copy() for s in ("left", "right")}
            report["status"] = "preflight_passed"
            print("只读预检通过：URS、SN、双臂反馈、IMU、发布独占及测试臂 HOME 起点。", flush=True)
        if args.execute:
            frames = int(math.ceil(cli.rate * (PLANNING_TIMEOUT + 3. + args.duration + 2.
                         + (args.converge + 1) * (args.settle + .5) + 1.))) + 128
            total_bytes = len(tasks) * frames * 48 * 8
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("计划记录缓冲超过 256 MiB，请减小 count/repeats/duration")
            buffers = [base.FrameBuffer(frames) for _ in tasks]
            base.save_report(output, report)
            # 分配/落盘发生在首次运动前，之后重新只读确认实际起点和状态。
            report["execution_preflight"] = base.preflight(cli, ros)
            try:
                check_start(cli, ros, args.side)
            finally:
                if hasattr(cli, "_random_reach_start_diagnostics"):
                    report["execution_start_diagnostics"] = cli._random_reach_start_diagnostics
            monitor = ReachMonitor(cli, ros, args.side)
            report["fixed_other_input"] = monitor.fixed_other.copy()
            gc_timing = base.GcTiming(monitor.started_at)
            gc_timing.start()
            report["status"] = "running"
            for task, buffer in zip(tasks, buffers):
                task["status"] = "attempted"
                row = dict(task, frames=buffer, initial_state_count=cli.state_count)
                report["trials"].append(row)
                monitor.rows = buffer
                result, path, passed, error = execute_leg(monitor, task, args, ros)
                row.update(result=result, actual_path=path, precision_pass=passed,
                           final_state_count=cli.state_count, max_send_gap_s=monitor.max_gap)
                task["status"] = row["status"] = "passed" if passed else "precision_not_met"
                print(f"{task['point']} #{task['repeat']} {task['leg']} "
                      f"{'热身' if task['discarded'] else '有效'} "
                      f"pos={result['pos_err'] * 1000:.3f}mm rot={math.degrees(result['rot_err']):.3f}deg",
                      flush=True)
                if error:
                    row["error"] = error
                    raise RuntimeError(error)
                if args.converge and not passed:
                    raise PrecisionError("闭环位置/姿态未达标，停止后续任务；不追加回程")
                if result["rot_err"] > math.radians(.5) or result["pos_err"] > .01:
                    raise PrecisionError("关闭闭环对照也超出 10 mm / 0.5°继续范围，停止序列")
            report["status"] = "completed"
            if any(not row["precision_pass"] for row in report["trials"] if not row["discarded"]):
                report["status"], code = "precision_not_met", 3
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 3 if isinstance(exc, PrecisionError) else 1
        report.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        if report["trials"]:
            row = report["trials"][-1]
            row.update(error=f"{type(exc).__name__}: {exc}", status="failed", precision_pass=False)
            if monitor is not None and monitor.last_path_summary is not None:
                row.setdefault("actual_path", monitor.last_path_summary)
            report["tasks"][row["index"]]["status"] = "failed"
        print(f"[停止] {exc}；不追加运动或状态切换。", file=sys.stderr, flush=True)
    finally:
        if cli is not None and hasattr(cli, "_random_reach_start_diagnostics"):
            report.setdefault("start_diagnostics", cli._random_reach_start_diagnostics)
        if gc_timing is not None:
            gc_timing.stop()
            report["gc_timing"] = gc_timing.report()
        if monitor is not None:
            report["max_send_gap_s"] = monitor.max_gap
        if cli is not None:
            # 此后不再发指令。收尾只读状态；失败也继续释放原节点与 ROS。
            try:
                report["final_action"] = cli.get_action()
                if not base.is_urs(report["final_action"]):
                    raise RuntimeError("结束状态未确认 URS")
            except Exception as exc:
                report["final_action_error"] = str(exc)
                code = code or 1
            for name, cleanup in (("query", monitor.close_query if monitor else lambda: None),
                                  ("client", cli.close),
                                  ("ros", lambda: cli.rclpy.shutdown() if cli.rclpy.ok() else None)):
                try:
                    cleanup()
                except Exception as exc:
                    report.setdefault("cleanup_errors", {})[name] = str(exc)
                    code = code or 1
        if plan is not None:
            report["summary"] = summarize(plan, report["tasks"], report["trials"])
        if code and report["status"] in ("completed", "preflight_passed"):
            report["status"] = "finalization_failed"
        report["exit_code"] = code
        report["finished_at"] = datetime.now().astimezone().isoformat()
        base.save_report(output, report)
        print(f"报告：{output}，退出码 {code}", flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    args = ap.parse_args(argv)
    try:
        validate_args(args)
        if args.output is not None and args.output.exists():
            raise ValueError(f"报告已存在，拒绝覆盖：{args.output}")
        if not args.offline:
            status = bootstrap(argv)
            if status is not None:
                return status
        import x2_sim_ros as ros
        return run(args, ros)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[拒绝执行] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
