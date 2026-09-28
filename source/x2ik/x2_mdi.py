"""MDI 前端：借用现有 ROS 客户端，主线程独占发送，后台只处理输入和纯 IK。"""
from __future__ import annotations

import copy
import math
import queue
import re
import sys
import threading
import time
from concurrent.futures import Future

import numpy as np

EOF = object()


class InputError(ValueError):
    """可在界面内纠正的输入错误；执行中的 ValueError 不属于此类。"""


class SessionError(RuntimeError):
    """终止整个会话的运行故障。"""


def validate_args(args):
    if getattr(args, "mode", "upper_body") != "upper_body":
        raise ValueError("MDI 只允许 upper_body 接口")
    for name, allow_zero in (("duration", False), ("settle", True)):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"--{name} 必须是有限{'非负' if allow_zero else '正'}数")
    relax = getattr(args, "relax", 90.)
    if not math.isfinite(relax) or not 0 <= relax <= 180:
        raise ValueError("--relax 必须是 0 到 180 之间的有限数")
    payload = getattr(args, "payload", 0.)
    if not math.isfinite(payload) or payload < 0:
        raise ValueError("--payload 必须是有限非负数")


def is_urs(action):
    return isinstance(action, str) and re.fullmatch(
        r"(?:URS|US|UPPERBODY_REMOTE_SPLIT)(?:\(\d+\))?", action.strip()) is not None


class InputLines:
    """readline 仅在输入线程执行，部分管道行不会阻塞 ROS 主线程。"""
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.lines = queue.Queue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, name="mdi-input", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stopped.is_set():
                line = self.stream.readline()
                self.lines.put(EOF if line == "" else line)
                if line == "":
                    break
        except Exception as exc:
            self.lines.put(exc)

    def poll(self):
        try:
            value = self.lines.get_nowait()
        except queue.Empty:
            return None
        if isinstance(value, Exception):
            raise SessionError(f"读取输入失败：{value}") from value
        return value

    def close(self):
        self.stopped.set()  # 不关闭共享 stdin，也不等待可能仍在等输入的 daemon。


class ActionQuery:
    """GetMcAction 异步请求；poll 本身不 spin、不等待服务、不发布运动。"""
    def __init__(self, cli, service_name):
        from aimdk_msgs.srv import GetMcAction
        self.cli, self.service_type = cli, GetMcAction
        self.service = cli.node.create_client(GetMcAction, service_name)
        self.future = None

    def start(self):
        self.future = None

    def poll(self):
        if self.future is None:
            if not self.service.service_is_ready():
                return False, None
            request = self.service_type.Request()
            stamp = self.cli.node.get_clock().now().to_msg()
            if hasattr(request, "request"):
                request.request.header.stamp = stamp
            elif hasattr(request, "header"):
                request.header.stamp = stamp
            self.future = self.service.call_async(request)
        if not self.future.done():
            return False, None
        response = self.future.result()
        info = getattr(response, "info", None)
        if info is None:
            return True, None
        description = getattr(info, "action_desc", "")
        if description:
            return True, description
        current = getattr(info, "current_action", None)
        return True, (self.cli._action_name(getattr(current, "value", current))
                      if current is not None else None)

    def close(self):
        if self.future is not None and not self.future.done():
            self.future.cancel()
        if self.service is not None:
            service, self.service = self.service, None
            self.cli.node.destroy_client(service)


def _solve_pose(model, ik, q_seed, pos, rot, relax_deg, allow_relax):
    """纯模型计算；参数不包含 ROS 客户端，不能发送或处理 ROS 回调。"""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    projected, clipped = ik.project_to_workspace(pos, rot)
    # 肩腕距离只是近似筛选，左腕非共点时不能据此提前拒绝真实可达目标。
    # 求解始终使用原始目标，只有通过完整 FK 复核的解才能解除预警。
    sol = ik.solve(pos, rot, q_seed=q_seed)
    if sol is not None:
        projected, clipped = pos, False
    relaxation = None
    if sol is None and allow_relax and relax_deg > 0:
        relaxed = ik.solve_hold_rotation(pos, rot, q_seed=q_seed,
                                         max_relax=math.radians(relax_deg))
        if relaxed is not None:
            rot = relaxed.rot
            projected, clipped = ik.project_to_workspace(pos, rot)
            sol = relaxed.sol if not clipped else None
            if relaxed.axis is not None:
                relaxation = dict(axis=relaxed.axis, frame=relaxed.frame,
                                  deviation_deg=math.degrees(relaxed.deviation))
    valid = sol is not None and not clipped
    if valid:
        valid = np.all(np.isfinite(sol.q)) and model.within_limits(sol.q)
    return dict(ok=bool(valid), clipped=bool(clipped),
                clip_mm=float(np.linalg.norm(projected - pos)) * 1000.,
                q=None if sol is None else sol.q.copy(),
                margin=0. if sol is None else float(model.limit_margin(sol.q)),
                pos_err=0. if sol is None else float(sol.pos_error),
                rot_err=0. if sol is None else float(sol.rot_error),
                rot=rot.copy(), relaxation=relaxation)


class PureIKWorker:
    def __init__(self, models, solver_type):
        self.models = copy.deepcopy(models)
        self.iks = {side: solver_type(model) for side, model in self.models.items()}
        self.jobs = queue.Queue()
        self.stopped = threading.Event()
        # ThreadPoolExecutor 的非 daemon 线程会在解释器退出时 join 正在运行的 IK。
        # 本线程只持有复制模型；关闭会话后即使求解尚未返回，也不阻止进程退出。
        self.thread = threading.Thread(target=self._work, name="mdi-pure-ik", daemon=True)
        self.thread.start()

    def _work(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            future, values = job
            if self.stopped.is_set():
                future.cancel()
                continue
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = _solve_pose(*values)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def submit(self, side, q_seed, pos, rot, relax_deg, allow_relax):
        if self.stopped.is_set():
            raise SessionError("IK 工作线程已关闭")
        future = Future()
        values = (self.models[side], self.iks[side], np.array(q_seed, copy=True),
                  np.array(pos, copy=True), np.array(rot, copy=True), relax_deg, allow_relax)
        self.jobs.put((future, values))
        return future

    def close(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job[0].cancel()
        self.jobs.put(None)


class Session:
    """借用 cli；本类从不创建第二个在线节点，也不负责关闭 cli。"""
    def __init__(self, cli, args, ros):
        from .x2_api import X2Arm
        self.cli, self.args, self.ros = cli, args, ros
        self.side = args.side
        self.duration, self.settle = args.duration, args.settle
        self.relax_deg = float(getattr(args, "relax", 90.))
        self.dry = bool(getattr(args, "dry", False))
        self.arm = X2Arm(args.side, connect=False)
        self.arm.cli, self.arm._ros = cli, ros
        self.arm.model, self.arm.solver = cli.models[args.side], cli.iks[args.side]
        self.arm.connection_config = {"gravity_source": cli.grav.source}
        self.worker = PureIKWorker(cli.models, ros.SrsArmIK)
        self.proxy = self.action_query = None
        self.ready = False
        self.feedback_seen = cli.state_count
        self.feedback_at = time.monotonic()

    def initialize(self):
        validate_args(self.args)
        if self.cli.mode != "upper_body":
            raise SessionError("MDI 只允许 upper_body")
        stiffness = np.asarray(self.cli.k_eff, float)
        if (stiffness.shape != (7,) or not np.all(np.isfinite(stiffness))
                or np.any(stiffness <= 0)):
            raise SessionError("MDI 刚度必须是 7 个有限正数")
        if not math.isfinite(self.cli.bias_limit) or not 0 <= self.cli.bias_limit <= math.pi:
            raise SessionError("MDI 重力偏置上限必须在 0 到 180 度之间")
        if self.cli.grav.source not in ("chest", "pelvis", "static"):
            raise SessionError("MDI 重力来源必须为 chest / pelvis / static")
        if not self.cli.wait_state(timeout=5.):
            raise SessionError("没有关节反馈")
        self.cli.spin(3.)  # 尚未开始发送，留出命令端点独立发现时间。
        current = self.cli.get_action(timeout=3.)
        if not is_urs(current):
            raise SessionError(f"当前 action={current!r}，仅允许已处于 URS；不切换状态")
        self.arm._check_graph()
        if not self.cli.fresh_state(.5):
            raise SessionError("初始化未获得新反馈")
        self.arm._check_feedback()
        self.action_query = ActionQuery(self.cli, self.ros.GET_ACTION_SRV)
        self._rebase()
        self.ready = True

    def _rebase(self):
        from .x2_api import _MotionClient
        self.arm._hold_inputs = {s: self.cli.q(s).copy() for s in ("left", "right")}
        self.proxy = _MotionClient(self.arm, single=False)
        self.feedback_seen, self.feedback_at = self.cli.state_count, time.monotonic()

    @property
    def raw(self):
        if self.proxy is not None and self.proxy.last_inputs is not None:
            return self.proxy.last_inputs
        return self.arm._hold_inputs

    def _save_raw(self):
        if self.proxy is not None and self.proxy.last_inputs is not None:
            self.arm._hold_inputs = {s: q.copy() for s, q in self.proxy.last_inputs.items()}

    def hold_tick(self):
        self._check_ready()
        started = time.time()
        now = time.monotonic()
        if self.cli.state_count != self.feedback_seen:
            self.feedback_seen, self.feedback_at = self.cli.state_count, now
        elif now - self.feedback_at > .2:
            raise SessionError("关节反馈超过 200 ms 未更新，停止 MDI")
        if self.dry:
            self.arm._check_feedback()
            self.arm._check_graph()
        else:
            self.proxy.single = False
            self.proxy.send(self.raw["left"], self.raw["right"])
            self._save_raw()
        self.ros._pace(self.cli, started, 1. / self.cli.rate)

    def confirm_action(self):
        self._check_ready()
        self.action_query.start()
        deadline = time.monotonic() + 3.
        while True:
            done, action = self.action_query.poll()
            if done:
                if not is_urs(action):
                    raise SessionError(f"当前 action={action!r}，停止 MDI，不切换状态")
                return
            if time.monotonic() >= deadline:
                raise SessionError("3s 内未确认 URS，停止 MDI")
            self.hold_tick()

    def set_dry(self, enabled):
        enabled = bool(enabled)
        if self.dry and not enabled:
            self.confirm_action()  # 仍在 dry 状态，等服务期间零发送。
            if not self.cli.fresh_state(.5):
                raise SessionError("退出 dry 未获得新反馈")
            self.arm._check_feedback()
            self.arm._check_graph()
            self._rebase()
            print("dry = False：已按当前新反馈重建两臂原始保持基准；不回放预览目标。", flush=True)
        elif enabled:
            print("dry = True：所有运动和等待保持均零发送。", flush=True)
        self.dry = enabled

    def plan_pose(self, side, pos, rot, allow_relax=False):
        self._check_ready()
        self.arm._check_feedback()
        future = self.worker.submit(side, self.cli.q(side).copy(), pos, rot,
                                    self.relax_deg, allow_relax)
        try:
            while not future.done():
                self.hold_tick()
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def move_joint(self, targets, side=None):
        self._check_ready()
        targets = {s: self.arm._joint_target(q, s) for s, q in targets.items()}
        if self.dry:
            print("  (dry: 仅预览关节目标，不发送)", flush=True)
            return {"dry": True}
        self.confirm_action()
        self.proxy.single, self.proxy.side = side is not None, side or self.side
        goals = {s: targets.get(s, self.raw[s]).copy() for s in ("left", "right")}
        try:
            result = self.ros.goto_joint(self.proxy, goals, self.duration, self.settle)
            if not self.proxy.fresh_state(.5):
                raise SessionError("MoveJ 结束未获得完整新反馈")
            self.arm._check_feedback()
            for selected in targets:
                q = self.cli.q(selected)
                error = q - goals[selected]
                result[selected].update(q=q, err=error, err_max=float(np.max(np.abs(error))), stale=False)
                if not math.isfinite(result[selected]["err_max"]):
                    raise SessionError("MoveJ 结束反馈无效")
            return result
        finally:
            self._save_raw()
            self.proxy.single = False

    def move_cartesian(self, side, pos, rot):
        self._check_ready()
        if self.dry:
            print("  (dry: 仅预检笛卡尔目标，不发送)", flush=True)
            return {"dry": True}
        self.confirm_action()
        self.proxy.single, self.proxy.side = True, side
        try:
            result = self.ros.goto_cartesian(self.proxy, side, pos, rot, self.duration, self.settle)
            if (result.get("stale") is not False or result.get("trajectory_valid") is not True
                    or result.get("aborted") or any(result.get(key, 0) for key in
                        ("ik_fails", "clipped", "step_rejects", "branch_rejects", "implicit_fallbacks"))):
                raise SessionError("MoveL 反馈过期或轨迹出现无解/裁剪/跳变，停止 MDI")
            if not math.isfinite(result["pos_err"]) or not math.isfinite(result["rot_err"]):
                raise SessionError("MoveL 结束误差无效")
            return result
        finally:
            self._save_raw()
            self.proxy.single = False

    def _check_ready(self):
        if not self.ready:
            raise SessionError("MDI 会话未就绪或已经关闭")
        if self.arm._send_uncertain:
            raise SessionError("此前发送结果不确定，停止 MDI；请退出后重新连接预检")

    def close(self):
        self.ready = False
        try:
            self.worker.close()
        finally:
            try:
                if self.action_query is not None:
                    query, self.action_query = self.action_query, None
                    query.close()
            finally:
                self.arm.cli = None  # 借用结束，原 CLI 的 finally 负责关闭节点。


def _numbers(tokens, count):
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise InputError("参数必须是数字") from exc
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise InputError(f"要求 {count} 个有限数")
    return values


def _show(session, side):
    session.arm._check_feedback()
    return session.ros._mdi_show(session.cli, side)


def _read_line(session, reader, prompt):
    print(prompt, end="", flush=True)
    while True:
        line = reader.poll()
        if line is not None:
            return line
        session.hold_tick()


def run(cli, args, ros_module):
    """MDI 的唯一入口；所有发送及 ROS 回调均由调用它的主线程执行。"""
    session = reader = None
    try:
        validate_args(args)
        session = Session(cli, args, ros_module)
        session.initialize()
        reader = InputLines()
        if args.home_first:
            print("启动 HOME 预览 ..." if session.dry else "先回双臂待机位 ...", flush=True)
            session.move_joint({"left": ros_module.HOME_Q, "right": ros_module.HOME_Q})
        print(f"MDI：upper_body / 当前 {session.side} / torso 系 / dry={session.dry}", flush=True)
        print(f"duration={session.duration}s  settle={session.settle}s  relax={session.relax_deg}deg", flush=True)
        print(ros_module.MDI_HELP, flush=True)
        _show(session, session.side)
        unit = 1.

        def go(pos, rot, allow_relax=False):
            plan = session.plan_pose(session.side, pos, rot, allow_relax)
            target_rot = plan["rot"]
            if plan.get("relaxation"):
                info = plan["relaxation"]
                print(f"  姿态放宽：{info['frame']} {'XYZ'[info['axis']]} 轴 "
                      f"{info['deviation_deg']:.2f} deg", flush=True)
            print(f"  -> xyz {np.round(pos, 5).tolist()} m / "
                  f"rpy {np.round(ros_module.rpy_deg(target_rot), 2).tolist()} deg", flush=True)
            if plan["clipped"]:
                print(f"  [x] 目标超出工作空间 {plan['clip_mm']:.1f} mm，拒绝执行。", flush=True)
            else:
                print(ros_module.preflight_line(plan), flush=True)
            ros_module.publish_target(cli, pos, target_rot)  # 可视化话题，不是运动话题。
            if not plan["ok"] or plan["clipped"]:
                print("  目标预检未通过，未执行运动。", flush=True)
                return
            result = session.move_cartesian(session.side, pos, target_rot)
            if not session.dry:
                print(f"  到位：反馈 FK 位置误差 {result['pos_err'] * 1000:.3f} mm / "
                      f"姿态误差 {math.degrees(result['rot_err']):.3f} deg", flush=True)

        while True:
            line = _read_line(session, reader, f"\nx2 mdi [{session.side}]> ")
            if line is EOF:
                print("\n(EOF)", flush=True)
                return 0
            tokens = line.split()
            if not tokens:
                continue
            head = tokens[0]
            try:
                if (head in ("q", "quit", "exit", "q!", "quit!", "?", "h", "help",
                             "w", "where", "mm", "dry", "home") and len(tokens) != 1):
                    raise InputError(f"{head} 不接受额外参数")
                if head in ("q", "quit", "exit"):
                    session.move_joint({"left": ros_module.HOME_Q, "right": ros_module.HOME_Q})
                    return 0
                if head in ("q!", "quit!"):
                    return 0
                if head in ("?", "h", "help"):
                    print(ros_module.MDI_HELP, flush=True)
                    continue
                if head in ("w", "where"):
                    _show(session, session.side)
                    continue
                if head == "side":
                    if len(tokens) != 2 or tokens[1] not in ("left", "right"):
                        raise InputError("用法：side left|right")
                    session.side = tokens[1]
                    _show(session, session.side)
                    continue
                if head == "mm":
                    unit = .001 if unit == 1. else 1.
                    print(f"位置输入单位：{'mm' if unit == .001 else 'm'}", flush=True)
                    continue
                if head == "dry":
                    session.set_dry(not session.dry)
                    continue
                if head in ("dur", "settle", "relax"):
                    value = _numbers(tokens[1:], 1)[0]
                    if head == "dur":
                        if value < .2:
                            raise InputError("dur 不能小于 0.2s")
                        session.duration = value
                    elif head == "settle":
                        if value < 0:
                            raise InputError("settle 不能为负")
                        session.settle = value
                    else:
                        if not 0 <= value <= 180:
                            raise InputError("relax 必须在 0 到 180deg 之间")
                        session.relax_deg = value
                    print(f"{head} = {value}", flush=True)
                    continue
                if head == "home":
                    session.move_joint({"left": ros_module.HOME_Q, "right": ros_module.HOME_Q})
                    _show(session, session.side)
                    continue
                if head == "j":
                    target = np.radians(_numbers(tokens[1:], 7))
                    if not cli.models[session.side].within_limits(target):
                        raise InputError("关节目标超出限位；未夹值执行")
                    session.move_joint({session.side: target}, side=session.side)
                    _show(session, session.side)
                    continue
                if head in ("d", "R", "t", "rpy"):
                    values = _numbers(tokens[1:], 3)
                    pos, rot = cli.models[session.side].forward_kinematics(cli.q(session.side))
                    if head == "d":
                        go(pos + np.array(values) * unit, rot, True)
                    elif head == "R":
                        go(pos, ros_module.rpy_to_matrix(np.radians(values)) @ rot)
                    elif head == "t":
                        go(pos, rot @ ros_module.rpy_to_matrix(np.radians(values)))
                    else:
                        go(pos, ros_module.rpy_to_matrix(np.radians(values)))
                    continue
                values = _numbers(tokens, len(tokens))
                if len(values) == 3:
                    _, rot = cli.models[session.side].forward_kinematics(cli.q(session.side))
                    go(np.array(values) * unit, rot, True)
                elif len(values) == 6:
                    go(np.array(values[:3]) * unit, ros_module.rpy_to_matrix(np.radians(values[3:])))
                else:
                    raise InputError("要求 xyz 或 xyz rx ry rz；敲 ? 查看帮助")
            except InputError as exc:
                print(f"  输入被拒绝：{exc}", flush=True)
    except KeyboardInterrupt:
        print("\nMDI 已中断；退出会话，不执行恢复动作。", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[MDI 停止] {exc}；未执行恢复动作或状态切换。", file=sys.stderr, flush=True)
        return 1
    finally:
        if reader is not None:
            reader.close()
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                print(f"[MDI 清理] {exc}", file=sys.stderr, flush=True)
