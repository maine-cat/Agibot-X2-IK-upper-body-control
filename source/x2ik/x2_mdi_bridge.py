"""Desktop MDI over an authenticated SSH stdin/stdout stream; no listening sockets.

Only the owning thread spins ROS or sends commands. The input thread renews the
lease and handles cancellation even while a synchronous trajectory is running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from .x2_arm_model import ArmModel, matrix_to_rpy, rpy_to_matrix
from .x2_frames import HOME_Q
from .x2_compensation import fixed_compensation
from .x2_mdi import Session, ActionQuery, is_urs, _solve_pose
from .x2_srs_ik import SrsArmIK
from .x2_tcp import TCP_MODES, load_tcp_tools

MODES = {"xyz": 3, "pose": 6, "d": 3, "R": 3, "t": 3, "rpy": 3, "j": 7}
LEASE_SECONDS = 2.0
URS_REQUIRED = "仅支持已处于 URS 的机器人；请先使用机器人原有操作方式切换到 URS，MDI 不提供模式切换"


class Cancelled(RuntimeError):
    pass


def validate_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("id"), (str, int)):
        raise ValueError("request requires an id")
    op = request.get("op")
    if any(key in request for key in ("tcp_mode", "tcp_file", "tcp_config")):
        raise ValueError("TCP 工具只能在连接前选择；请断开后重新连接")
    if op == "action":
        raise ValueError(URS_REQUIRED)
    if op not in ("heartbeat", "arm", "disarm", "mdi", "home", "close", "state"):
        raise ValueError("unknown operation")
    if op in ("mdi", "home"):
        for key, default, minimum in (("duration", 8., .2), ("settle", 2., 0.)):
            value = request.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or value > 120:
                raise ValueError(f"{key} must be finite and in [{minimum}, 120] seconds")
        if not isinstance(request.get("preview", False), bool):
            raise ValueError("preview must be boolean")
    if op == "mdi":
        mode, values = request.get("mode"), request.get("values")
        if mode not in MODES or request.get("side") not in ("left", "right"):
            raise ValueError("invalid MDI mode or side")
        if (not isinstance(values, list) or len(values) != MODES[mode]
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            raise ValueError("invalid number of finite MDI values")
    return request


class Transport:
    """Bounded input and independent heartbeat; never buffer future motion."""
    def __init__(self, reader, output):
        self.reader, self.output = reader, output
        self.commands = queue.Queue(maxsize=1)
        self.cancel = threading.Event()
        self.eof = threading.Event()
        self.busy = False
        self.last_heartbeat = time.monotonic()
        self.lock = threading.Lock()
        self.writer_lock = threading.Lock()

    def emit(self, value):
        try:
            with self.writer_lock:
                self.output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                self.output.flush()
        except (BrokenPipeError, OSError):
            self.cancel.set()
            self.eof.set()
            raise Cancelled("SSH output disconnected")

    def result(self, request, ok, message, **extra):
        self.emit(dict(type="result", id=request.get("id"), ok=ok, message=message, **extra))

    def read(self):
        try:
            while True:
                line = self.reader.readline(65537)
                if not line:
                    break
                request = {}
                try:
                    if len(line) > 65536:
                        raise ValueError("request exceeds 64 KiB")
                    request = validate_request(json.loads(line))
                    op = request["op"]
                    if op == "heartbeat":
                        self.last_heartbeat = time.monotonic()
                        continue
                    if op in ("disarm", "close"):
                        self.cancel.set()
                        if op == "close":
                            self.eof.set()
                        self.result(request, True, "停止发送请求已收到；不会自动 HOME 或切换状态")
                        continue
                    with self.lock:
                        if self.busy or not self.commands.empty():
                            self.result(request, False, "控制器忙；该指令已丢弃，不排队执行")
                        else:
                            self.commands.put_nowait(request)
                except (ValueError, TypeError, queue.Full) as exc:
                    self.result(request if isinstance(request, dict) else {}, False, str(exc))
        except (OSError, Cancelled):
            pass
        finally:
            self.cancel.set()
            self.eof.set()

    def check_lease(self):
        if self.cancel.is_set() or self.eof.is_set():
            raise Cancelled("已停止发送或 SSH 连接关闭")
        if time.monotonic() - self.last_heartbeat > LEASE_SECONDS:
            raise Cancelled("桌面心跳超过 2 秒，停止发送")


class Bridge:
    def __init__(self, transport, demo=False, tcp_mode="none", tcp_file=None):
        self.transport, self.demo = transport, demo
        self.armed = self.busy = False
        self.session = self.cli = self.query = None
        self.action = "UPPERBODY_REMOTE_SPLIT" if demo else None
        self.action_at = time.monotonic() if demo else 0.
        self.last_state = 0.
        self.next_action_query = 0.
        self.feedback_count, self.feedback_at = -1, 0.
        self.tcp_tools = load_tcp_tools(tcp_mode, tcp_file)
        self.tcp_mode = tcp_mode
        self.models = {s: ArmModel(s, tcp_offset=t["translation_m"],
                                   tcp_rotation=t["rotation_matrix"])
                       for s, t in self.tcp_tools.items()}
        self.iks = {s: SrsArmIK(self.models[s]) for s in self.models}
        self.demo_q = {s: HOME_Q.copy() for s in self.models}
        if not demo:
            self._connect()

    def _connect(self):
        from . import x2_sim_ros as ros
        self.ros = ros
        cfg = fixed_compensation()
        self.cli = ros.X2ArmClient("upper_body", read_only=True, verbose=False,
                                  joint_stiffness=np.full(7, cfg["stiffness"]),
                                  bias_limit=math.radians(cfg["bias_limit_deg"]),
                                  gravity_source=cfg["gravity_source"])
        # Keep the client's original wrist-based dynamics/compensation models.
        self.cli.models, self.cli.iks = self.models, self.iks
        self.query = ActionQuery(self.cli, ros.GET_ACTION_SRV)
        original_spin = self.cli.spin
        def observable_spin(seconds):
            # Session initialization includes a 3s discovery wait. Keep desktop
            # feedback current and cancellation responsive during that wait.
            remaining = max(0., seconds)
            while remaining > 1e-9:
                if self.armed:
                    self.transport.check_lease()
                step = min(.05, remaining)
                original_spin(step)
                self.poll_action()
                self.publish_state()
                remaining -= step
            if self.armed:
                self.transport.check_lease()
        self.cli.spin = observable_spin
        original_send = self.cli.send
        def guarded_send(*args, **kwargs):
            self.transport.check_lease()
            if not self.armed:
                raise Cancelled("桌面发送未启用")
            self.poll_action()
            if not self.urs_confirmed():
                raise Cancelled("URS 状态未确认或状态反馈过期")
            self.publish_state()
            self.transport.check_lease()
            return original_send(*args, **kwargs)
        self.cli.send = guarded_send

    def poll_action(self):
        if self.demo:
            self.action_at = time.monotonic()
            return
        if time.monotonic() < self.next_action_query:
            return
        done, action = self.query.poll()
        if done:
            self.action, self.action_at = action, time.monotonic()
            self.query.start()
            self.next_action_query = time.monotonic() + .5

    def q(self, side):
        return self.demo_q[side] if self.demo else self.cli.q(side)

    def urs_confirmed(self):
        return is_urs(self.action) and time.monotonic() - self.action_at <= 3.

    def fresh(self):
        if self.demo:
            return True
        if self.cli.state_count != self.feedback_count:
            self.feedback_count, self.feedback_at = self.cli.state_count, time.monotonic()
        return bool(self.cli.state_count and self.ros._complete_arm_feedback(self.cli)
                    and time.monotonic() - self.feedback_at < .2)

    def snapshot(self):
        fresh = self.fresh()
        arms = {}
        if fresh:
            for side, model in self.models.items():
                q = self.q(side)
                pos, rot = model.forward_kinematics(q)
                points, rotations = model.joint_frames(q)
                arms[side] = dict(q_deg=np.degrees(q).tolist(), xyz=pos.tolist(),
                                  rpy_deg=np.degrees(matrix_to_rpy(rot)).tolist(),
                                  points=[p.tolist() for p in points] + [pos.tolist()],
                                  tcp=self.tcp_tools[side],
                                  link_transforms=[dict(xyz=p.tolist(), rotation=r.tolist())
                                                   for p, r in zip(points, rotations)])
        return dict(type="state", demo=self.demo, connected=True, armed=self.armed,
                    busy=self.busy, action=self.action, fresh=fresh,
                    tcp_mode=self.tcp_mode,
                    command_endpoint_owned=False if self.cli is None else getattr(self.cli, 'pub', None) is not None,
                    urs_confirmed=self.urs_confirmed(),
                    arms=arms)

    def publish_state(self, force=False):
        now = time.monotonic()
        if force or now - self.last_state >= .2:
            self.transport.emit(self.snapshot())
            self.last_state = now

    def disarm(self):
        self.armed = False
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.cli is not None:
            self.cli.disable_commands()

    def require_motion(self):
        self.transport.check_lease()
        if not self.armed:
            raise ValueError("先在桌面显式启用发送")
        if not self.fresh():
            raise ValueError("无新鲜完整双臂反馈")
        if not self.urs_confirmed():
            raise ValueError(URS_REQUIRED)
        if not self.demo and self.session is None:
            self.cli.enable_commands()
            args = SimpleNamespace(side="right", duration=8., settle=2., relax=0., dry=False,
                                   mode="upper_body", payload=0.)
            self.session = Session(self.cli, args, self.ros)
            self.session.initialize()
            self.transport.check_lease()
            # initialize can take several seconds but never publishes; refresh action now.
            self.action = self.cli.get_action(timeout=1.)
            self.action_at = time.monotonic()
            if not is_urs(self.action):
                raise ValueError("MDI 初始化后状态不是 URS")

    def plan(self, req):
        side, mode = req["side"], req["mode"]
        values = np.asarray(req["values"], float)
        q = self.q(side)
        if mode == "j":
            target = np.radians(values)
            if not self.models[side].within_limits(target):
                raise ValueError("关节目标超限")
            return dict(q=target, joint=True)
        pos, rot = self.models[side].forward_kinematics(q)
        if mode == "xyz":
            pos = values
        elif mode == "pose":
            pos, rot = values[:3], rpy_to_matrix(np.radians(values[3:]))
        elif mode == "d":
            pos = pos + values
        elif mode == "R":
            rot = rpy_to_matrix(np.radians(values)) @ rot
        elif mode == "t":
            rot = rot @ rpy_to_matrix(np.radians(values))
        elif mode == "rpy":
            rot = rpy_to_matrix(np.radians(values))
        if self.session is not None:
            plan = self.session.plan_pose(side, pos, rot, False)
        else:
            plan = _solve_pose(self.models[side], self.iks[side], q, pos, rot, 0., False)
        if not plan["ok"] or plan["clipped"]:
            raise ValueError("目标 IK 预检失败（不放宽姿态、不裁剪目标）")
        return dict(q=plan["q"], pos=pos, rot=plan["rot"], joint=False)

    def execute(self, req):
        validate_request(req)
        op = req["op"]
        if op == "state":
            self.publish_state(True)
            return "状态已更新"
        if op == "arm":
            self.transport.check_lease()
            if not self.fresh():
                raise ValueError("无新鲜完整双臂反馈")
            if not self.urs_confirmed():
                raise ValueError(URS_REQUIRED)
            self.armed = True
            return "已启用发送；仅显式下发指令才开始运动"
        preview = req.get("preview", False)
        if op in ("mdi", "home"):
            if not self.fresh():
                raise ValueError("无新鲜完整双臂反馈，不能规划")
            if preview and self.session is not None:
                # Readonly preview must not leave a hidden hold publisher active.
                self.disarm()
            if not preview:
                self.require_motion()
            if op == "home":
                goals = {s: HOME_Q.copy() for s in self.models}
                for s, q in goals.items():
                    if not self.models[s].within_limits(q):
                        raise ValueError("HOME 超限")
                plan = None
            else:
                plan = self.plan(req)
                goals = {req["side"]: plan["q"]}
            if preview:
                return "预检通过；未发送运动（预检不含碰撞检测）"
            self.transport.check_lease()
            if self.demo:
                self.demo_q.update(goals)
                return "离线演示完成；未连接机器人"
            self.session.duration = req.get("duration", 8.)
            self.session.settle = req.get("settle", 2.)
            if plan is None or plan["joint"]:
                self.session.move_joint(goals, side=req["side"] if op == "mdi" else None)
            else:
                self.session.move_cartesian(req["side"], plan["pos"], plan["rot"])
            return "指令完成；会话内持续保持，关闭发送则停止发布"
        raise ValueError("unsupported request")

    def run(self):
        threading.Thread(target=self.transport.read, name="mdi-ssh-input", daemon=True).start()
        self.publish_state(True)
        try:
            while not self.transport.eof.is_set():
                if self.transport.cancel.is_set():
                    self.disarm()
                    # Invalidate any accepted but not started request before re-enabling.
                    while not self.transport.commands.empty():
                        req = self.transport.commands.get_nowait()
                        self.transport.result(req, False, "已停止发送；待执行请求已丢弃")
                    self.transport.cancel.clear()
                try:
                    if self.armed:
                        self.transport.check_lease()
                    if not self.demo:
                        if self.session is not None:
                            self.session.hold_tick()
                        else:
                            self.cli.spin(.02)
                    else:
                        time.sleep(.02)
                    self.poll_action()
                    if self.armed and not self.urs_confirmed():
                        raise Cancelled("URS 状态未确认或已过期，停止发送；请在机器人外部操作界面确认模式")
                    self.publish_state()
                    with self.transport.lock:
                        try:
                            req = self.transport.commands.get_nowait()
                        except queue.Empty:
                            continue
                        self.transport.busy = self.busy = True
                    self.publish_state(True)
                    try:
                        message = self.execute(req)
                        self.transport.result(req, True, message)
                    except Exception as exc:
                        self.disarm()
                        self.transport.result(req, False, str(exc))
                    finally:
                        with self.transport.lock:
                            self.transport.busy = self.busy = False
                        self.publish_state(True)
                except (Cancelled, RuntimeError) as exc:
                    self.disarm()
                    self.transport.emit(dict(type="event", message=str(exc)))
                    self.publish_state(True)
        finally:
            self.disarm()
            if self.query is not None:
                self.query.close()
            if self.cli is not None:
                self.cli.close()
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--tcp-mode", choices=TCP_MODES, default="none")
    parser.add_argument("--tcp-file", help="custom 模式的机器人本地 TCP 标定 JSON")
    args = parser.parse_args(argv)
    original_stdout = sys.stdout
    output = original_stdout
    # Native ROS/DDS libraries write directly to fd 1, bypassing sys.stdout.
    # Reserve a duplicate for JSONL, then route fd 1 to diagnostics until exit
    # (DDS static destructors can still log after main returns).
    try:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        pass  # In-memory streams used by embedded/offline tests.
    else:
        sys.stdout.flush()
        output = os.fdopen(os.dup(stdout_fd), 'w', encoding='utf-8', buffering=1)
        os.dup2(stderr_fd, stdout_fd)
    sys.stdout = sys.stderr  # Legacy helper diagnostics must never corrupt JSONL.
    try:
        return Bridge(Transport(sys.stdin, output), demo=args.demo,
                      tcp_mode=args.tcp_mode, tcp_file=args.tcp_file).run()
    except Exception as exc:
        print(f"MDI bridge stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        sys.stdout = original_stdout
        if output is not original_stdout:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
