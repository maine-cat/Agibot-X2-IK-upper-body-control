#!/usr/bin/env python3
"""X2 上肢四接口封装:MoveJ / MoveL / 正运动学 / 逆运动学。

一个连接控制两臂；显式左右 MoveJ 入口不受构造时默认侧影响:

    from x2_api import HOME, X2Arm

    with X2Arm() as robot:
        robot.R_move_J(HOME, duration=8, settle=2)  # 只让右臂到目标
        robot.L_move_J(HOME, duration=8, settle=2)  # 上一步完成后再动左臂
        robot.move_j_both(q_left=HOME, q_right=HOME,
                          duration=8, settle=2)   # 同一轨迹同步控制两臂

默认侧的 MoveJ / MoveL / FK / IK 调用仍兼容:

    from x2_api import HOME, X2Arm

    with X2Arm("right") as arm:     # 只读确认机器人已在 URS，不切换状态
        result = arm.move_j(HOME, duration=8, settle=2)
        pos, rpy = arm.fk(HOME)
        result = arm.move_l(pos, rpy, duration=2, settle=2, converge=8)
        print(result["converged"], result["pos_err"])

正/逆运动学是纯计算,**不需要机器人也不需要 ROS**:

    arm = X2Arm("right", connect=False)
    q = arm.ik([0.35, -0.25, -0.10])
    pos, rpy = arm.fk(q)

单位一律 **rad / m / s**。姿态用 rpy(torso 系固定轴 X-Y-Z),
和 URDF 的 `<origin rpy>` 同一约定。要角度的话自己 np.degrees。

这一层不做新算法,只是把 x2_sim_ros 的 goto_joint / goto_cartesian 和
x2_arm_model / x2_srs_ik 收成四个名字。原理见 INTRO.md。
MDI 操作前端同时保留，入口 ./x2ik.py mdi，说明见 MDI_GUIDE.md。
人工前端与开放接口共用上肢话题，同一时刻只能由一个控制入口发送。
"""

from __future__ import annotations

import math
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from x2_arm_model import ArmModel, matrix_to_rpy, rpy_to_matrix
from x2_srs_ik import SrsArmIK
from x2_frames import HOME_Q
from x2_compensation import fixed_compensation

#: 待机位关节角 (rad),7 个。move_j 的默认目标。
HOME = HOME_Q.copy()

#: 关节顺序。下发消费的是顺序,不是名字 —— 这是唯一的语义约定。
JOINT_ORDER = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_yaw", "wrist_pitch", "wrist_roll")


def _vector(value, size, name):
    result = np.asarray(value, float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是 {size} 个有限数")
    return result.copy()


def _timing(duration, settle):
    if (not math.isfinite(duration) or duration <= 0
            or not math.isfinite(settle) or settle < 0):
        raise ValueError("duration 必须为有限正数，settle 必须为有限非负数")


class _MotionClient:
    """API 专用发送闸；不改变 CLI 默认路径或真实反馈读取语义。"""

    def __init__(self, arm, single, side=None):
        self.arm = arm
        self.client = arm.cli
        self.single = single
        self.side = arm.side if side is None else side
        self.last_inputs = None
        self.state_seen = self.client.state_count
        self.state_at = time.monotonic()
        self.last_send = None
        self.graph_at = self.state_at

    def __getattr__(self, name):
        return getattr(self.client, name)

    def send(self, q_left, q_right, dq_left=None, dq_right=None):
        now = time.monotonic()
        if self.last_send is not None and now - self.last_send > .2:
            raise RuntimeError("发送间隔超过 200 ms，停止本次运动")
        if self.client.state_count != self.state_seen:
            self.state_seen, self.state_at = self.client.state_count, now
        if now - self.state_at > .2:
            raise RuntimeError("关节反馈超过 200 ms 未更新，停止本次运动")
        self.arm._check_feedback()
        if now - self.graph_at >= .1:
            self.arm._check_graph()
            self.graph_at = now
        inputs = {"left": np.asarray(q_left, float), "right": np.asarray(q_right, float)}
        speeds = {"left": dq_left, "right": dq_right}
        if self.single:
            other = "left" if self.side == "right" else "right"
            inputs[other] = self.arm._hold_inputs[other]
            speeds[other] = None
        for side in inputs:
            self.arm._joint_target(inputs[side], side)
        sent_at = time.monotonic()
        if self.last_send is not None and sent_at - self.last_send > .2:
            raise RuntimeError("发送前检查耗时导致间隔超过 200 ms，停止本次运动")
        try:
            self.client.send(inputs["left"], inputs["right"], speeds["left"], speeds["right"])
        except BaseException:
            # 底层在 publish 后还可能因录制失败抛异常，无法确认该帧是否已发。
            # 不用可能过时的保持基准继续另一臂动作；须关闭并重新连接预检。
            self.arm._send_uncertain = True
            raise
        self.last_inputs = {s: q.copy() for s, q in inputs.items()}
        self.last_send = sent_at

    def fresh_state(self, timeout=.5):
        if self.last_inputs is None:
            return self.client.fresh_state(timeout)
        return self.arm._ros._fresh_hold(
            self, "right", self.last_inputs["right"], self.last_inputs["left"], timeout)


class X2Arm:
    """共用连接的左右臂 API；一个进程只允许一个已连接实例。

    R_move_J / L_move_J 分别控制右/左臂；两臂同步运动使用 move_j_both。
    所有运动方法顺序阻塞，不能通过并发调用左右方法实现同步。
    side 只决定 move_j / move_l / FK / IK / 反馈等原有接口的默认侧。

    上肢话题始终包含双臂。单臂方法固定另一臂的原始输入，避免重复按反馈
    加重力补偿；这不保证另一臂的实际位置绝对不变。不要并行运行其他控制器。
    方法执行期间持续发布，返回后不后台保位。返回误差是结束瞬间的测量，
    空闲超过 mc 的约 200 ms 指令有效期后，闭环修正可能不再保持。

    参数
    ----
    side : "left" | "right"
    connect : True 连机器人(要 ROS);False 只做运动学计算
    mode : 连接时只允许 "upper_body"；只读确认 URS，绝不请求状态切换。
    stiffness / bias_limit_deg / gravity_source : 显式值 > 按 SN 配置 > 内置默认。
        默认 40 N·m/rad / 8 deg / chest。当前 SN 配置是 40 / 12 deg / pelvis。
        bias_limit_deg=0 表示不补重力。静态参数不保证 1 mm。
    tcp_offset : TCP 相对腕 roll 连杆的固定外参 (m)。装手后必须实测标定。
    robot_sn : 显式 SN；不给则从环境及 x2ik.conf 读取 X2_ROBOT_SN。
    """

    _connected = None

    def __init__(self, side: str = "right", connect: bool = True,
                 mode: str = "upper_body",
                 stiffness: Optional[float] = None,
                 bias_limit_deg: Optional[float] = None,
                 gravity_source: Optional[str] = None,
                 tcp_offset: Optional[Sequence[float]] = None,
                 payload: float = 0.0,
                 verbose: bool = True, *, robot_sn: Optional[str] = None,
                 _fixed_compensation: bool = False):
        if side not in ("left", "right"):
            raise ValueError("side 必须是 'left' 或 'right'")
        self.side = side
        self.verbose = verbose
        if connect and mode != "upper_body":
            raise ValueError("X2Arm 连接只允许 upper_body；本轮禁止状态切换和关节直控")
        offset = None if tcp_offset is None else _vector(tcp_offset, 3, "tcp_offset")
        self.model = ArmModel(side, tcp_offset=offset)
        self.solver = SrsArmIK(self.model)
        self.cli = None
        self._ros = None
        self.robot_sn = robot_sn
        self._fixed_compensation = _fixed_compensation
        self.connection_config = {}
        self._hold_inputs = {}
        self._send_uncertain = False
        self._motion_lock = threading.Lock()
        if connect:
            self._connect(mode, stiffness, bias_limit_deg, gravity_source, payload)

    def _connect(self, mode, stiffness, bias_limit_deg, gravity_source, payload):
        active = self._connected() if self._connected is not None else None
        if active is not None and active.cli is not None:
            raise RuntimeError("同一进程只能连接一个 X2Arm；请复用实例或先 close()")
        from x2ik import load_conf
        load_conf()
        import x2_sim_ros as ros
        self._ros = ros
        self.robot_sn = self.robot_sn or os.environ.get("X2_ROBOT_SN")
        if self._fixed_compensation:
            # Public Robot uses one fixed profile. SN is identity metadata only;
            # even malformed or stale calibration files must not affect it.
            profile = fixed_compensation()
            stiffness = profile["stiffness"]
            bias_limit_deg = profile["bias_limit_deg"]
            gravity_source = profile["gravity_source"]
        else:
            calibration = ros.load_calibration(self.robot_sn) if self.robot_sn else None
            if self.robot_sn and calibration is None:
                raise ValueError(f"找不到 SN={self.robot_sn} 的有效配置，拒绝静默使用其他机器参数")
            calibration = calibration or {}
            if calibration.get("sn", self.robot_sn) != self.robot_sn:
                raise ValueError("配置文件内 SN 与所选 robot_sn 不一致")
            stiffness = stiffness if stiffness is not None else calibration.get("stiffness", 40.)
            bias_limit_deg = (bias_limit_deg if bias_limit_deg is not None
                              else calibration.get("bias_limit_deg", 8.))
            gravity_source = (gravity_source if gravity_source is not None
                              else calibration.get("gravity_source", "chest"))
        if not math.isfinite(stiffness) or stiffness <= 0:
            raise ValueError("stiffness 必须是有限正数")
        if not math.isfinite(bias_limit_deg) or not 0 <= bias_limit_deg <= 180:
            raise ValueError("bias_limit_deg 必须在 0 到 180 度之间")
        if gravity_source not in ("chest", "pelvis", "static"):
            raise ValueError("gravity_source 必须为 chest / pelvis / static")
        if not math.isfinite(payload) or payload < 0:
            raise ValueError("payload 必须是有限非负数")
        self.connection_config = dict(robot_sn=self.robot_sn, stiffness=stiffness,
                                      bias_limit_deg=bias_limit_deg,
                                      gravity_source=gravity_source, payload=payload)
        k = None if stiffness is None else np.full(7, float(stiffness))
        try:
            self.cli = ros.X2ArmClient(mode, ros.DEFAULT_ARM_ORDER,
                                   joint_stiffness=k, payload=payload,
                                   gravity_source=gravity_source,
                                   bias_limit=math.radians(bias_limit_deg),
                                   verbose=self.verbose)
            self.cli.models[self.side] = self.model
            self.cli.iks[self.side] = self.solver
            if not self.cli.wait_state(timeout=10.0):
                raise RuntimeError("等了 10s 收不到关节反馈。检查 ROS_DOMAIN_ID / "
                               "RMW_IMPLEMENTATION 是否与机器人一致,或跑 "
                               "./x2ik.py doctor")
            # DDS 对命令端点与反馈端点独立发现；连接阶段尚未发运动。
            self.cli.spin(3.)
            self._prepare_motion()
            X2Arm._connected = weakref.ref(self)
            if self.verbose:
                print(f"[X2Arm] 已只读确认 URS；配置 {self.connection_config}")
        except BaseException:
            self.close()
            raise

    def _need_robot(self, what: str) -> None:
        if self.cli is None:
            raise RuntimeError(f"{what} 需要连接机器人。构造时用 X2Arm(side) "
                               f"而不是 X2Arm(side, connect=False)")

    def _joint_target(self, q, side):
        target = _vector(q, 7, "q")
        model = self.model if side == self.side else self.cli.models[side]
        if not model.within_limits(target):
            raise ValueError(f"{side} 目标超出 URDF 关节限位；未发送该目标")
        return target

    def _check_graph(self):
        topic = self._ros.UPPER_BODY_TOPIC
        pubs = self.cli.node.count_publishers(topic)
        subs = self.cli.node.count_subscribers(topic)
        if pubs != 1 or subs < 1:
            raise RuntimeError(f"{topic}: 发布者={pubs}（须仅本实例 1 个），订阅者={subs}")

    def _check_feedback(self):
        if not self._ros._complete_arm_feedback(self.cli):
            raise RuntimeError("最新反馈必须含双臂完整 14 关节且角度有限")
        for side in ("left", "right"):
            if not np.all(np.isfinite(self.cli.dq(side))):
                raise RuntimeError(f"{side} 速度反馈非有限数")
            self._joint_target(self.cli.q(side), side)
        source = self.connection_config.get("gravity_source", "static")
        if source != "static" and self.cli.imu_count[source] <= 0:
            raise RuntimeError(f"未收到 {source} IMU，拒绝静默使用默认重力")
        if source == "pelvis" and self.cli.imu_has_orientation[source] is not True:
            raise RuntimeError("pelvis IMU 必须提供有效姿态")

    def _prepare_motion(self):
        self._need_robot("运动")
        if self._send_uncertain:
            raise RuntimeError("上次底层发送异常，最后一帧是否发布无法确认；"
                               "请 close() 后重新连接并只读预检，不能直接继续运动")
        if self.cli.mode != "upper_body":
            raise RuntimeError("只允许 upper_body 接口")
        action = self.cli.get_action()
        if (action or "").split("(", 1)[0].strip() not in (
                "UPPERBODY_REMOTE_SPLIT", "URS", "US"):
            raise RuntimeError(f"当前状态 {action!r}；只允许已经处于 URS，不切换状态")
        self._check_graph()
        if not self.cli.fresh_state(.5):
            raise RuntimeError("未收到新关节反馈，未开始运动")
        self._check_feedback()
        if not self._hold_inputs:
            self._hold_inputs = {s: self.cli.q(s).copy() for s in ("left", "right")}

    def _motion(self, execute, *, single=True, side=None):
        if not self._motion_lock.acquire(blocking=False):
            raise RuntimeError("同一 X2Arm 的运动方法必须顺序调用")
        client = None
        try:
            self._prepare_motion()
            client = _MotionClient(self, single, side)
            return execute(client)
        finally:
            # 每侧都保留最后确认发送完成的原始输入，包括单臂/双臂/MoveL 及中途
            # 异常。交替操作另一侧时不能重取漂移反馈，或拉回旧的连接基准。
            if client is not None and client.last_inputs is not None:
                self._hold_inputs = {s: q.copy() for s, q in client.last_inputs.items()}
            self._motion_lock.release()

    # ------------------------------------------------------------------ 正运动学

    def fk(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """正运动学:关节角 -> TCP 位姿。

        入 q[7] (rad),出 (pos[3] m, rpy[3] rad),torso 系。
        纯计算,不需要机器人。
        """
        q = _vector(q, 7, "q")
        pos, rot = self.model.forward_kinematics(q)
        return pos, matrix_to_rpy(rot)

    def fk_matrix(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """同 fk,但姿态返回 3x3 旋转矩阵而不是 rpy。避免 rpy 的万向锁。"""
        return self.model.forward_kinematics(_vector(q, 7, "q"))

    # ------------------------------------------------------------------ 逆运动学

    def ik(self, pos: Sequence[float], rpy: Optional[Sequence[float]] = None,
           q_seed: Optional[Sequence[float]] = None,
           relax_deg: float = 0.0) -> Optional[np.ndarray]:
        """逆运动学:TCP 位姿 -> 关节角。

        pos[3] (m) 必给。rpy[3] (rad) 不给就沿用待机姿态。
        q_seed 是求解种子,给上一帧的解有助于提高连续性，但不保证不跳解；
        不给用待机位。运动前仍须检查相邻解的关节变化。
        relax_deg > 0 时姿态变成软约束:位置到不了就允许绕单轴让最多这么多度,
        返回的解姿态可能与请求的不同 —— 用 fk 复核。

        解不出来返回 None(不抛异常),调用方必须判 None。
        纯计算,不需要机器人。
        """
        pos = _vector(pos, 3, "pos")
        seed = HOME.copy() if q_seed is None else _vector(q_seed, 7, "q_seed")
        if not math.isfinite(relax_deg) or relax_deg < 0:
            raise ValueError("relax_deg 必须为有限非负数")
        if rpy is None:
            _, rot = self.model.forward_kinematics(HOME)
        else:
            rot = rpy_to_matrix(_vector(rpy, 3, "rpy"))
        if relax_deg > 0:
            relaxed = self.solver.solve_hold_rotation(
                pos, rot, q_seed=seed, max_relax=math.radians(relax_deg))
            return None if relaxed is None else relaxed.sol.q
        sol = self.solver.solve(pos, rot, q_seed=seed)
        return None if sol is None else sol.q

    def ik_full(self, pos: Sequence[float], rpy: Optional[Sequence[float]] = None,
                q_seed: Optional[Sequence[float]] = None):
        """同 ik,但返回完整的 IKSolution(带 psi / 解支 / 残差),不只是 q。"""
        pos = _vector(pos, 3, "pos")
        seed = HOME.copy() if q_seed is None else _vector(q_seed, 7, "q_seed")
        rot = (self.model.forward_kinematics(HOME)[1] if rpy is None
               else rpy_to_matrix(_vector(rpy, 3, "rpy")))
        return self.solver.solve(pos, rot, q_seed=seed)

    def reachable(self, pos: Sequence[float],
                  rpy: Optional[Sequence[float]] = None) -> bool:
        """这个位姿解得出来吗。就是 ik() is not None,读起来顺一点。"""
        return self.ik(pos, rpy) is not None

    # ------------------------------------------------------------------ MoveJ

    def move_j(self, q: Optional[Sequence[float]] = None,
               duration: float = 3.0, settle: float = 1.5) -> Dict:
        """MoveJ:点到点关节运动。五次多项式插值,起终点速度加速度为 0。

        q[7] (rad) 不给就默认侧回待机位。另一臂沿用最后成功发送的原始输入，
        初次来自连接时反馈。MoveJ 无闭环精定位，不承诺 TCP 1 mm。
        duration 是运动用时 (s),settle 是到位后保持的时间 (s) ——
        保持段不是可省的:位置环要时间稳定,不等就量到运动中途的值。

        返回 dict:q(结束时新反馈)/err(逐关节误差)/err_max(最大误差 rad)/
        tau(反馈力矩)/stale(False)。未获得新反馈时抛 RuntimeError。
        """
        return self._move_j_side(self.side, q, duration, settle)

    def R_move_J(self, q: Optional[Sequence[float]] = None,
                 duration: float = 3.0, settle: float = 1.5) -> Dict:
        """右臂 MoveJ，保持左臂最后原始输入；参数/返回值同 move_j。

        不受构造 side 影响。阻塞到本次动作结束；与 L_move_J 顺序调用不等于
        双臂同时运动，同时运动请用 move_j_both(q_left, q_right)。
        """
        return self._move_j_side("right", q, duration, settle)

    def L_move_J(self, q: Optional[Sequence[float]] = None,
                 duration: float = 3.0, settle: float = 1.5) -> Dict:
        """左臂 MoveJ，保持右臂最后原始输入；参数/返回值同 move_j。

        不受构造 side 影响，也不改变后续 FK / IK / MoveL 的默认侧。
        """
        return self._move_j_side("left", q, duration, settle)

    def _move_j_side(self, side, q, duration, settle):
        self._need_robot("MoveJ")
        target = self._joint_target(HOME if q is None else q, side)
        _timing(duration, settle)
        other = "left" if side == "right" else "right"
        def execute(client):
            goal = {side: target, other: self._hold_inputs[other]}
            return self._goto_joint(client, goal, duration, settle)[side]
        return self._motion(execute, side=side)

    def _goto_joint(self, client, goals, duration, settle):
        result = self._ros.goto_joint(client, goals, duration, settle)
        if not client.fresh_state(.5):
            raise RuntimeError("MoveJ 结束时未获得完整新反馈，不能判定到位")
        for side in goals:
            measured = client.q(side)
            error = measured - goals[side]
            result[side].update(q=measured, err=error, err_max=float(np.max(np.abs(error))),
                                tau=client.tau(side), stale=False)
        return result

    def move_j_both(self, q_left: Sequence[float], q_right: Sequence[float],
                    duration: float = 3.0, settle: float = 1.5) -> Dict:
        """两臂共用五次时间进度和 50 Hz 发布，同步生成左右目标。

        duration / settle 两侧相同，返回 {"left": {...}, "right": {...}}。
        这是命令轨迹同步，不保证两侧物理跟踪误差相同；不做双臂碰撞规划。
        """
        self._need_robot("move_j_both")
        goals = {"left": self._joint_target(q_left, "left"),
                 "right": self._joint_target(q_right, "right")}
        _timing(duration, settle)
        return self._motion(lambda client: self._goto_joint(client, goals, duration, settle),
                            single=False)

    def home(self, duration: float = 3.0, settle: float = 2.0) -> Dict:
        """显式让两条臂一起回待机位；只回本臂请用 move_j(HOME)。"""
        return self.move_j_both(HOME, HOME, duration, settle)

    # ------------------------------------------------------------------ MoveL

    def move_l(self, pos: Sequence[float], rpy: Optional[Sequence[float]] = None,
               duration: float = 3.0, settle: float = 2.0, *,
               converge: int = 0, converge_tol: float = .001,
               converge_step: float = math.radians(.5),
               converge_total: float = math.radians(3.)) -> Dict:
        """MoveL:笛卡尔直线运动。TCP 走直线,姿态走测地线插值。

        pos[3] (m) 必给,rpy[3] (rad) 不给就保持待机姿态。
        逐帧 IK 锁解支;单帧跳变超过 0.05 rad 的候选会被**拒绝**并保持上一条
        已接受指令 —— 所以返回值里的 step_rejects / trajectory_valid 要看,
        不是装饰:trajectory_valid=False 意味着这条轨迹没走完。

        converge > 0 在尾部增加关节空间闭环，当前建议上限 8 轮。
        converge_tol 用 m，step/total 用 rad；默认关闭。pos_err 是反馈关节角
        经 URDF FK 的位置误差，不含装配误差。检查 converged 和 converge_reason，
        不要仅凭函数正常返回判定达标。返回后不持续保位。

        返回 dict,关键字段:
            pos_err     末端位置误差 (m)
            rot_err     末端姿态误差 (rad)
            trajectory_valid  轨迹是否完整走完
            step_rejects / branch_rejects   被拒绝的跳变候选数
            ik_fails    IK 失败帧数
        """
        self._need_robot("move_l")
        pos = _vector(pos, 3, "pos")
        _timing(duration, settle)
        self._ros._validate_converge(converge, converge_tol, converge_step, converge_total)
        rot = (self.model.forward_kinematics(HOME)[1] if rpy is None
               else rpy_to_matrix(_vector(rpy, 3, "rpy")))
        return self._motion(lambda client: self._ros.goto_cartesian(
            client, self.side, pos, rot, duration, settle,
            converge=converge, converge_tol=converge_tol,
            converge_step=converge_step, converge_total=converge_total))

    # ------------------------------------------------------------------ 反馈

    def joints(self) -> np.ndarray:
        """当前关节角 (rad),7 个。"""
        self._need_robot("joints")
        if not self.cli.fresh_state(.5):
            raise RuntimeError("未收到新关节反馈")
        self._check_feedback()
        return self.cli.q(self.side).copy()

    def velocities(self) -> np.ndarray:
        """当前关节速度 (rad/s)。"""
        self._need_robot("velocities")
        return self.cli.dq(self.side)

    def torques(self) -> np.ndarray:
        """当前关节力矩 (N·m)。真机实测非零,峰值约 7.6 N·m。"""
        self._need_robot("torques")
        return self.cli.tau(self.side)

    def pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """当前 TCP 位姿 (pos m, rpy rad) —— 由反馈关节角正解出来。"""
        return self.fk(self.joints())

    # ------------------------------------------------------------------ 录制

    def start_record(self, path, tcp: bool = True) -> None:
        """开始录制轨迹 CSV。之后每下发一帧落一行,直到 stop_record()。

        录的是给定位置 / 实发位置(含重力偏置) / 反馈位置 / 反馈速度 /
        反馈力矩,两臂 14 关节全量。用 `./x2ik.py jump <csv>` 分析。
        """
        self._need_robot("start_record")
        from x2_record import JointRecorder
        if self.cli.recorder is not None:
            self.cli.recorder.close()
        self.cli.recorder = JointRecorder(Path(path), tcp=tcp)

    def stop_record(self) -> None:
        """停止录制并写完文件。"""
        if self.cli is not None and self.cli.recorder is not None:
            self.cli.recorder.close()
            self.cli.recorder = None

    # ------------------------------------------------------------------ 收尾

    def close(self) -> None:
        """断开并释放节点。录制写盘失败也执行清理，异常向调用方传播。"""
        try:
            self.stop_record()
        finally:
            client, self.cli = self.cli, None
            try:
                if client is not None:
                    try:
                        client.close()
                    finally:
                        if client.rclpy.ok():
                            client.rclpy.shutdown()
            finally:
                active = self._connected() if self._connected is not None else None
                if active is self:
                    X2Arm._connected = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------
# 自检:不连机器人也能跑,验证 FK/IK 一致性
# --------------------------------------------------------------------------

def _selftest() -> int:
    print("X2Arm 离线自检(FK/IK,不需要机器人)\n")
    ok = True
    for side in ("left", "right"):
        arm = X2Arm(side, connect=False)
        pos, rpy = arm.fk(HOME)
        print(f"{side:5s} FK(待机位) pos = [{pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f}] m"
              f"   rpy = [{math.degrees(rpy[0]):7.2f} {math.degrees(rpy[1]):7.2f}"
              f" {math.degrees(rpy[2]):7.2f}] deg")
        # FK->IK->FK 闭环:解出来的 q 未必等于原 q(7 轴有零空间),
        # 但正解回去的位姿必须一致。这才是正确的判据。
        q = arm.ik(pos, rpy)
        if q is None:
            print(f"{side:5s} [!] IK 解不出待机位姿,这不应该发生")
            ok = False
            continue
        p2, r2 = arm.fk(q)
        dp = float(np.linalg.norm(p2 - pos))
        dr = float(np.max(np.abs(np.degrees(r2 - rpy))))
        flag = "ok" if dp < 1e-3 and dr < 0.5 else "!!"
        print(f"{side:5s} IK 闭环   位置残差 {dp * 1000:.4f} mm   "
              f"姿态残差 {dr:.4f} deg   [{flag}]")
        if flag == "!!":
            ok = False
        # 抓取常用的前伸点,验证不是只有待机位能解
        probe = pos + np.array([0.10, 0.0, -0.05])
        q2 = arm.ik(probe, rpy, q_seed=q)
        print(f"{side:5s} IK 前伸点 {'解出' if q2 is not None else '无解'}"
              f"  {np.round(probe, 3).tolist()}")
    print("\n" + ("全部通过" if ok else "有项目未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
