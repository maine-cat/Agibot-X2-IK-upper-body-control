#!/usr/bin/env python3
"""X2 上肢控制桥接层 —— 两种关节接口方案的统一封装。

  InterfaceMode.POSITION_ONLY  方案一:原生上下肢分离,只能下发位置。
                               反馈接口与方案二相同(含力矩/位置/速度)。
  InterfaceMode.FULL_JOINT     方案二:开放关节控制接口,可下发力矩/位置/速度。

两种模式共用同一套 IK 与同一套重力模型,差异被收敛在 `compute_command()`
一个函数里 —— 这是刻意的:上层运动规划不应该知道底层用哪种接口,
将来切换方案时只改一个枚举。

依赖 rclpy;没有 ROS 环境时本文件可以被 import(不会崩),只是无法 spin。
用 `python3 x2_arm_bridge.py --dry-run` 可以在无 ROS 机器上跑通指令生成逻辑。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from x2_arm_model import ArmModel
from x2_arm_dynamics import ArmDynamics
from x2_srs_ik import IKSolution, SrsArmIK

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    HAS_ROS = True
except ImportError:                      # 允许无 ROS 环境下做纯算法验证
    HAS_ROS = False
    Node = object                        # type: ignore


class InterfaceMode(Enum):
    POSITION_ONLY = "position_only"
    FULL_JOINT = "full_joint"


@dataclass
class JointCommand:
    """一帧关节指令。方案一只会填 position;方案二三项都填。"""
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    torque: Optional[np.ndarray] = None
    kp: Optional[np.ndarray] = None
    kd: Optional[np.ndarray] = None


@dataclass
class ControlConfig:
    mode: InterfaceMode = InterfaceMode.FULL_JOINT
    control_hz: float = 100.0

    # --- 方案二用 ---
    # MIT 式关节控制 tau = kp(q_d-q) + kd(dq_d-dq) + tau_ff
    # 这里的数值是保守起步值,不是标定结果。上机前必须按实际关节整定,
    # 肩部大惯量关节应比腕部高一个量级。
    kp: Sequence[float] = field(default_factory=lambda: [60., 60., 40., 40., 20., 10., 10.])
    kd: Sequence[float] = field(default_factory=lambda: [3.0, 3.0, 2.0, 2.0, 1.0, 0.5, 0.5])
    gravity_ff_scale: float = 0.8        # 前馈打折,留给模型误差

    # --- 方案一用 ---
    # 等效关节刚度 N·m/rad,必须实测:固定姿态测稳态角度偏差,
    # k = tau_g(该姿态) / 角度偏差。抄 A3 的数会错,X2 减速比和臂长都不同。
    joint_stiffness: Sequence[float] = field(default_factory=lambda: [0.] * 7)
    bias_limit_rad: Sequence[float] = field(default_factory=lambda: [0.03] * 7)

    # --- 通用安全限 ---
    max_joint_step_rad: float = 0.05     # 单帧最大关节变化,防跳解/防拉飞
    payload_mass: float = 0.0
    payload_com: Sequence[float] = field(default_factory=lambda: [0., 0., 0.])
    tcp_offset: Sequence[float] = field(default_factory=lambda: [0., 0., 0.])


class ArmController:
    """单臂控制器:目标位姿 -> 关节指令。不含任何 ROS 依赖,便于单测。"""

    def __init__(self, side: str = "right", config: Optional[ControlConfig] = None):
        self.config = config or ControlConfig()
        self.model = ArmModel(side, tcp_offset=np.asarray(self.config.tcp_offset, float))
        self.ik = SrsArmIK(self.model)
        self.dyn = ArmDynamics(self.model,
                               payload_mass=self.config.payload_mass,
                               payload_com=self.config.payload_com)
        self.q_cmd = self.ik.q_ref.copy()
        self.q_meas = self.ik.q_ref.copy()
        self.dq_meas = np.zeros(7)
        self.tau_meas = np.zeros(7)
        self.psi_cmd: Optional[float] = None
        self.stats = {"frames": 0, "ik_fail": 0, "clamped_reach": 0, "clamped_step": 0}

    # ---------- 反馈 ----------

    def update_feedback(self, q: Sequence[float],
                        dq: Optional[Sequence[float]] = None,
                        tau: Optional[Sequence[float]] = None) -> None:
        """两种方案的反馈接口相同,所以这个函数不分模式。"""
        self.q_meas = np.asarray(q, float)
        if dq is not None:
            self.dq_meas = np.asarray(dq, float)
        if tau is not None:
            self.tau_meas = np.asarray(tau, float)

    def gravity_residual(self) -> np.ndarray:
        """实测力矩 - 模型重力力矩。方案二独有的诊断量:

        这个残差稳定非零 = 模型质量/质心不准或有未建模负载;
        残差随姿态剧烈变化 = 摩擦或减速比未建模。
        标定 payload_mass 时就盯这个数。
        """
        return self.tau_meas - self.dyn.gravity_torque(self.q_meas)

    # ---------- 指令生成 ----------

    def compute_command(self, pos: Sequence[float], rot: np.ndarray,
                        psi_desired: Optional[float] = None,
                        track_from_measured: bool = False) -> Optional[JointCommand]:
        """核心:目标位姿 -> 一帧关节指令。

        track_from_measured=True 时以实测关节角为 IK 种子(闭环),
        False 时以上一帧指令为种子(开环)。开环更平滑,闭环更抗外扰 ——
        方案二有力矩反馈,建议闭环;方案一位置环刚性高,开环即可。
        """
        cfg = self.config
        pos = np.asarray(pos, float)
        rot = np.asarray(rot, float)

        pos_clamped, was_clamped = self.ik.project_to_workspace(pos, rot)
        if was_clamped:
            self.stats["clamped_reach"] += 1

        seed = self.q_meas if track_from_measured else self.q_cmd
        sol: Optional[IKSolution] = self.ik.track(
            pos_clamped, rot, q_prev=seed,
            psi_prev=self.psi_cmd, psi_desired=psi_desired)
        self.stats["frames"] += 1
        if sol is None:
            self.stats["ik_fail"] += 1
            return None                  # 调用方应保持上一帧指令,不要下发默认值

        q_target = self._limit_step(sol.q, seed)
        q_target = self.model.clamp(q_target)
        dq_target = (q_target - self.q_cmd) * cfg.control_hz
        self.q_cmd = q_target
        self.psi_cmd = sol.psi

        if cfg.mode is InterfaceMode.FULL_JOINT:
            return JointCommand(
                position=q_target,
                velocity=dq_target,
                torque=self.dyn.feedforward_torque(q_target, scale=cfg.gravity_ff_scale),
                kp=np.asarray(cfg.kp, float),
                kd=np.asarray(cfg.kd, float),
            )

        # 方案一:无力矩通道,把重力折成位置偏置
        bias = self.dyn.position_bias(q_target, cfg.joint_stiffness, cfg.bias_limit_rad)
        return JointCommand(position=self.model.clamp(q_target + bias))

    def _limit_step(self, q_new: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        """单帧关节变化限幅。触发即说明 IK 换支或目标跳变,是重要的告警信号。"""
        step = q_new - q_ref
        peak = float(np.max(np.abs(step)))
        cap = self.config.max_joint_step_rad
        if peak > cap:
            self.stats["clamped_step"] += 1
            step = step * (cap / peak)
        return q_ref + step

    def report(self) -> str:
        s = self.stats
        return (f"帧数 {s['frames']}  IK失败 {s['ik_fail']}  "
                f"工作空间裁剪 {s['clamped_reach']}  步长限幅 {s['clamped_step']}")


# --------------------------------------------------------------------------
# ROS2 节点
# --------------------------------------------------------------------------

if HAS_ROS:

    class ArmBridgeNode(Node):
        """订阅关节反馈、发布关节指令。

        话题名按 X2 实际接口填 —— 这里给的是 A3 demo 的命名习惯作为占位,
        接入前务必用 `ros2 topic list` 核对,X2 的命名大概率不同。
        """

        def __init__(self, sides=("right",), config: Optional[ControlConfig] = None):
            super().__init__("x2_arm_bridge")
            self.controllers: Dict[str, ArmController] = {
                s: ArmController(s, config) for s in sides
            }
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(JointState, "/motion/control/arm_joint_state",
                                     self._on_state, qos)
            self.pub = self.create_publisher(JointState,
                                             "/motion/control/arm_joint_command", qos)
            self.name_index: Optional[Dict[str, int]] = None
            self.get_logger().info(
                f"X2 arm bridge 启动,模式={self.controllers[sides[0]].config.mode.value}")

        def _on_state(self, msg: JointState) -> None:
            """按关节名索引,绝不按下标硬取。

            X2 腕部顺序是 yaw->pitch->roll,而 A3 是 roll->pitch->yaw;
            从 A3 迁移时按下标复制会静默地把腕部装反,而且 FK 仍然自洽,
            只有实机动起来才会发现。所以这里强制名字匹配。
            """
            if self.name_index is None:
                self.name_index = {n: i for i, n in enumerate(msg.name)}
            for ctrl in self.controllers.values():
                try:
                    idx = [self.name_index[n] for n in ctrl.model.joint_names]
                except KeyError as exc:
                    self.get_logger().error(f"反馈里找不到关节 {exc};实际名单={msg.name}")
                    return
                pick = lambda seq: (np.array([seq[i] for i in idx])
                                    if seq is not None and len(seq) > max(idx) else None)
                ctrl.update_feedback(pick(msg.position),
                                     pick(msg.velocity),
                                     pick(msg.effort))

        def send(self, commands: Dict[str, JointCommand]) -> None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            names: List[str] = []
            pos: List[float] = []
            vel: List[float] = []
            eff: List[float] = []
            for side, cmd in commands.items():
                ctrl = self.controllers[side]
                names.extend(ctrl.model.joint_names)
                pos.extend(cmd.position.tolist())
                vel.extend((cmd.velocity if cmd.velocity is not None
                            else np.zeros(7)).tolist())
                eff.extend((cmd.torque if cmd.torque is not None
                            else np.zeros(7)).tolist())
            msg.name, msg.position, msg.velocity, msg.effort = names, pos, vel, eff
            self.pub.publish(msg)


# --------------------------------------------------------------------------
# 离线自检
# --------------------------------------------------------------------------

def _dry_run(mode: InterfaceMode, frames: int = 300) -> None:
    """无 ROS 环境下跑通两种模式的指令生成,并打印关键量。"""
    cfg = ControlConfig(mode=mode, payload_mass=0.3, payload_com=[0., 0., 0.04],
                        joint_stiffness=[900., 900., 600., 600., 0., 0., 0.])
    ctrl = ArmController("right", cfg)
    p0, r0 = ctrl.model.forward_kinematics(ctrl.q_cmd)

    peak_step = 0.0
    peak_tau = np.zeros(7)
    fails = 0
    for k in range(frames):
        t = k / frames * 2 * np.pi
        pos = p0 + np.array([0.04 * np.sin(t), 0.03 * np.cos(t) - 0.03, 0.02 * np.sin(2 * t)])
        prev = ctrl.q_cmd.copy()
        ctrl.update_feedback(ctrl.q_cmd)          # 理想跟随,当作仿真反馈
        cmd = ctrl.compute_command(pos, r0)
        if cmd is None:
            fails += 1
            continue
        peak_step = max(peak_step, float(np.max(np.abs(ctrl.q_cmd - prev))))
        if cmd.torque is not None:
            peak_tau = np.maximum(peak_tau, np.abs(cmd.torque))

    print(f"--- {mode.value} ---")
    print(f"  {ctrl.report()}  (额外无解 {fails})")
    print(f"  单帧最大关节变化 : {np.degrees(peak_step):.3f} deg "
          f"(限 {np.degrees(cfg.max_joint_step_rad):.1f} deg)")
    if mode is InterfaceMode.FULL_JOINT:
        print(f"  前馈力矩峰值     : {np.round(peak_tau, 3).tolist()} N·m")
        print(f"  占额定比例       : "
              f"{np.round(peak_tau / ctrl.model.tau_max_safe * 100, 1).tolist()} %")
    else:
        bias = ctrl.dyn.position_bias(ctrl.q_cmd, cfg.joint_stiffness, cfg.bias_limit_rad)
        print(f"  末帧位置偏置     : {np.round(np.degrees(bias), 4).tolist()} deg")
        print("  注意: joint_stiffness 是占位值,必须实测标定后才有意义")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="无 ROS 离线自检")
    args = parser.parse_args()

    if args.dry_run or not HAS_ROS:
        if not HAS_ROS and not args.dry_run:
            print("未检测到 rclpy,转为离线自检。\n")
        for mode in InterfaceMode:
            _dry_run(mode)
        return

    rclpy.init()
    node = ArmBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
