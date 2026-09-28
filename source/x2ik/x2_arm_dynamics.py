#!/usr/bin/env python3
"""X2 单臂重力与负载前馈。

用途按接口方案分两种:

  方案二(开放关节控制接口,含力矩):
      tau_cmd = kp*(q_d - q) + kd*(dq_d - dq) + tau_g(q) + tau_load(q)
      重力项直接进 tau_ff,是真正的前馈,能显著降低稳态跟踪误差。

  方案一(原生上下肢分离,只有位置指令):
      没有 tau_ff 通道,只能把重力折算成"位置偏置"再叠到指令上:
      q_cmd = q_d + tau_g(q_d) / k_joint_effective
      这正是 A3 demo 里 DYNAMIC_FF_* 那套增益在做的事。注意 k_joint_effective
      是被控关节的等效刚度,不是 URDF 里的任何一个数 —— 必须实测标定
      (给定姿态下测稳态角度偏差 / 该姿态的重力力矩),不能从 A3 直接抄。

重力力矩由虚功原理算,不需要完整 RNEA:

    U(q) = -sum_j  m_j * g^T * c_j(q)
    tau_i = dU/dq_i = -sum_{j>=i} m_j * g^T * (a_i x (c_j - o_i))

其中 a_i / o_i 是关节 i 的轴与原点(torso 系),c_j 是连杆 j 质心。
只对 i 下游的连杆求和 —— 上游连杆不受关节 i 影响。

坐标系:重力向量默认取 torso 系的 (0,0,-9.81)。X2 有 3 个腰关节且下肢跑
RL 控制器,躯干会持续倾斜,所以实机上必须用腰部反馈或 IMU 把重力向量转到
torso 系再传进来,否则大臂展姿态下前馈会算错方向。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .x2_arm_model import ArmModel, cross3, rpy_to_matrix

GRAVITY = np.array([0.0, 0.0, -9.81])

# 手臂各关节对应的下游连杆(URDF link 名后缀),顺序与 ARM_JOINT_SUFFIX 一致
ARM_LINK_SUFFIX = (
    "shoulder_pitch_link",
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_yaw_link",
    "wrist_pitch_link",
    "wrist_roll_link",
)


@dataclass(frozen=True)
class LinkInertial:
    name: str
    mass: float
    com: np.ndarray        # 连杆自身系下的质心


def parse_urdf_inertials(urdf_path: Path) -> Dict[str, LinkInertial]:
    """只取质量与质心 —— 重力前馈用不到惯量张量。"""
    text = Path(urdf_path).read_text(encoding="utf-8")
    out: Dict[str, LinkInertial] = {}
    for match in re.finditer(r'<link\s+name="([^"]+)"\s*>(.*?)</link>', text, re.S):
        name, body = match.groups()
        inertial = re.search(r"<inertial>(.*?)</inertial>", body, re.S)
        if not inertial:
            continue
        blob = inertial.group(1)
        mass_m = re.search(r'<mass\s+value="([^"]+)"', blob)
        origin_m = re.search(r'<origin[^/>]*xyz="([^"]+)"', blob)
        if not mass_m:
            continue
        com = (np.array([float(v) for v in origin_m.group(1).split()])
               if origin_m else np.zeros(3))
        out[name] = LinkInertial(name=name, mass=float(mass_m.group(1)), com=com)
    return out


class ArmDynamics:
    """单臂重力 / 负载前馈。

    payload_mass + payload_com 描述装在腕 roll 连杆末端的手爪或工件。
    URDF 里没有 hand link,所以这部分必须由调用方按实际手型给出。
    payload_com 保持历史约定:从 TCP 原点到质心的偏移,在 wrist_roll_link
    坐标系表达。改变 TCP 轴方向不会转动同一个物理负载的质心。
    """

    def __init__(self, model: ArmModel,
                 payload_mass: float = 0.0,
                 payload_com: Optional[Sequence[float]] = None,
                 gravity: Optional[Sequence[float]] = None):
        self.model = model
        inertials = parse_urdf_inertials(model.urdf_path)
        names = [f"{model.side}_{s}" for s in ARM_LINK_SUFFIX]
        missing = [n for n in names if n not in inertials]
        if missing:
            raise RuntimeError(f"URDF 中缺少 inertial: {missing}")
        self.links: List[LinkInertial] = [inertials[n] for n in names]
        self.masses = np.array([l.mass for l in self.links])
        self.arm_mass = float(self.masses.sum())

        self.payload_mass = float(payload_mass)
        self.payload_com = (np.zeros(3) if payload_com is None
                            else np.asarray(payload_com, float))
        self.gravity = GRAVITY.copy() if gravity is None else np.asarray(gravity, float)

    # ---------- 质心 ----------

    def link_coms(self, q: np.ndarray) -> List[np.ndarray]:
        """各臂连杆质心在 torso 系下的位置。

        约定:连杆 j 固连在关节 j 的输出侧,所以用 rotations[j](已含 q_j)。
        """
        positions, rotations = self.model.joint_frames(q)
        return [positions[j] + rotations[j] @ self.links[j].com for j in range(7)]

    def total_com(self, q: np.ndarray) -> np.ndarray:
        """整条臂(含负载)的合质心,可用于整机平衡/腰部补偿。"""
        coms = self.link_coms(q)
        weighted = sum(m * c for m, c in zip(self.masses, coms))
        total = self.arm_mass
        if self.payload_mass > 0.0:
            positions, rotations = self.model.joint_frames(q)
            tip = positions[6] + rotations[6] @ (self.model.tcp_offset + self.payload_com)
            weighted = weighted + self.payload_mass * tip
            total += self.payload_mass
        return weighted / total

    # ---------- 重力力矩 ----------

    def gravity_torque(self, q: np.ndarray, include_payload: bool = True) -> np.ndarray:
        """tau_g(q),单位 N·m。正号约定与 URDF 关节正方向一致。"""
        q = np.asarray(q, float)
        positions, rotations = self.model.joint_frames(q)
        axes = self.model.axes_at(q)
        coms = [positions[j] + rotations[j] @ self.links[j].com for j in range(7)]

        masses = list(self.masses)
        if include_payload and self.payload_mass > 0.0:
            coms.append(positions[6] + rotations[6]
                        @ (self.model.tcp_offset + self.payload_com))
            masses.append(self.payload_mass)

        tau = np.zeros(7)
        for i in range(7):
            axis, origin = axes[i], positions[i]
            acc = 0.0
            for j in range(i, len(masses)):       # 只累下游连杆
                acc += masses[j] * float(np.dot(self.gravity,
                                                cross3(axis, coms[j] - origin)))
            tau[i] = -acc
        return tau

    def gravity_torque_ratio(self, q: np.ndarray) -> np.ndarray:
        """重力力矩占可用力矩上限的比例。>1 意味着该姿态举不住。

        分母用 tau_max_safe(URDF/MJCF 取小),所以腕部比例会比按 URDF 算的高一倍多。
        """
        return np.abs(self.gravity_torque(q)) / np.maximum(self.model.tau_max_safe, 1e-9)

    # ---------- 方案一:位置偏置折算 ----------

    def position_bias(self, q: np.ndarray,
                      joint_stiffness: Sequence[float],
                      limit: Optional[Sequence[float]] = None) -> np.ndarray:
        """把重力力矩折算成位置指令偏置(rad),供只有位置接口的方案一使用。

        joint_stiffness: 补偿计算采用的各关节等效刚度 N·m/rad；客户入口当前固定为
        40，不代表已完成实测辨识。给 0 表示该关节
        不做补偿(例如腕部三轴力臂短、重力影响可忽略)。
        limit: 每关节偏置上限,防止标定不准时把指令推飞。
        """
        stiff = np.asarray(joint_stiffness, float)
        tau = self.gravity_torque(q)
        bias = np.where(stiff > 1e-9, tau / np.maximum(stiff, 1e-9), 0.0)
        if limit is not None:
            lim = np.abs(np.asarray(limit, float))
            bias = np.clip(bias, -lim, lim)
        return bias

    # ---------- 方案二:完整前馈力矩 ----------

    def feedforward_torque(self, q: np.ndarray,
                           tau_task: Optional[np.ndarray] = None,
                           scale: float = 1.0,
                           saturate: bool = True) -> np.ndarray:
        """tau_ff = scale * tau_g(q) + tau_task,并按 URDF effort 限幅。

        scale < 1 用于上线初期保守起步:模型质量和实际总有差,先给 0.5~0.8,
        看实际稳态误差往哪边偏再往上调。

        限幅用 model.tau_max_safe = min(URDF effort, MJCF ctrlrange)。
        官方 v1.3.0 里这两者对腕部不一致(4.8 vs 2.2 N·m),取小是保守做法:
        按 4.8 下发可能直接触发驱动器保护。口径仍需向 FAE 确认。
        """
        tau = scale * self.gravity_torque(q)
        if tau_task is not None:
            tau = tau + np.asarray(tau_task, float)
        if saturate:
            limit = self.model.tau_max_safe
            tau = np.clip(tau, -limit, limit)
        return tau

    def task_torque(self, q: np.ndarray, wrench: Sequence[float]) -> np.ndarray:
        """末端期望力/力矩 -> 关节力矩,tau = J^T * F。用于力控或柔顺贴合。"""
        return self.model.jacobian(q).T @ np.asarray(wrench, float).reshape(6)


if __name__ == "__main__":
    for side in ("right", "left"):
        model = ArmModel(side)
        dyn = ArmDynamics(model, payload_mass=0.5, payload_com=[0.0, 0.0, 0.05])
        print(f"===== {side} =====")
        print(f"  臂总质量 : {dyn.arm_mass:.3f} kg  (逐连杆 "
              f"{np.round(dyn.masses, 3).tolist()})")
        for label, q in (("零位", np.zeros(7)),
                         ("平举", np.array([0.0, -1.5 if side == 'right' else 1.5,
                                            0.0, 0.0, 0.0, 0.0, 0.0])),
                         ("前伸", np.array([-1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))):
            tau = dyn.gravity_torque(model.clamp(q))
            ratio = dyn.gravity_torque_ratio(model.clamp(q))
            print(f"  {label}: tau_g = {np.round(tau, 3).tolist()} N·m")
            print(f"        占额定 = {np.round(ratio * 100, 1).tolist()} %")
        print()
