#!/usr/bin/env python3
"""X2 上肢运动学模型层。

从 URDF 中把单臂 7 个关节的几何(相对父连杆的 origin xyz/rpy、axis)抽出来,
构成一个不依赖 pinocchio / ROS 的纯 numpy 运动学模型。

之所以自己抽而不是直接用 pinocchio:
  1. 解析 IK 需要的是"肩心 / 肘位 / 腕心 + 各轴方向"这组几何量,
     pinocchio 的 reduced model 不直接暴露,自己展开一次反而更清楚;
  2. 这一层要能在没有 ROS 环境的机器上跑通验证。

坐标系约定:本模型所有量都在 torso_link 系下表达(URDF 里臂的父连杆)。

模型基准
--------
`x2_ultra.urdf` = AgibotTech/agibot_x2_urdf 官方 **X2_URDF-v1.3.0**
(robot name="x2_ultra", md5 d66ca4f97ea39227f6cad18adcbd163e)。
`x2_ultra_internal_T2.5.urdf` 是先前使用的内部版本,仅保留供追溯,不参与计算。
两者手臂运动学几何(origin/rpy/axis)逐字节相同,差异只在肘限位与连杆质量,
详见 README「版本基准」一节。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parent / "x2_ultra.urdf"
DEFAULT_MJCF = Path(__file__).resolve().parent / "x2_ultra.xml"

# 单臂 7 关节顺序。注意 X2 腕部顺序是 yaw -> pitch -> roll,
# 与 A3 的 roll -> pitch -> yaw 不同,迁移时这里是最容易错的地方。
ARM_JOINT_SUFFIX = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)

# 末端连杆(腕 roll 之后的连杆)。装手后的 TCP 需要在此基础上再加一段标定外参,
# 见 ArmModel.tcp_offset 与 ArmModel.tcp_rotation。
EE_LINK_SUFFIX = "wrist_roll_link"


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3 维叉乘。np.cross 为支持任意轴/批量做了大量前置检查,
    在 IK 内层循环里那部分开销比乘法本身还大,这里直接展开。"""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF 的 rpy 是固定轴 X-Y-Z 顺序,等价于 R = Rz @ Ry @ Rx。"""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """rpy_to_matrix 的逆:从 R 反解固定轴 X-Y-Z 的 (roll, pitch, yaw),单位 rad。

    R = Rz(y) @ Ry(p) @ Rx(r),于是 R[2,0] = -sin(p)。
    p = ±90 deg 时 roll 与 yaw 绕同一根轴(万向锁),二者只有和/差有意义 ——
    这时把 roll 归零、全部记到 yaw 上,保证 rpy_to_matrix(matrix_to_rpy(R)) == R。
    """
    rot = np.asarray(rot, float)
    sp = -float(np.clip(rot[2, 0], -1.0, 1.0))
    cp = float(np.hypot(rot[0, 0], rot[1, 0]))
    if cp < 1e-9:                       # 万向锁
        return np.array([0.0, np.arcsin(sp),
                         float(np.arctan2(-rot[0, 1], rot[1, 1]))])
    return np.array([float(np.arctan2(rot[2, 1], rot[2, 2])),
                     float(np.arcsin(sp)),
                     float(np.arctan2(rot[1, 0], rot[0, 0]))])


def rpy_deg(rot: np.ndarray) -> np.ndarray:
    """matrix_to_rpy 的 deg 版,打印用。"""
    return np.degrees(matrix_to_rpy(rot))


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 公式 R = cI + (1-c)kkᵀ + s[k]ₓ。axis 必须已归一化。

    这里把九个元素直接写开,而不是 `np.eye(3) + s*skew + (1-c)*skew@skew`:
    该写法要建 3 个临时 3x3 矩阵并做一次矩阵乘,而本函数是整个 IK 的头号热点
    (profile 里占 track 耗时约 1/3,单帧调用 400+ 次),展开后约快 3 倍。
    """
    kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    txx, tyy, tzz = t * kx * kx, t * ky * ky, t * kz * kz
    txy, txz, tyz = t * kx * ky, t * kx * kz, t * ky * kz
    sx, sy, sz = s * kx, s * ky, s * kz
    return np.array([
        [c + txx, txy - sz, txz + sy],
        [txy + sz, c + tyy, tyz - sx],
        [txz - sy, tyz + sx, c + tzz],
    ])


def log3(rot: np.ndarray) -> np.ndarray:
    """SO(3) 对数映射,返回旋转向量。与 pin.log3 等价。"""
    cos_theta = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        # 小角度:一阶近似即可,避免 sin(theta) 除零
        return np.array([rot[2, 1] - rot[1, 2],
                         rot[0, 2] - rot[2, 0],
                         rot[1, 0] - rot[0, 1]]) * 0.5
    if np.pi - theta < 1e-6:
        # 接近 pi 时上式病态,改用对角元开方定符号
        diag = np.clip((np.diag(rot) + 1.0) * 0.5, 0.0, None)
        vec = np.sqrt(diag)
        idx = int(np.argmax(vec))
        sign_src = np.array([rot[2, 1] - rot[1, 2],
                             rot[0, 2] - rot[2, 0],
                             rot[1, 0] - rot[0, 1]])
        if sign_src[idx] < 0:
            vec = -vec
        return vec / np.linalg.norm(vec) * theta
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([rot[2, 1] - rot[1, 2],
                              rot[0, 2] - rot[2, 0],
                              rot[1, 0] - rot[0, 1]])


def log3_batch(rot: np.ndarray) -> np.ndarray:
    """log3 的批量版。(...,3,3) -> (...,3)。分支与标量版逐一对应。

    三个分支(小角度 / 近 pi / 一般)在批量下全部算出来再用 np.where 选,
    比按样本分派便宜 —— 反正每个分支都只有几次逐元素运算。
    """
    rot = np.asarray(rot, float)
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    skew_vec = np.stack([rot[..., 2, 1] - rot[..., 1, 2],
                         rot[..., 0, 2] - rot[..., 2, 0],
                         rot[..., 1, 0] - rot[..., 0, 1]], axis=-1)
    small = theta < 1e-8
    near_pi = (np.pi - theta) < 1e-6
    sin_theta = np.sin(theta)
    factor = np.where(small | near_pi, 0.0,
                      theta / (2.0 * np.where(sin_theta == 0.0, 1.0, sin_theta)))
    out = factor[..., None] * skew_vec
    out = np.where(small[..., None], skew_vec * 0.5, out)
    if np.any(near_pi):
        diag = np.clip((np.stack([rot[..., 0, 0], rot[..., 1, 1], rot[..., 2, 2]],
                                 axis=-1) + 1.0) * 0.5, 0.0, None)
        vec = np.sqrt(diag)
        idx = np.argmax(vec, axis=-1)
        pick = np.take_along_axis(skew_vec, idx[..., None], axis=-1)
        vec = np.where(pick < 0.0, -vec, vec)
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        alt = vec / np.where(norm < 1e-15, 1.0, norm) * theta[..., None]
        out = np.where(near_pi[..., None], alt, out)
    return out


@dataclass(frozen=True)
class JointGeom:
    """一个转动关节相对其父连杆的固定变换 + 转轴。"""
    name: str
    origin_xyz: np.ndarray      # 父连杆系下的位置
    origin_rot: np.ndarray      # 父连杆系下的姿态
    axis: np.ndarray            # 关节自身系下的转轴(已归一化)
    lower: float
    upper: float
    effort: float
    velocity: float


def parse_urdf_joints(urdf_path: Path) -> Dict[str, JointGeom]:
    """极简 URDF 关节解析。只取 revolute/continuous,够运动学用。"""
    text = Path(urdf_path).read_text(encoding="utf-8")
    out: Dict[str, JointGeom] = {}
    pattern = re.compile(
        r'<joint\s+name="([^"]+)"\s+type="(revolute|continuous)"\s*>(.*?)</joint>',
        re.S,
    )
    for match in pattern.finditer(text):
        name, _jtype, body = match.groups()
        origin = re.search(r"<origin([^/>]*)/?>", body)
        xyz = np.zeros(3)
        rot = np.eye(3)
        if origin:
            attrs = origin.group(1)
            mx = re.search(r'xyz="([^"]+)"', attrs)
            mr = re.search(r'rpy="([^"]+)"', attrs)
            if mx:
                xyz = np.array([float(v) for v in mx.group(1).split()])
            if mr:
                rot = rpy_to_matrix([float(v) for v in mr.group(1).split()])
        axis_m = re.search(r'<axis[^/>]*xyz="([^"]+)"', body)
        axis = (np.array([float(v) for v in axis_m.group(1).split()])
                if axis_m else np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

        def _attr(key: str, default: float) -> float:
            m = re.search(rf'<limit[^/>]*{key}="([^"]+)"', body)
            return float(m.group(1)) if m else default

        out[name] = JointGeom(
            name=name,
            origin_xyz=xyz,
            origin_rot=rot,
            axis=axis,
            lower=_attr("lower", -np.pi),
            upper=_attr("upper", np.pi),
            effort=_attr("effort", 0.0),
            velocity=_attr("velocity", 0.0),
        )
    return out


def parse_mjcf_ctrlrange(mjcf_path: Path) -> Dict[str, float]:
    """从 MJCF 取每个关节 motor 的 ctrlrange 上限(N·m)。

    存在的理由:官方 v1.3.0 里 URDF 与 MJCF 对腕部力矩上限口径不一致 ——
    URDF `effort="4.8"`,MJCF `ctrlrange="-2.2 2.2"`。按 URDF 限幅是偏乐观的,
    真按 4.8 下发可能直接触发驱动器保护。这里把两者都读进来,由调用方取小。
    """
    path = Path(mjcf_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    out: Dict[str, float] = {}
    for match in re.finditer(r"<motor\b([^>]*)>", text):
        attrs = match.group(1)
        joint = re.search(r'joint="([^"]+)"', attrs)
        rng = re.search(r'ctrlrange="\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*"', attrs)
        if joint and rng:
            out[joint.group(1)] = max(abs(float(rng.group(1))), abs(float(rng.group(2))))
    return out


class ArmModel:
    """X2 单臂 7-DOF 运动学。

    所有位姿都在 torso_link 系下。q 的顺序即 ARM_JOINT_SUFFIX。
    """

    def __init__(
        self,
        side: str = "right",
        urdf_path: Path = DEFAULT_URDF,
        tcp_offset: Optional[np.ndarray] = None,
        mjcf_path: Optional[Path] = DEFAULT_MJCF,
        tcp_rotation: Optional[np.ndarray] = None,
    ):
        if side not in ("left", "right"):
            raise ValueError("side 必须是 'left' 或 'right'")
        self.side = side
        self.urdf_path = Path(urdf_path)
        joints = parse_urdf_joints(self.urdf_path)
        self.joint_names: Tuple[str, ...] = tuple(f"{side}_{s}" for s in ARM_JOINT_SUFFIX)
        missing = [n for n in self.joint_names if n not in joints]
        if missing:
            raise RuntimeError(f"URDF 中缺少关节: {missing}")
        self.geoms: List[JointGeom] = [joints[n] for n in self.joint_names]
        self.ee_link = f"{side}_{EE_LINK_SUFFIX}"

        self.q_min = np.array([g.lower for g in self.geoms])
        self.q_max = np.array([g.upper for g in self.geoms])
        self.tau_max = np.array([g.effort for g in self.geoms])       # URDF effort
        self.dq_max = np.array([g.velocity for g in self.geoms])

        # MJCF ctrlrange。缺文件时退回 URDF 值,tau_max_safe 便等于 tau_max。
        ctrl = parse_mjcf_ctrlrange(mjcf_path) if mjcf_path else {}
        self.tau_max_mjcf = np.array([ctrl.get(n, np.nan) for n in self.joint_names])
        self.tau_max_safe = np.where(np.isnan(self.tau_max_mjcf),
                                     self.tau_max,
                                     np.minimum(self.tau_max, self.tau_max_mjcf))
        # 力矩下发一律用 tau_max_safe;tau_max 只作为 URDF 原始值留档。
        self.tau_disagreement = [
            (n, float(a), float(b))
            for n, a, b in zip(self.joint_names, self.tau_max, self.tau_max_mjcf)
            if not np.isnan(b) and abs(a - b) > 1e-9
        ]

        # T_wrist_tcp: 平移在 wrist_roll_link 系表达,旋转将 TCP 向量变换到腕系。
        # 必须同时去掉这两部分才能从 TCP 目标位姿反推腕部目标。
        self.tcp_offset = (np.zeros(3) if tcp_offset is None
                           else np.array(tcp_offset, dtype=float, copy=True))
        self.tcp_rotation = (np.eye(3) if tcp_rotation is None
                             else np.array(tcp_rotation, dtype=float, copy=True))
        if self.tcp_offset.shape != (3,) or not np.isfinite(self.tcp_offset).all():
            raise ValueError("tcp_offset 必须是有限的 3 维向量,单位 m")
        if (self.tcp_rotation.shape != (3, 3)
                or not np.isfinite(self.tcp_rotation).all()
                or not np.allclose(self.tcp_rotation.T @ self.tcp_rotation,
                                   np.eye(3), atol=1e-6, rtol=0.0)
                or not np.isclose(np.linalg.det(self.tcp_rotation), 1.0,
                                  atol=1e-6, rtol=0.0)):
            raise ValueError("tcp_rotation 必须是有限的 3x3 SO(3) 旋转矩阵")

        self._cache_batch_tables()
        self._cache_zero_geometry()

    # ---------- 正向运动学 ----------

    def joint_frames(self, q: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """返回每个关节原点在 torso 系下的位置,以及转过该关节角之后的姿态。"""
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"q 必须是 7 维,收到 {q.shape}")
        cos_q = np.cos(q)
        sin_q = np.sin(q)
        # origin_rot @ Rot(axis, q) 的三表线性组合,见 _cache_batch_tables
        step = (cos_q[:, None, None] * self._bat_a
                + sin_q[:, None, None] * self._bat_b
                + (1.0 - cos_q)[:, None, None] * self._bat_c)
        n = len(self.geoms)
        positions = np.empty((n, 3))
        rotations = np.empty((n, 3, 3))
        pos = np.zeros(3)
        rot = np.eye(3)
        ox = self._bat_ox
        for i in range(n):
            pos = pos + rot @ ox[i]
            positions[i] = pos
            rot = rot @ step[i]
            rotations[i] = rot
        return list(positions), list(rotations)

    def axes_at(self, q: np.ndarray) -> List[np.ndarray]:
        """各关节转轴在 torso 系下的方向(转轴本身不受该关节自转影响)。"""
        _, rotations = self.joint_frames(q)
        axes: List[np.ndarray] = []
        for idx, geom in enumerate(self.geoms):
            rot_before = np.eye(3) if idx == 0 else rotations[idx - 1]
            axes.append(rot_before @ geom.origin_rot @ geom.axis)
        return axes

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """末端 TCP 的位置与姿态。"""
        positions, rotations = self.joint_frames(q)
        rot = rotations[-1]
        pos = positions[-1] + rot @ self.tcp_offset
        return pos, rot @ self.tcp_rotation

    def wrist_center(self, q: np.ndarray) -> np.ndarray:
        """腕三轴交点。解析 IK 的关键中间量。

        用固连在末端连杆上的常量偏置变换,而不是直接取 positions[6] ——
        两者对 X2 恰好相同(偏置为 0),但对偏置非零的机型才正确。
        """
        positions, rotations = self.joint_frames(q)
        return positions[6] + rotations[6] @ self.wrist_center_local

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x7 几何雅可比,LOCAL_WORLD_ALIGNED 约定(与 A3 demo 一致)。"""
        positions, rotations = self.joint_frames(q)
        ee_pos, _ = self.forward_kinematics(q)
        jac = np.zeros((6, 7))
        for idx, geom in enumerate(self.geoms):
            rot_before = np.eye(3) if idx == 0 else rotations[idx - 1]
            axis_world = rot_before @ geom.origin_rot @ geom.axis
            jac[:3, idx] = cross3(axis_world, ee_pos - positions[idx])
            jac[3:, idx] = axis_world
        return jac

    # ---------- 批量正向运动学 ----------

    def joint_frames_batch(self, q_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """joint_frames 的批量版。q_arr (...,7) -> (positions (...,7,3), rotations (...,7,3,3))。

        7 个关节的串联是真串行,躲不掉;但"每个关节的旋转矩阵"和"N 组样本"
        这两层都能广播掉,于是不论 N 多大,numpy 调用次数恒为 ~20 次。
        单点 0.028 ms、100 点 0.05 ms —— 这就是批量化的全部收益来源。
        """
        q_arr = np.asarray(q_arr, float)
        if q_arr.shape[-1] != 7:
            raise ValueError(f"q_arr 最后一维必须是 7,收到 {q_arr.shape}")
        lead = q_arr.shape[:-1]
        flat = q_arr.reshape(-1, 7)
        num = flat.shape[0]
        cos_q = np.cos(flat)
        sin_q = np.sin(flat)
        step = (cos_q[..., None, None] * self._bat_a
                + sin_q[..., None, None] * self._bat_b
                + (1.0 - cos_q)[..., None, None] * self._bat_c)      # (N,7,3,3)
        n = len(self.geoms)
        positions = np.empty((num, n, 3))
        rotations = np.empty((num, n, 3, 3))
        pos = np.zeros((num, 3))
        rot = np.broadcast_to(np.eye(3), (num, 3, 3)).copy()
        ox = self._bat_ox
        for i in range(n):
            pos = pos + rot @ ox[i]
            positions[:, i] = pos
            rot = rot @ step[:, i]
            rotations[:, i] = rot
        return (positions.reshape(lead + (n, 3)),
                rotations.reshape(lead + (n, 3, 3)))

    def fk_batch(self, q_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """末端 TCP 位姿的批量版。返回 (pos (...,3), rot (...,3,3))。"""
        positions, rotations = self.joint_frames_batch(q_arr)
        rot = rotations[..., -1, :, :]
        pos = positions[..., -1, :] + rot @ self.tcp_offset
        return pos, rot @ self.tcp_rotation

    def axes_batch(self, q_arr: np.ndarray) -> np.ndarray:
        """各关节转轴在 torso 系下的方向,批量版。返回 (...,7,3)。"""
        _, rotations = self.joint_frames_batch(q_arr)
        lead = rotations.shape[:-3]
        before = np.empty_like(rotations)
        before[..., 0, :, :] = np.eye(3)
        before[..., 1:, :, :] = rotations[..., :-1, :, :]
        # (...,7,3,3) @ (7,3) 逐关节取自己的轴
        return np.einsum('...ijk,ik->...ij', before, self._bat_axis_local)

    def jacobian_batch(self, q_arr: np.ndarray) -> np.ndarray:
        """6x7 几何雅可比的批量版,返回 (...,6,7)。约定与 jacobian 一致。"""
        return self.jacobian_from_frames(*self.joint_frames_batch(q_arr))

    def jacobian_from_frames(self, positions: np.ndarray,
                             rotations: np.ndarray) -> np.ndarray:
        """已有 joint_frames_batch 结果时直接拼雅可比,省掉一次串联 FK。

        polish 的增广雅可比要在同一批关节角上同时取"雅可比 + SEW 角",
        走这个入口就只需一次 joint_frames_batch。
        """
        rot_ee = rotations[..., -1, :, :]
        ee_pos = positions[..., -1, :] + rot_ee @ self.tcp_offset
        before = np.empty_like(rotations)
        before[..., 0, :, :] = np.eye(3)
        before[..., 1:, :, :] = rotations[..., :-1, :, :]
        axis_world = np.einsum('...ijk,ik->...ij', before, self._bat_axis_local)  # (...,7,3)
        arm = ee_pos[..., None, :] - positions                                    # (...,7,3)
        lin = np.cross(axis_world, arm)
        jac = np.empty(positions.shape[:-2] + (6, 7))
        jac[..., :3, :] = np.swapaxes(lin, -1, -2)
        jac[..., 3:, :] = np.swapaxes(axis_world, -1, -2)
        return jac

    def manipulability_batch(self, q_arr: np.ndarray) -> np.ndarray:
        """Yoshikawa 可操作度的批量版,返回 (...)。"""
        jac = self.jacobian_batch(q_arr)
        gram = jac @ np.swapaxes(jac, -1, -2)
        return np.sqrt(np.maximum(np.linalg.det(gram), 0.0))

    def within_limits_batch(self, q_arr: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """within_limits 的批量版,返回 bool (...)。"""
        q_arr = np.asarray(q_arr, float)
        return (np.all(q_arr >= self.q_min - tol, axis=-1)
                & np.all(q_arr <= self.q_max + tol, axis=-1))

    def limit_margin_batch(self, q_arr: np.ndarray) -> np.ndarray:
        """limit_margin 的批量版,返回 (...)。"""
        q_arr = np.asarray(q_arr, float)
        return np.minimum(q_arr - self.q_min, self.q_max - q_arr).min(axis=-1)

    # ---------- SRS 几何量 ----------

    def _cache_zero_geometry(self) -> None:
        """在零位提取解析 IK 需要的 SRS 常量,并校验球腕/球肩假设。"""
        q0 = np.zeros(7)
        positions, rotations = self.joint_frames(q0)
        axes = self.axes_at(q0)

        # 肩心/腕心都取三轴的最小二乘公共交点,不取任何单个关节原点
        self.shoulder_center, self.shoulder_axis_residual = \
            self._sphere_center(positions, axes, (0, 1, 2))
        self.wrist_center_zero, self.wrist_axis_residual = \
            self._sphere_center(positions, axes, (4, 5, 6))
        self.elbow_pos_zero = positions[3].copy()

        # 腕心在末端连杆(link 7)自身系下的常量表达,供由目标位姿反推腕心用
        self.wrist_center_local = rotations[6].T @ (self.wrist_center_zero - positions[6])

        self.upper_arm_len = float(np.linalg.norm(self.elbow_pos_zero - self.shoulder_center))
        self.fore_arm_len = float(np.linalg.norm(self.wrist_center_zero - self.elbow_pos_zero))
        self.max_reach = self.upper_arm_len + self.fore_arm_len
        self.min_reach = abs(self.upper_arm_len - self.fore_arm_len)

    def _cache_batch_tables(self) -> None:
        """预乘出批量 FK 需要的三张常量表。

        关键恒等式:每个关节的"固定变换 x 关节自转"可以写成三张常量矩阵的
        线性组合 —— 把 Rodrigues 的 R = cI + t·kkᵀ + s·[k]ₓ 左乘 origin_rot:

            origin_rot @ Rot(k, q) = cos(q)·A + sin(q)·B + (1-cos q)·C
            A = origin_rot,  B = origin_rot @ [k]ₓ,  C = (origin_rot @ k) kᵀ

        于是 N 组关节角、7 个关节的全部旋转可以用**一次**广播算出来
        ((N,7,1,1) 乘 (7,3,3)),不再是 7N 次 axis_angle_to_matrix 调用。
        这是把 psi 扫描从标量循环搬到批量的地基。
        """
        n = len(self.geoms)
        self._bat_ox = np.array([g.origin_xyz for g in self.geoms], float)     # (7,3)
        mat_a = np.empty((n, 3, 3))
        mat_b = np.empty((n, 3, 3))
        mat_c = np.empty((n, 3, 3))
        for i, geom in enumerate(self.geoms):
            kx, ky, kz = (float(v) for v in geom.axis)
            skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
            mat_a[i] = geom.origin_rot
            mat_b[i] = geom.origin_rot @ skew
            mat_c[i] = np.outer(geom.origin_rot @ geom.axis, geom.axis)
        self._bat_a, self._bat_b, self._bat_c = mat_a, mat_b, mat_c
        # 零位各关节轴在 torso 系下的方向,雅可比批量版要用
        self._bat_axis_local = np.array([g.origin_rot @ g.axis for g in self.geoms], float)

    @staticmethod
    def _line_distance(p1: np.ndarray, d1: np.ndarray,
                       p2: np.ndarray, d2: np.ndarray) -> float:
        cross = cross3(d1, d2)
        norm = np.linalg.norm(cross)
        if norm < 1e-9:  # 平行
            return float(np.linalg.norm(cross3(p2 - p1, d1)))
        return float(abs(np.dot(p2 - p1, cross / norm)))

    @staticmethod
    def _sphere_center(positions, axes, idx3) -> Tuple[np.ndarray, float]:
        """求三条关节轴的公共交点(最小二乘),返回 (球心, 最大点到轴距离)。

        为什么不能直接拿某个关节原点当球心 —— 这是本模型踩过的一个真实坑:
        X2 的 shoulder_roll 关节 origin 带 -0.0005 m 的 x 偏置,而 shoulder_pitch
        轴是 y 方向,所以 roll 关节原点偏离 pitch 轴 0.5 mm。若直接取它作球心,
        `W(q) = S + R_S·w(q4)` 这个 IK 基本假设就会有 ~0.94 mm 的系统误差。

        另外注意:判断"是否球关节"必须用**点到轴距离**,不能用两两轴线距离。
        三条轴线两两距离都可以是 0(各自相交)而并不共点 —— 这正是上述偏置
        被掩盖的原因(两两距离只有 0.0001 mm,看起来完美)。

        最小化 sum_i |(I - d_i d_i^T)(x - p_i)|^2,法方程 A x = b,
        A = sum_i (I - d_i d_i^T),b = A_i p_i 之和。
        """
        mat = np.zeros((3, 3))
        vec = np.zeros(3)
        for i in idx3:
            axis = axes[i]
            proj = np.eye(3) - np.outer(axis, axis)
            mat += proj
            vec += proj @ positions[i]
        center = np.linalg.solve(mat, vec) if abs(np.linalg.det(mat)) > 1e-12 \
            else np.linalg.lstsq(mat, vec, rcond=None)[0]
        residual = max(
            float(np.linalg.norm(cross3(center - positions[i], axes[i])))
            for i in idx3
        )
        return center, residual

    # ---------- 限位 / 度量 ----------

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(q, float), self.q_min), self.q_max)

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        q = np.asarray(q, float)
        return bool(np.all(q >= self.q_min - tol) and np.all(q <= self.q_max + tol))

    def limit_margin(self, q: np.ndarray) -> float:
        """距离最近限位的裕度(rad),负值表示越界。"""
        q = np.asarray(q, float)
        return float(np.min(np.minimum(q - self.q_min, self.q_max - q)))

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa 可操作度 sqrt(det(J Jᵀ))。越接近 0 越靠近奇异。"""
        jac = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))

    def describe(self) -> str:
        return (
            f"X2 {self.side} arm  ({self.urdf_path.name})\n"
            f"  关节顺序   : {', '.join(s.replace('_joint', '') for s in ARM_JOINT_SUFFIX)}\n"
            f"  肩心(torso): {np.round(self.shoulder_center, 5).tolist()}\n"
            f"  上臂长度   : {self.upper_arm_len * 1000:.2f} mm\n"
            f"  前臂长度   : {self.fore_arm_len * 1000:.2f} mm\n"
            f"  伸展范围   : {self.min_reach * 1000:.2f} ~ {self.max_reach * 1000:.2f} mm (肩心到腕心)\n"
            f"  球肩残差   : {self.shoulder_axis_residual * 1000:.4f} mm  (球心到三轴最大距离)\n"
            f"  球腕残差   : {self.wrist_axis_residual * 1000:.4f} mm\n"
            f"  力矩上限   : URDF {self.tau_max.tolist()} N·m\n"
            f"             : MJCF {np.round(self.tau_max_mjcf, 3).tolist()} N·m\n"
            f"             : 取小 {self.tau_max_safe.tolist()} N·m  <- 实际限幅用这组\n"
            f"  速度上限   : {self.dq_max.tolist()} rad/s"
        )


if __name__ == "__main__":
    for _side in ("right", "left"):
        print(ArmModel(_side).describe())
        print()
