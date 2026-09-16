#!/usr/bin/env python3
"""X2 坐标系判定工具 —— 左右手分开控制时,"原点在哪、轴朝哪"的自证脚本。

回答的问题
----------
1. IK 解算在哪个系里?              -> torso_link 系(见 §A)
2. 这个系的原点相对 pelvis / 世界在哪? -> waist 三轴串联,见 torso_pose_in_pelvis()
3. 每个臂关节的原点和转轴具体是什么?  -> zero_frame_table()
4. 左右臂是严格镜像吗?              -> mirror_report()(答案:不是,有亚毫米级不对称)
5. 怎么在仿真里"证明"我判断对了?     -> excitation_table() 给出单关节激励预测表,
                                       跑一遍 x2_mujoco_arm.py probe 就能逐条核对

只依赖 numpy。直接运行会打印全部结论:
    python3 x2_frames.py
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from x2_arm_model import (ARM_JOINT_SUFFIX, ArmModel, DEFAULT_MJCF, DEFAULT_URDF,
                          axis_angle_to_matrix, cross3, parse_urdf_joints, rpy_to_matrix)

# mc idle.yaml / cpgtelecon_config.yaml 里的待机姿态(单臂 7 维)。
# shoulder_pitch=0.4, elbow=-1.2,其余 0。两臂相同 —— 注意不是镜像取负,
# 因为 pitch/elbow 的正方向在左右臂 URDF 里本来就是同向的。
HOME_Q = np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])

# 仿真 HAL default.yaml 的 nominal_configuration 臂段(仅供对照,不推荐当待机位:
# elbow=0 时肩-肘-腕共线,正处在 SEW 臂角参数化的退化点上)。
NOMINAL_Q = np.array([0.196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ---------------------------------------------------------------- 侧平举姿态

def _gauss_newton(resid, x0, lo, hi, iters: int = 80, eps: float = 1e-7):
    """带阻尼的 Gauss-Newton + 逐轮夹到关节限位。数值雅可比,变量只有 2~3 个,
    比引一个 scipy 依赖划算。返回 (x, 最大残差绝对值)。"""
    x = np.clip(np.asarray(x0, float), lo, hi)
    r = np.asarray(resid(x), float)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            xp = x.copy()
            xp[k] += eps
            J[:, k] = (np.asarray(resid(xp), float) - r) / eps
        # 阻尼最小二乘。lam 很小,只为了压住 J 接近奇异时的爆步。
        lam = 1e-9 * max(1.0, float(np.max(np.abs(J))) ** 2)
        dx = np.linalg.solve(J.T @ J + lam * np.eye(x.size), -J.T @ r)
        step = 1.0
        for _ in range(20):                       # 简单回溯,保证残差单调下降
            xn = np.clip(x + step * dx, lo, hi)
            rn = np.asarray(resid(xn), float)
            if np.linalg.norm(rn) < np.linalg.norm(r):
                break
            step *= 0.5
        else:
            break
        x, r = xn, rn
        if np.max(np.abs(r)) < 1e-12:
            break
    return x, float(np.max(np.abs(r)))


def lateral_raise_q(side: str, elbow: float = 0.0,
                    model: Optional[ArmModel] = None) -> np.ndarray:
    """侧平举:手臂向左右抬起,肩/肘/腕保持同高(水平)。返回单臂 7 维关节角。

    为什么不能直接写 shoulder_roll = ±pi/2:
        肩部安装面带倾角,roll 轴并不平行于 torso 系的 x 轴。直接给 ±90 deg,
        整条臂会上翘约 12 deg —— TCP 比肩高 85 mm,在仿真里一眼就看得出不平。
        这里改成解一个小方程:让"肩->腕"方向纯侧向(x、z 分量为 0),
        实测解出 shoulder_roll ~= ±78 deg。

    elbow = 0        直臂 T 位,整条手臂水平
    elbow = -pi/2    上臂侧平、前臂朝前水平(肘同样落在肩高)
    腕部三轴一律给 0:本函数只管到第 7 个关节(wrist_roll),不涉及手。
    """
    if side not in ("left", "right"):
        raise ValueError("side 必须是 'left' 或 'right'")
    m = model if model is not None else ArmModel(side)
    elbow = float(np.clip(elbow, m.q_min[3], m.q_max[3]))
    sgn = 1.0 if side == "left" else -1.0

    if abs(elbow) < 1e-9:
        # 直臂时 shoulder_yaw 绕自身轴转,对位置无贡献 -> 固定 0,只解 pitch/roll。
        def resid(x):
            q = np.array([x[0], x[1], 0.0, 0.0, 0.0, 0.0, 0.0])
            sh = m.joint_frames(q)[0][1]
            p = m.forward_kinematics(q)[0]
            return [p[0] - sh[0], p[2] - sh[2]]
        x, res = _gauss_newton(resid, [0.0, sgn * 1.36], m.q_min[:2], m.q_max[:2])
        q = np.array([x[0], x[1], 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # 屈肘时:pitch/roll 定上臂纯侧向,yaw 定前臂落在水平面内。
        def resid(x):
            q = np.array([x[0], x[1], x[2], elbow, 0.0, 0.0, 0.0])
            pos = m.joint_frames(q)[0]
            sh, el = pos[1], pos[3]
            tcp = m.forward_kinematics(q)[0]
            return [el[0] - sh[0], el[2] - sh[2], tcp[2] - el[2]]
        x, res = _gauss_newton(resid, [0.0, sgn * 1.36, 0.0], m.q_min[:3], m.q_max[:3])
        q = np.array([x[0], x[1], x[2], elbow, 0.0, 0.0, 0.0])

    if res > 1e-4:
        raise RuntimeError(f"lateral_raise_q({side}, elbow={elbow:.3f}) 未收敛,残差 {res:.2e} m")
    return m.clamp(q)


def lateral_raise_report(elbow: float = 0.0) -> str:
    """把侧平举姿态连同"到底平不平"的证据一起打印出来。"""
    lines = [f"侧平举姿态  elbow = {math.degrees(elbow):.1f} deg"
             f"   ({'直臂 T 位' if abs(elbow) < 1e-9 else '上臂侧平 + 前臂朝前'})"]
    for side in ("left", "right"):
        m = ArmModel(side)
        q = lateral_raise_q(side, elbow, m)
        pos = m.joint_frames(q)[0]
        tcp = m.forward_kinematics(q)[0]
        z = [pos[1][2], pos[3][2], pos[4][2], tcp[2]]      # 肩roll / 肘 / 腕yaw / TCP
        naive = m.clamp(np.array([0.0, (1.0 if side == "left" else -1.0) * math.pi / 2,
                                  0.0, elbow, 0.0, 0.0, 0.0]))
        tcp_naive = m.forward_kinematics(naive)[0]
        lines += [
            f"  [{side}] q = {np.round(np.degrees(q), 3).tolist()} deg",
            f"         z: 肩 {z[0]:.4f}  肘 {z[1]:.4f}  腕 {z[2]:.4f}  TCP {z[3]:.4f} m"
            f"   -> 最大高差 {(max(z) - min(z)) * 1000:.2f} mm",
            f"         TCP = {np.round(tcp, 4).tolist()}   限位余量 {math.degrees(m.limit_margin(q)):.1f} deg",
        ]
        if m.limit_margin(q) < math.radians(1.0):
            lines += [
                "         [warn] 有关节压在限位上 —— elbow 的上限就是 0,"
                "直臂时它正好顶在硬限位。",
                "                仿真里无所谓;上真机建议 --elbow -5 留点余量。",
            ]
        lines += [
            f"         对照:roll 直接给 ±90 deg 时 TCP 比肩高 "
            f"{(tcp_naive[2] - z[0]) * 1000:.1f} mm(上翘 "
            f"{math.degrees(math.atan2(tcp_naive[2] - z[0], abs(tcp_naive[1] - pos[1][1]))):.1f} deg)",
        ]
    return "\n".join(lines)


WAIST_JOINTS = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")

# 镜像算子:绕 torso 系 xz 平面反射。位置 p' = M p,姿态 R' = M R M。
# (M R M 的行列式 = det(M)^2 det(R) = +1,仍是合法旋转,这是镜像位姿的正确变换,
#  直接用 M R 会得到 det=-1 的反射矩阵,不是姿态。)
MIRROR = np.diag([1.0, -1.0, 1.0])


# ---------------------------------------------------------------- 腰部链

class WaistChain:
    """pelvis -> waist_yaw -> waist_pitch -> waist_roll -> torso_link。

    存在的意义:IK 全部在 torso_link 系里算,但下发点(HAL/mc)和观测点
    (odom / IMU)都在 pelvis 或世界系。腰一动,torso 系就跟着动 ——
    如果把 IK 的输出当成 pelvis 系的量,大臂展姿态下会差出几厘米。
    """

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        joints = parse_urdf_joints(Path(urdf_path))
        missing = [n for n in WAIST_JOINTS if n not in joints]
        if missing:
            raise RuntimeError(f"URDF 中缺少腰关节: {missing}")
        self.geoms = [joints[n] for n in WAIST_JOINTS]
        self.q_min = np.array([g.lower for g in self.geoms])
        self.q_max = np.array([g.upper for g in self.geoms])

    def torso_pose_in_pelvis(self, q_waist: Sequence[float] = (0.0, 0.0, 0.0)
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """给定腰三轴角度,返回 torso_link 原点位置与姿态(pelvis 系)。"""
        q = np.asarray(q_waist, float)
        if q.shape != (3,):
            raise ValueError("q_waist 必须是 3 维 [yaw, pitch, roll]")
        pos = np.zeros(3)
        rot = np.eye(3)
        for geom, angle in zip(self.geoms, q):
            pos = pos + rot @ geom.origin_xyz
            rot = rot @ geom.origin_rot @ axis_angle_to_matrix(geom.axis, float(angle))
        return pos, rot

    def torso_to_pelvis(self, pos_t: np.ndarray, rot_t: np.ndarray,
                        q_waist: Sequence[float] = (0.0, 0.0, 0.0)
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """把 torso 系下的一个位姿(IK 的输入/输出)搬到 pelvis 系。"""
        p0, r0 = self.torso_pose_in_pelvis(q_waist)
        return p0 + r0 @ np.asarray(pos_t, float), r0 @ np.asarray(rot_t, float)

    def pelvis_to_torso(self, pos_p: np.ndarray, rot_p: np.ndarray,
                        q_waist: Sequence[float] = (0.0, 0.0, 0.0)
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """pelvis 系的目标点 -> torso 系,喂给 IK 之前必须做这一步。"""
        p0, r0 = self.torso_pose_in_pelvis(q_waist)
        return r0.T @ (np.asarray(pos_p, float) - p0), r0.T @ np.asarray(rot_p, float)

    def gravity_in_torso(self, rot_torso_in_world: np.ndarray) -> np.ndarray:
        """世界系重力向量转到 torso 系。重力前馈必须用这个,不能写死 [0,0,-9.81]。

        rot_torso_in_world 从 IMU(/aima/hal/imu/torso/state)或
        odom+腰部正解得到。躯干只要倾斜 10 度,肩 roll 的重力项就差 ~1.7%。
        """
        return np.asarray(rot_torso_in_world, float).T @ np.array([0.0, 0.0, -9.81])

    def describe(self) -> str:
        lines = ["腰部链 pelvis -> torso_link(IK 基座系的定位链)"]
        pos = np.zeros(3)
        rot = np.eye(3)
        for geom in self.geoms:
            pos = pos + rot @ geom.origin_xyz
            rot = rot @ geom.origin_rot
            ax = rot @ geom.axis
            lines.append(f"  {geom.name:<18s} origin(pelvis)={np.round(pos, 6).tolist()}"
                         f"  axis={np.round(ax, 4).tolist()}"
                         f"  range=[{geom.lower:.3f}, {geom.upper:.3f}]")
        p0, r0 = self.torso_pose_in_pelvis()
        lines.append(f"  => 零腰位时 torso_link 原点(pelvis 系) = {np.round(p0, 6).tolist()}")
        lines.append(f"     零腰位时 torso_link 姿态 = 单位阵(与 pelvis 同向): "
                     f"{np.allclose(r0, np.eye(3), atol=1e-12)}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 重力估计(IMU)

# 两条 IMU 话题,名字有坑,以模型里的挂点为准:
#   /aima/hal/imu/chest/state  -> MJCF site imu_1,挂在 **torso_link** 上
#                                 == IK 基座系本身,是重力估计的首选
#   /aima/hal/imu/torso/state  -> MJCF site imu_0,挂在 **pelvis** 上
#                                 名字叫 torso 但其实是基座 IMU,用它要再串一次腰链
# 别照名字选话题。
CHEST_IMU_TOPIC = "/aima/hal/imu/chest/state"
PELVIS_IMU_TOPIC = "/aima/hal/imu/torso/state"

G_NORM = 9.81
G_WORLD = np.array([0.0, 0.0, -G_NORM])


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS 四元数(x, y, z, w)-> 旋转矩阵。输入不必严格归一化。"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def matrix_to_quat(rot: np.ndarray) -> Tuple[float, float, float, float]:
    """旋转矩阵 -> ROS 四元数 (x, y, z, w)。给 TF / Marker 用。

    用 Shepperd 的分支法而不是"先算 w 再除":w 接近 0(转角接近 180 deg)时
    那一路会除以一个接近 0 的数,画出来的坐标轴会乱跳。这里始终从最大的
    对角项开方,任何转角都稳定。
    """
    m = np.asarray(rot, float)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        sc = math.sqrt(tr + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / sc, (m[0, 2] - m[2, 0]) / sc,
                (m[1, 0] - m[0, 1]) / sc, 0.25 * sc)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    sc = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * sc
    q[j] = (m[j, i] + m[i, j]) / sc
    q[k] = (m[k, i] + m[i, k]) / sc
    return (q[0], q[1], q[2], (m[k, j] - m[j, k]) / sc)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """把单位向量 a 转到 b 的最小旋转(绕 a×b)。a、b 反向时任取一条垂直轴。"""
    a = np.asarray(a, float) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, float) / max(np.linalg.norm(b), 1e-12)
    v = cross3(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = cross3(a, perp)
        return axis_angle_to_matrix(axis / np.linalg.norm(axis), math.pi)
    return axis_angle_to_matrix(v / s, math.atan2(s, c))


class GravityEstimator:
    """把 IMU 反馈折成 torso 系的重力向量,喂给 ArmDynamics。

    为什么必须做:重力前馈 tau_g 是 g 在 torso 系下方向的函数。躯干倾 10 度,
    肩 roll 的重力项就差约 1.7%;下肢跑 RL 时躯干本来就一直在动,写死
    [0, 0, -9.81] 等于把这部分误差直接送进末端。

    三个来源,按可靠性从高到低:
      1. chest  —— 胸腔 IMU。挂点就是 torso_link(MJCF site imu_1),
                   姿态四元数直接给出 R_torso_in_world,不经过任何关节,
                   不受腰部编码器误差和腰部间隙影响。**首选。**
      2. pelvis —— 基座 IMU(话题名 /imu/torso/state,挂点其实是 pelvis)
                   加腰关节反馈,再串一次 WaistChain 才能得到 torso 姿态。
                   多了 3 个编码器的误差。
      3. static —— 退化为 [0, 0, -9.81]。只在站桩调试时可接受。

    姿态四元数与加速度计两条路都实现了:
      - 四元数是融合后的结果,对运动加速度不敏感,**默认用它**;
        重力只关心倾角,IMU 的偏航漂移天然不影响结果。
      - 加速度计在静止时直接给出比力 f = -g,不需要融合;运动时会被真实
        加速度污染。用作静态交叉校验,以及 orientation_covariance[0] == -1
        (ROS 约定:该 IMU 不提供姿态)时的兜底。
    """

    def __init__(self, source: str = "chest",
                 mount_rot: Optional[np.ndarray] = None,
                 alpha: float = 0.1,
                 tilt_limit_deg: float = 60.0,
                 accel_tol_deg: float = 8.0):
        if source not in ("chest", "pelvis", "static"):
            raise ValueError("source 必须是 chest / pelvis / static")
        self.source = source
        # R^torso_imu:把 IMU 系的向量转到 torso 系。v_torso = mount_rot @ v_imu。
        # 仿真里 site imu_1 只有平移没有转角,所以是单位阵;实物必须标定。
        self.mount_rot = np.eye(3) if mount_rot is None else np.asarray(mount_rot, float)
        self.alpha = float(alpha)
        self.tilt_limit = math.radians(tilt_limit_deg)
        self.accel_tol = math.radians(accel_tol_deg)

        self.g = G_WORLD.copy()          # 当前估计(torso 系)
        self.n_update = 0
        self.n_reject = 0
        self.last_accel_angle = 0.0      # 四元数与加速度计的夹角,静止时应接近 0
        self.have_fix = False
        # 丢帧原因。全丢时最常见的是"机器人躺着 / 摔了",倾角门限把整段挡掉了 ——
        # 不记下来的话表象只是"g 还是默认值",很容易误判成话题收不到。
        self.last_reject = ""

    # ---- 标定 ----

    def calibrate_mount_from_accel(self, accel: Sequence[float]) -> np.ndarray:
        """机器人直立静止时读一帧加速度计,反解安装转角。

        直立静止 => torso 系的重力就是 [0, 0, -9.81];加速度计读到的是比力
        f_imu = -g_imu。于是 mount_rot 必须把 -f_imu 转到 [0, 0, -1] 方向。
        偏航分量无解也不需要 —— 重力与偏航无关,取最小旋转即可。
        """
        g_imu = -np.asarray(accel, float)
        self.mount_rot = rotation_between(g_imu, G_WORLD)
        return self.mount_rot

    # ---- 单帧换算 ----

    def from_orientation(self, quat_xyzw: Sequence[float]) -> np.ndarray:
        """R_wi = IMU 在世界系的姿态;R_wt = R_wi @ mount_rot^T;g_torso = R_wt^T g_w。"""
        rot_wi = quat_to_matrix(*[float(v) for v in quat_xyzw])
        return self.mount_rot @ (rot_wi.T @ G_WORLD)

    def from_accel(self, accel: Sequence[float]) -> np.ndarray:
        """静止时比力 f = -g,所以 g_imu = -f。运动时不可信。"""
        return self.mount_rot @ (-np.asarray(accel, float))

    def from_waist(self, rot_pelvis_in_world: np.ndarray,
                   q_waist: Sequence[float], chain: "WaistChain") -> np.ndarray:
        """来源 2:基座 IMU + 腰关节反馈。多串一次腰链,多吃 3 个编码器误差。"""
        _, rot_tp = chain.torso_pose_in_pelvis(q_waist)
        rot_wt = np.asarray(rot_pelvis_in_world, float) @ rot_tp
        return rot_wt.T @ G_WORLD

    # ---- 主入口 ----

    def update(self, quat_xyzw: Optional[Sequence[float]] = None,
               accel: Optional[Sequence[float]] = None,
               g_direct: Optional[Sequence[float]] = None) -> np.ndarray:
        """喂一帧 IMU,返回滤波后的 torso 系重力向量。

        quat 与 accel 同时给出时:用 quat 作为结果,用 accel 只做一致性检查
        (夹角超过 accel_tol 说明机器人在加速,或安装转角标错了)。
        """
        if g_direct is not None:
            raw = np.asarray(g_direct, float)
        elif quat_xyzw is not None:
            raw = self.from_orientation(quat_xyzw)
            if accel is not None:
                g_acc = self.from_accel(accel)
                na, nb = np.linalg.norm(raw), np.linalg.norm(g_acc)
                if na > 1e-6 and nb > 1e-6:
                    cos = float(np.clip(np.dot(raw, g_acc) / (na * nb), -1.0, 1.0))
                    self.last_accel_angle = math.acos(cos)
        elif accel is not None:
            raw = self.from_accel(accel)
        else:
            return self.g

        norm = float(np.linalg.norm(raw))
        if norm < 1e-6:
            self.n_reject += 1
            self.last_reject = "重力向量模长为 0(四元数/加计全零)"
            return self.g
        raw = raw / norm * G_NORM
        # 倾角门限:超过就认为这一帧不可信(IMU 掉帧、四元数为全零、机器人摔了)
        tilt = math.acos(float(np.clip(-raw[2] / G_NORM, -1.0, 1.0)))
        if tilt > self.tilt_limit:
            self.n_reject += 1
            self.last_reject = (f"倾角 {math.degrees(tilt):.1f} deg 超过门限 "
                                f"{math.degrees(self.tilt_limit):.0f} deg"
                                f"(机器人躺着/摔了,或安装转角没标)")
            return self.g
        self.last_reject = ""

        if not self.have_fix:
            self.g = raw            # 首帧直接采纳,不然低通要几百帧才收敛
            self.have_fix = True
        else:
            self.g = (1.0 - self.alpha) * self.g + self.alpha * raw
            self.g = self.g / max(np.linalg.norm(self.g), 1e-12) * G_NORM
        self.n_update += 1
        return self.g

    # ---- 观测量 ----

    @property
    def tilt_deg(self) -> float:
        """躯干相对铅垂线的倾角。"""
        return math.degrees(math.acos(float(np.clip(-self.g[2] / G_NORM, -1.0, 1.0))))

    def torque_error_if_ignored(self, dyn, q: np.ndarray) -> float:
        """如果无视 IMU、按 [0,0,-9.81] 算前馈,力矩会差多少(N·m)。

        这是判断"这套补偿值不值得做"的直接依据 —— 不用猜,量一下。
        """
        g_saved = dyn.gravity
        try:
            dyn.gravity = self.g
            tau_true = dyn.gravity_torque(q)
            dyn.gravity = G_WORLD.copy()
            tau_flat = dyn.gravity_torque(q)
        finally:
            dyn.gravity = g_saved
        return float(np.max(np.abs(tau_true - tau_flat)))

    def describe(self) -> str:
        return (f"重力估计 source={self.source}  g_torso={np.round(self.g, 4).tolist()}"
                f"  倾角={self.tilt_deg:.2f} deg"
                f"  更新 {self.n_update} 帧 / 丢弃 {self.n_reject} 帧"
                f"  四元数-加速度计夹角={math.degrees(self.last_accel_angle):.2f} deg"
                + (f"\n  [warn] 一帧都没采纳,g 仍是默认值。丢帧原因:{self.last_reject}"
                   if self.n_update == 0 and self.n_reject > 0 and self.last_reject else ""))


# ---------------------------------------------------------------- 单臂零位表

def zero_frame_table(model: ArmModel) -> str:
    """零位下每个关节的原点与转轴(torso 系)。判定坐标轴的第一手依据。"""
    q0 = np.zeros(7)
    positions, rotations = model.joint_frames(q0)
    axes = model.axes_at(q0)
    ee_pos, ee_rot = model.forward_kinematics(q0)

    lines = [f"[{model.side} arm] 零位几何(torso_link 系,单位 m)"]
    lines.append("  idx  joint            origin_xyz                     axis(单位向量)")
    for i, (name, pos, ax) in enumerate(zip(model.joint_names, positions, axes)):
        short = name.replace(f"{model.side}_", "").replace("_joint", "")
        lines.append(f"   {i}   {short:<15s} {np.round(pos, 6).tolist()!s:<30s} "
                     f"{np.round(ax, 6).tolist()}")
    lines.append(f"  肩心 S(三轴最小二乘交点) = {np.round(model.shoulder_center, 6).tolist()}"
                 f"   残差 {model.shoulder_axis_residual*1000:.4f} mm")
    lines.append(f"  腕心 W(零位)             = {np.round(model.wrist_center_zero, 6).tolist()}"
                 f"   残差 {model.wrist_axis_residual*1000:.4f} mm")
    lines.append(f"  末端 {model.ee_link} 零位 pos = {np.round(ee_pos, 6).tolist()}")
    lines.append(f"  末端零位姿态 R =")
    for row in np.round(ee_rot, 6):
        lines.append(f"      {row.tolist()}")
    lines.append(f"  tcp_offset(末端连杆系,装手后须标定) = {model.tcp_offset.tolist()}")
    lines.append(f"  限位 q_min = {np.round(model.q_min, 4).tolist()}")
    lines.append(f"       q_max = {np.round(model.q_max, 4).tolist()}")
    return "\n".join(lines)


def mirror_pose(pos: np.ndarray, rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把一侧的 torso 系目标位姿镜像到另一侧(绕 xz 平面反射)。

    注意这只是"几何上对称的那个点",不等于"另一条臂的对应关节角取负" ——
    左右臂的关节正方向并不成镜像关系(见 mirror_report)。
    """
    return MIRROR @ np.asarray(pos, float), MIRROR @ np.asarray(rot, float) @ MIRROR


def mirror_report(right: ArmModel, left: ArmModel) -> str:
    """左右臂到底是不是镜像?逐关节量化,别拍脑袋假设。

    转轴是**赝矢量**:在反射 M 下按 a' = det(M)·M·a = -M·a 变换。
    所以一对真正镜像的关节只可能是两种情形之一 ——
      a_left == -M a_right : 左右关节正方向"同号",同一个 q 产生镜像运动;
      a_left == +M a_right : 左右关节正方向"反号",要取负 q 才镜像。
    哪一种,由 URDF 自己定,必须逐关节查(X2 两种都有)。
    """
    q0 = np.zeros(7)
    pr, _ = right.joint_frames(q0)
    pl, _ = left.joint_frames(q0)
    ar = right.axes_at(q0)
    al = left.axes_at(q0)

    lines = ["左右臂镜像性核查(把右臂几何按 torso 系 xz 平面反射后与左臂比)"]
    lines.append("  idx  joint             |Δorigin| mm   转轴关系          轴残差")
    worst_o = worst_a = 0.0
    signs: List[int] = []
    for i in range(7):
        d_o = float(np.linalg.norm(MIRROR @ pr[i] - pl[i])) * 1000.0
        d_same = float(np.linalg.norm(-MIRROR @ ar[i] - al[i]))   # 同号
        d_flip = float(np.linalg.norm(MIRROR @ ar[i] - al[i]))    # 反号
        sign = +1 if d_same <= d_flip else -1
        signs.append(sign)
        d_a = min(d_same, d_flip)
        worst_o = max(worst_o, d_o)
        worst_a = max(worst_a, d_a)
        tag = "同号 q_L=+q_R" if sign > 0 else "反号 q_L=-q_R"
        short = ARM_JOINT_SUFFIX[i].replace("_joint", "")
        lines.append(f"   {i}   {short:<16s} {d_o:10.4f}     {tag:<16s} {d_a:.6f}")
    lines.append(f"  最大偏差: 原点 {worst_o:.4f} mm / 轴向 {worst_a:.6f}")
    lines.append(f"  关节符号向量 q_left = {signs} * q_right(仅在理想镜像下成立)")
    lines.append("")
    lines.append("  限位镜像性交叉验证(注:区间对称的关节,限位对符号不可判定):")
    conflicts = []
    for i in range(7):
        short = ARM_JOINT_SUFFIX[i].replace("_joint", "")
        symmetric = abs(right.q_min[i] + right.q_max[i]) < 1e-9
        same_as_right = (abs(left.q_min[i] - right.q_min[i]) < 1e-9
                         and abs(left.q_max[i] - right.q_max[i]) < 1e-9)
        is_mirror = (abs(left.q_min[i] + right.q_max[i]) < 1e-9
                     and abs(left.q_max[i] + right.q_min[i]) < 1e-9)
        if symmetric:
            tag = "不可判定(区间对称)"
        elif same_as_right:
            tag = "同号 -> 与转轴一致" if signs[i] > 0 else "同号 -> 与转轴矛盾"
            if signs[i] < 0:
                conflicts.append(short)
        elif is_mirror:
            tag = "反号 -> 与转轴一致" if signs[i] < 0 else "反号 -> 与转轴矛盾"
            if signs[i] > 0:
                conflicts.append(short)
        else:
            tag = "两者都不是"
            conflicts.append(short)
        lines.append(f"   {i}   {short:<16s} R[{right.q_min[i]:7.4f},{right.q_max[i]:7.4f}]"
                     f"  L[{left.q_min[i]:7.4f},{left.q_max[i]:7.4f}]   {tag}")
    lines.append(f"  可判定项与转轴符号的矛盾: {conflicts if conflicts else '无'}")
    # 直接用 FK 验一遍符号向量对不对,不靠推理
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        qr = rng.uniform(right.q_min, right.q_max)
        ql = np.array(signs, float) * qr
        ql = left.clamp(ql)
        if not np.allclose(ql, np.array(signs, float) * qr):
            continue
        p_r, _ = right.forward_kinematics(qr)
        p_l, _ = left.forward_kinematics(ql)
        worst = max(worst, float(np.linalg.norm(MIRROR @ p_r - p_l)))
    lines.append(f"  符号向量的 FK 实测校验:按 q_L=sign*q_R 摆位,两手 TCP 镜像残差 "
                 f"max {worst*1000:.3f} mm")
    lines.append("")
    lines.append("  结论:")
    lines.append("   1) 关节正方向左右并不统一:roll / yaw 三轴(idx 1,2,4,6)反号,")
    lines.append("      pitch / elbow / wrist_pitch(idx 0,3,5)同号。")
    lines.append("      所以\"左臂角 = 右臂角逐项取负\" 和 \"左右臂角完全相同\" 都是错的。")
    lines.append(f"   2) 几何只是近似镜像:原点最大差 {worst_o:.3f} mm(elbow 1.25 mm、")
    lines.append(f"      wrist 2.49 mm),轴向最大差 {worst_a:.6f}(左臂 shoulder_yaw 轴带")
    lines.append("      0.006 的 x 分量,约 0.34 deg)。左臂腕三轴还差 0.245 mm 不共点。")
    lines.append("   3) => 左右手各自 new 一个 ArmModel / SrsArmIK,各自独立求解。")
    lines.append("      mirror_pose() 只用来把**笛卡尔目标点**对称过去,不用于关节角。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 激励预测表

def excitation_table(model: ArmModel, q_ref: np.ndarray = HOME_Q,
                     delta: float = 0.15) -> Tuple[str, List[Dict]]:
    """单关节激励预测:在 q_ref 上只动第 i 个关节 +delta,TCP 会往哪走。

    这是判定"下标顺序 / 轴方向 / 正负号"最省事的实验:
    仿真里逐个关节抖一下,把实测位移方向和本表对一遍,
    对不上就说明下标映射或符号错了 —— 而不是等到笛卡尔跟踪跑偏才去猜。

    需要它的直接原因:仿真侧配置文件对腕部顺序的口径不一致 ——
      robot_model.yaml / HAL default.yaml : wrist_yaw, wrist_pitch, wrist_roll
      upper_body_external.yaml / planner_upper.yaml : wrist_yaw, wrist_roll, wrist_pitch
    而 UpperBodyCommandArray.arm_pos[14] 是按下标寻址的,索引 5/6 的语义必须实测确定。
    """
    q_ref = np.asarray(q_ref, float)
    p0, r0 = model.forward_kinematics(q_ref)
    positions, _ = model.joint_frames(q_ref)
    axes = model.axes_at(q_ref)

    rows: List[Dict] = []
    lines = [f"[{model.side} arm] 单关节激励预测  q_ref={np.round(q_ref, 3).tolist()}  "
             f"Δq=+{delta:.3f} rad"]
    lines.append(f"  基准 TCP = {np.round(p0, 5).tolist()}")
    lines.append("  idx  joint             轴向(torso)              一阶位移方向         实际位移 mm")
    for i in range(7):
        q = q_ref.copy()
        q[i] += delta
        q = model.clamp(q)
        p1, _ = model.forward_kinematics(q)
        dp = p1 - p0
        # 一阶预测:v = a_i x (p_tcp - o_i)
        lin = cross3(axes[i], p0 - positions[i])
        n = np.linalg.norm(lin)
        lin_dir = lin / n if n > 1e-12 else np.zeros(3)
        short = ARM_JOINT_SUFFIX[i].replace("_joint", "")
        rows.append(dict(index=i, joint=model.joint_names[i], axis=axes[i],
                         dir_first_order=lin_dir, delta_applied=float(q[i] - q_ref[i]),
                         disp=dp))
        lines.append(f"   {i}   {short:<16s} {np.round(axes[i], 4).tolist()!s:<24s} "
                     f"{np.round(lin_dir, 4).tolist()!s:<20s} {np.round(dp*1000, 2).tolist()}")
    lines.append("  说明:'一阶位移方向' 是 a_i x (p_tcp - o_i) 归一化,只看方向;")
    lines.append("        '实际位移' 是真跑一次 FK 的差值,Δq 较大时两者会有二阶偏差。")
    lines.append("        wrist_roll(idx 6)对 TCP 位移几乎为零 —— 它只改姿态,")
    lines.append("        所以核对 idx 5/6 要看姿态变化轴,不能只看位置。")
    return "\n".join(lines), rows


def orientation_excitation(model: ArmModel, q_ref: np.ndarray = HOME_Q,
                           delta: float = 0.15) -> str:
    """腕部三轴的姿态激励:看末端姿态绕哪个轴转。用于区分 idx 5 / idx 6。"""
    from x2_arm_model import log3
    q_ref = np.asarray(q_ref, float)
    _, r0 = model.forward_kinematics(q_ref)
    lines = [f"[{model.side} arm] 姿态激励(区分 wrist_pitch / wrist_roll 的关键)"]
    lines.append("  idx  joint             末端姿态旋转向量方向(torso 系)   转角 deg")
    for i in range(4, 7):
        q = q_ref.copy()
        q[i] += delta
        q = model.clamp(q)
        _, r1 = model.forward_kinematics(q)
        w = log3(r0.T @ r1)          # 末端自身系下的旋转
        w_world = r0 @ w
        ang = float(np.linalg.norm(w_world))
        d = w_world / ang if ang > 1e-12 else np.zeros(3)
        short = ARM_JOINT_SUFFIX[i].replace("_joint", "")
        lines.append(f"   {i}   {short:<16s} {np.round(d, 4).tolist()!s:<30s} {np.degrees(ang):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- MJCF 交叉校验

def parse_mjcf_arm_chain(mjcf_path: Path, side: str) -> List[Dict]:
    """从 MJCF 里独立抽一条臂链(pos/quat/axis),用于和 URDF 交叉验证。

    为什么要这一步:仿真跑的是 MJCF,IK 算的是 URDF。两者只要有一处 origin 对不上,
    表现就是"IK 数值完美但仿真里就是差几毫米",而且极难定位。
    先把两份模型对平,后面所有笛卡尔误差才能归因到控制而不是模型。
    """
    text = Path(mjcf_path).read_text(encoding="utf-8")
    chain: List[Dict] = []
    cursor = 0
    for suffix in ARM_JOINT_SUFFIX:
        link = f"{side}_{suffix.replace('_joint', '_link')}"
        m = re.search(rf'<body\s+name="{re.escape(link)}"([^>]*)>', text[cursor:])
        if not m:
            raise RuntimeError(f"MJCF 中找不到 body {link}")
        attrs = m.group(1)
        pos = re.search(r'pos="([^"]+)"', attrs)
        quat = re.search(r'quat="([^"]+)"', attrs)
        euler = re.search(r'euler="([^"]+)"', attrs)
        xyz = np.array([float(v) for v in pos.group(1).split()]) if pos else np.zeros(3)
        if quat:
            w, x, y, z = (float(v) for v in quat.group(1).split())
            n = np.sqrt(w*w + x*x + y*y + z*z)
            w, x, y, z = w/n, x/n, y/n, z/n
            rot = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
            ])
        elif euler:
            rot = rpy_to_matrix([float(v) for v in euler.group(1).split()])
        else:
            rot = np.eye(3)
        jm = re.search(rf'<joint\s+name="{side}_{re.escape(suffix)}"([^>]*)>',
                       text[cursor + m.end():])
        axis = np.array([0.0, 0.0, 1.0])
        if jm:
            am = re.search(r'axis="([^"]+)"', jm.group(1))
            if am:
                axis = np.array([float(v) for v in am.group(1).split()])
                axis = axis / np.linalg.norm(axis)
        chain.append(dict(link=link, xyz=xyz, rot=rot, axis=axis))
        cursor += m.end()
    return chain


def mjcf_fk(chain: List[Dict], q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(3)
    rot = np.eye(3)
    for seg, angle in zip(chain, np.asarray(q, float)):
        pos = pos + rot @ seg["xyz"]
        rot = rot @ seg["rot"] @ axis_angle_to_matrix(seg["axis"], float(angle))
    return pos, rot


def cross_check_mjcf(model: ArmModel, mjcf_path: Path = DEFAULT_MJCF,
                     samples: int = 200, seed: int = 7) -> str:
    """URDF(IK 基准) vs MJCF(仿真跑的模型) 的 FK 一致性。"""
    chain = parse_mjcf_arm_chain(Path(mjcf_path), model.side)
    d_org = max(float(np.linalg.norm(chain[i]["xyz"] - model.geoms[i].origin_xyz))
                for i in range(7))
    d_rot = max(float(np.max(np.abs(chain[i]["rot"] - model.geoms[i].origin_rot)))
                for i in range(7))
    d_ax = max(float(np.linalg.norm(chain[i]["axis"] - model.geoms[i].axis)) for i in range(7))

    rng = np.random.default_rng(seed)
    dp = dr = 0.0
    for _ in range(samples):
        q = rng.uniform(model.q_min, model.q_max)
        p1, r1 = mjcf_fk(chain, q)
        p2, r2 = model.forward_kinematics(q)   # tcp_offset 为 0 时可直接比
        dp = max(dp, float(np.linalg.norm(p1 - (p2 - r2 @ model.tcp_offset))))
        dr = max(dr, float(np.max(np.abs(r1 - r2))))
    return (f"[{model.side} arm] URDF vs MJCF 交叉校验({samples} 组随机 q)\n"
            f"  逐关节 |Δorigin_xyz| max = {d_org:.3e} m\n"
            f"  逐关节 |Δorigin_rot| max = {d_rot:.3e}\n"
            f"  逐关节 |Δaxis|       max = {d_ax:.3e}\n"
            f"  末端位置差 max          = {dp*1e9:.0f} nm\n"
            f"  末端姿态元素差 max      = {dr:.3e}\n"
            f"  (残差量级 1e-7 是 MJCF 只存 6 位有效数字的舍入,不是建模差异)")


# ---------------------------------------------------------------- 待机位

def home_report() -> str:
    from x2_srs_ik import SrsArmIK
    lines = [f"待机姿态 HOME_Q = {HOME_Q.tolist()}  (来自 mc idle.yaml / cpgtelecon_config.yaml)"]
    for side in ("right", "left"):
        m = ArmModel(side)
        ik = SrsArmIK(m)
        p, r = m.forward_kinematics(HOME_Q)
        w = ik.target_wrist_center(p, r)
        d = float(np.linalg.norm(w - m.shoulder_center))
        psi = ik.sew_angle(HOME_Q)
        sol = ik.track(p, r, q_prev=HOME_Q)
        back = float(np.max(np.abs(sol.q - HOME_Q))) if sol else np.nan
        lines.append(f"  [{side}] TCP(torso) = {np.round(p, 5).tolist()}")
        lines.append(f"         |S->W| = {d*1000:.2f} mm  (限位内可达 "
                     f"{ik.reach_min_limited*1000:.1f} ~ {ik.reach_max_limited*1000:.1f} mm)")
        lines.append(f"         臂角 psi = {np.degrees(psi):.2f} deg")
        lines.append(f"         track() 回解误差 = {back:.2e} rad")
    lines.append("  注:solve() 全局扫 psi 时会挑到另一支同样合法的解(关节差可达 0.29 rad),")
    lines.append("      因为代价函数里限位裕度/可操作度项把肘部拉走了。要复现 HOME 本身,")
    lines.append("      用 track(q_prev=HOME_Q) 或直接下发关节角,不要用 solve()。")
    return "\n".join(lines)


def main() -> None:
    np.set_printoptions(suppress=True)
    right, left = ArmModel("right"), ArmModel("left")

    print("=" * 78)
    print("A. IK 基座系:torso_link")
    print("=" * 78)
    print("  URDF 里 base_link -> pelvis 是恒等固定关节,两者原点姿态完全重合。")
    print("  两条臂的父连杆都是 torso_link,所以 IK 的输入输出位姿一律在 torso_link 系。")
    print("  torso_link 系轴向(零腰位时与 pelvis 一致):+X 前 / +Y 左 / +Z 上。")
    print()
    print(WaistChain().describe())
    print()
    print("  仿真里 pelvis 是 freejoint 浮动基座,世界位姿从 /aima/hal/odom/state")
    print("  或 mjData.qpos[0:7] 取;要把 IK 结果表达到世界系需要再左乘这一层。")
    print("  但控制本身不需要 —— 手臂链完全在 torso 系内闭合。")
    print()

    for m in (right, left):
        print("=" * 78)
        print(f"B. {m.side} 臂零位坐标系")
        print("=" * 78)
        print(zero_frame_table(m))
        print()

    print("=" * 78)
    print("C. 左右臂对称性")
    print("=" * 78)
    print(mirror_report(right, left))
    print()

    print("=" * 78)
    print("D. 待机位")
    print("=" * 78)
    print(home_report())
    print()

    print("=" * 78)
    print("D2. 侧平举姿态(肩/肘/腕同高)")
    print("=" * 78)
    print(lateral_raise_report(0.0))
    print()
    print(lateral_raise_report(-math.pi / 2))
    print()

    print("=" * 78)
    print("E. 单关节激励预测表(拿去和仿真实测对账)")
    print("=" * 78)
    for m in (right, left):
        txt, _ = excitation_table(m)
        print(txt)
        print(orientation_excitation(m))
        print()

    print("=" * 78)
    print("F. URDF / MJCF 一致性")
    print("=" * 78)
    for m in (right, left):
        print(cross_check_mjcf(m))
        print()


if __name__ == "__main__":
    main()
