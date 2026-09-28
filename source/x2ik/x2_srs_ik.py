#!/usr/bin/env python3
"""X2 单臂七轴逆运动学：SRS/SEW 解析搜索与六维位姿精修。

模型在 torso_link 中描述肩、肘、腕几何，给定 SEW 臂角后枚举肘部、
肩部和腕部解析分支，按限位及代价筛选，左腕轴并非严格共点，解析候选
必须用完整 URDF 和所选 TCP 的 FK 复核。

solve 保持已满足目标的种子，优先局部求解，并对漏解做有限次恢复，
track 使用有限迭代与 SEW 邻域搜索，检查分支、残差和关节步长，
运动入口禁用全局恢复，算法只处理运动学，不提供碰撞或动力学规划。

参考
----
Shimizu et al., "Analytical Inverse Kinematics for 7 DOF Redundant Manipulators
with Joint Limits", IEEE T-RO 2008        —— arm-angle 参数化
Elias & Wen, "IK-Geo: Unified robot inverse kinematics using subproblem
decomposition", Mech. Mach. Theory 2025   —— Paden-Kahan 子问题
Elias & Wen, "Redundancy parameterization and inverse kinematics of 7-DOF
revolute manipulators" (stereo-SEW)       —— 见下方 SEW 奇异说明
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .x2_arm_model import (ArmModel, axis_angle_to_matrix, cross3, log3,
                          log3_batch)
from .x2_srs_batch import BatchGrid, BatchSrsCore, bcross

_EPS = 1e-12


def angle_delta(q_new: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """逐元素最短角差,用于连续性代价而非关节限位判断。"""
    return np.arctan2(np.sin(np.asarray(q_new) - np.asarray(q_ref)),
                      np.cos(np.asarray(q_new) - np.asarray(q_ref)))


def branch_tuple(sol: "IKSolution") -> Tuple[int, int, int]:
    """返回解析 IK 解支标识。"""
    return (sol.elbow_branch, sol.shoulder_branch, sol.wrist_branch)


# --------------------------------------------------------------------------
# Paden-Kahan 子问题
# --------------------------------------------------------------------------

def subproblem1(axis: np.ndarray, vec_from: np.ndarray, vec_to: np.ndarray) -> float:
    """求 theta 使 Rot(axis, theta) @ vec_from = vec_to。

    要求两向量在 axis 上的投影相等(否则无解,这里取最小二乘意义的角度)。
    """
    a_perp = vec_from - np.dot(axis, vec_from) * axis
    b_perp = vec_to - np.dot(axis, vec_to) * axis
    return float(np.arctan2(np.dot(axis, cross3(a_perp, b_perp)),
                            float(np.dot(a_perp, b_perp))))


def subproblem2(axis1: np.ndarray, axis2: np.ndarray,
                vec_from: np.ndarray, vec_to: np.ndarray,
                tol: float = 1e-7) -> List[Tuple[float, float]]:
    """求 (t1, t2) 使 Rot(axis1,t1) @ Rot(axis2,t2) @ vec_from = vec_to。

    两轴相交(此处都过肩心/腕心)时的经典双解。轴不必正交 —— 这正是我们要
    精确吸收 URDF 里那些小 rpy 偏置的原因。
    """
    dot12 = float(np.dot(axis1, axis2))
    denom = 1.0 - dot12 * dot12
    if abs(denom) < 1e-10:          # 两轴平行,退化
        return []
    d1t = float(np.dot(axis1, vec_to))
    d2n = float(np.dot(axis2, vec_from))
    alpha = (d1t - dot12 * d2n) / denom
    beta = (d2n - dot12 * d1t) / denom
    cross12 = cross3(axis1, axis2)
    gamma_sq = (float(np.dot(vec_from, vec_from))
                - alpha * alpha - beta * beta
                - 2.0 * alpha * beta * dot12) / denom
    if gamma_sq < -tol:
        return []
    gamma = float(np.sqrt(max(gamma_sq, 0.0)))

    out: List[Tuple[float, float]] = []
    for sign in ((1.0, -1.0) if gamma > tol else (1.0,)):
        mid = alpha * axis1 + beta * axis2 + sign * gamma * cross12
        t2 = subproblem1(axis2, vec_from, mid)
        t1 = subproblem1(axis1, mid, vec_to)
        out.append((t1, t2))
    return out


def decompose_rotation(rot: np.ndarray,
                       axis1: np.ndarray, axis2: np.ndarray, axis3: np.ndarray
                       ) -> List[Tuple[float, float, float]]:
    """把已知旋转分解为 Rot(a1,t1) Rot(a2,t2) Rot(a3,t3)。

    先用 a3 是 Rot(a3,t3) 不动点这一性质消掉 t3,得到子问题 2;再回代求 t3。
    """
    solutions: List[Tuple[float, float, float]] = []
    target = rot @ axis3
    for t1, t2 in subproblem2(axis1, axis2, axis3, target):
        pre = axis_angle_to_matrix(axis1, t1) @ axis_angle_to_matrix(axis2, t2)
        residual = pre.T @ rot                      # 应等于 Rot(axis3, t3)
        # 取任一与 axis3 不平行的向量来测角
        probe = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(probe, axis3))) > 0.9:
            probe = np.array([0.0, 1.0, 0.0])
        probe = probe - np.dot(probe, axis3) * axis3
        probe /= np.linalg.norm(probe)
        t3 = subproblem1(axis3, probe, residual @ probe)
        solutions.append((t1, t2, t3))
    return solutions


def _frame_from_two_vectors(primary: np.ndarray, secondary: np.ndarray) -> Optional[np.ndarray]:
    """由主方向 + 辅方向构造右手正交基,列向量为基向量。"""
    e1 = primary / np.linalg.norm(primary)
    e2 = secondary - np.dot(secondary, e1) * e1
    norm2 = np.linalg.norm(e2)
    if norm2 < 1e-9:                                # 两向量共线,无法定基
        return None
    e2 /= norm2
    return np.column_stack([e1, e2, cross3(e1, e2)])


# --------------------------------------------------------------------------
# 解
# --------------------------------------------------------------------------

@dataclass
class IKSolution:
    q: np.ndarray
    psi: float              # SEW 角 (rad)
    elbow_branch: int       # 0/1  肘部两支
    shoulder_branch: int    # 0/1  肩部 Euler 双解
    wrist_branch: int       # 0/1  腕部 Euler 双解
    in_limits: bool
    pos_error: float = 0.0
    rot_error: float = 0.0
    source: str = "analytic"


@dataclass
class RelaxedSolution:
    """位置优先求解的结果 —— 见 SrsArmIK.solve_hold_rotation。

    要点是 pos / rot 都是**实际采用**的值,未必等于请求值:pos 可能被
    project_to_workspace 拉回过(clipped),rot 可能被降级过(axis 非 None)。
    上层报误差时必须拿这两个当基准,否则会把"我们主动改的量"算成跟踪误差。
    """
    sol: IKSolution
    pos: np.ndarray            # 实际使用的位置
    rot: np.ndarray            # 实际使用的姿态
    deviation: float           # rot 与首选姿态的等效轴角 (rad);未降级时为 0
    axis: Optional[int]        # 降级绕的轴 0/1/2 = X/Y/Z;None = 没降级
    frame: str                 # "torso"(左乘) | "tcp"(右乘);未降级时为 ""
    clipped: bool              # 位置是否被拉回过工作空间
    tried: int                 # 试过多少个候选姿态,用来解释这一下为什么慢


#: 降级方向的候选轴。torso 系里就是 X/Y/Z,TCP 系里是 TCP 自身的三轴。
_RELAX_AXES = np.eye(3)


# --------------------------------------------------------------------------
# 求解器
# --------------------------------------------------------------------------

class SrsArmIK:
    """X2 单臂解析 IK。

    sew_reference: SEW 角的参考方向(torso 系)。默认 -Z(肘朝下),
    这是人形臂最自然的参考。注意传统 arm-angle 在 S->W 与参考方向共线时
    存在算法奇异 —— 本实现把 psi 只当作"圆上的参数",求解本身不经过该奇异,
    但 psi 的数值在奇异锥附近会失去意义,solve() 因此按关节空间连续性而非
    psi 连续性挑解。要彻底消除,可换用 stereo-SEW 参数化(见模块 docstring)。
    """

    POSITION_TOL = 1e-5               # m，模型内位置残差
    ROTATION_TOL = math.radians(0.001)
    RECOVERY_RESTARTS = 16
    TRACK_LOCAL_ITERATIONS = 12
    TRACK_MAX_STEP = 0.05            # rad，单次输出拒绝阈值

    def __init__(self, model: ArmModel,
                 sew_reference: Optional[Sequence[float]] = None):
        self.model = model
        ref = np.array([0.0, 0.0, -1.0]) if sew_reference is None else np.asarray(sew_reference, float)
        self.sew_reference = ref / np.linalg.norm(ref)
        # 备用参考:主参考与 S->W 共线时启用
        self.sew_reference_alt = np.array([1.0, 0.0, 0.0])

        q0 = np.zeros(7)
        positions, rotations = model.joint_frames(q0)
        self.axes0 = model.axes_at(q0)
        self.shoulder = model.shoulder_center.copy()
        self.wrist0 = model.wrist_center_zero.copy()
        self.rot_ee0 = rotations[6].copy()
        # 腕心相对末端连杆的常量偏置(X2 上为 0,但公式里必须带着)
        self.wrist_local = model.wrist_center_local.copy()

        a4 = self.axes0[3]
        elbow_raw = positions[3]
        # 把肘点沿肘轴投影,使 u1 严格垂直于肘轴 —— 后面的余弦定理依赖这一点
        self.elbow0 = elbow_raw - float(np.dot(elbow_raw - self.shoulder, a4)) * a4
        self.u1 = self.elbow0 - self.shoulder
        self.u2 = self.wrist0 - self.elbow0
        self.len1 = float(np.linalg.norm(self.u1))
        u2_par = float(np.dot(self.u2, a4)) * a4
        u2_perp = self.u2 - u2_par
        self.len2_perp = float(np.linalg.norm(u2_perp))
        self.u2_norm_sq = float(np.dot(self.u2, self.u2))
        # u1 到 u2 垂直分量的有符号夹角(绕肘轴),即零位时的肘部常量偏置
        self.elbow_offset = float(np.arctan2(
            float(np.dot(a4, cross3(self.u1, u2_perp))),
            float(np.dot(self.u1, u2_perp)),
        ))

        self.reach_max = float(np.sqrt(self.len1 ** 2 + self.u2_norm_sq
                                       + 2.0 * self.len1 * self.len2_perp))
        self.reach_min = float(np.sqrt(max(self.len1 ** 2 + self.u2_norm_sq
                                           - 2.0 * self.len1 * self.len2_perp, 0.0)))

        # "肩腕距离 = f(q4)" 的几何容差。腕三轴不严格共点时该前提有误差,
        # 实测上界约为球腕残差的两倍,再留一点余量。X2 右臂 ~0,左臂 ~0.47 mm。
        self.reach_tol = max(3.0 * self.model.wrist_axis_residual, 1e-6)

        # 上面的 reach_max/min 是纯几何极限(肘角不受限时)。实际肘角受 URDF 限位
        # 约束,可达范围更窄 —— 官方 v1.3.0 把肘下限从 -2.556 收紧到 -2.3556,
        # 直接抬高了最小伸展距离,这是版本差异里唯一影响工作空间的一项。
        self.reach_max_limited, self.reach_min_limited = self._reach_under_limits()

        sign = -1.0 if model.side == "right" else 1.0
        self.q_ref = np.array([0.0, sign * 0.30, 0.0, -0.80, 0.0, 0.0, 0.0])

        # 批量后端。必须放在最后 —— 它要读上面缓存的 axes0 / u1 / rot_ee0 等常量。
        self.batch = BatchSrsCore(self)
        # 固定种子与目标无关，仅用于单目标求解失败后的有限次恢复。
        rng = np.random.default_rng(492781)
        self._recovery_seeds = np.vstack((
            0.5 * (model.q_min + model.q_max),
            rng.uniform(model.q_min, model.q_max, (self.RECOVERY_RESTARTS, 7)),
        ))
        self._max_reach = (sum(np.linalg.norm(g.origin_xyz) for g in model.geoms)
                           + np.linalg.norm(model.tcp_offset))

    def _reach_under_limits(self) -> Tuple[float, float]:
        """在肘关节限位范围内扫一遍 |S->W|,给出真实可达的伸展区间。"""
        q4 = np.linspace(self.model.q_min[3], self.model.q_max[3], 2001)
        dist_sq = (self.len1 ** 2 + self.u2_norm_sq
                   + 2.0 * self.len1 * self.len2_perp * np.cos(self.elbow_offset + q4))
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return float(dist.max()), float(dist.min())

    # ---------- 几何辅助 ----------

    def target_wrist_center(self, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """由 TCP 目标位姿反推腕心:先去掉 TCP 外参回到末端连杆原点,再加腕心偏置。"""
        rot_wrist = np.asarray(rot, float) @ self.model.tcp_rotation.T
        ee_origin = np.asarray(pos, float) - rot_wrist @ self.model.tcp_offset
        return ee_origin + rot_wrist @ self.wrist_local

    def reach_of(self, pos: np.ndarray, rot: np.ndarray) -> float:
        """目标位姿对应的肩-腕距离。用于在下发前判可达性。"""
        return float(np.linalg.norm(self.target_wrist_center(pos, rot) - self.shoulder))

    def project_to_workspace(self, pos: np.ndarray, rot: np.ndarray,
                             margin: float = 2e-3) -> Tuple[np.ndarray, bool]:
        """把不可达目标沿肩->腕方向拉回球壳内,返回 (新位置, 是否被裁剪)。

        margin 是留给数值精度的余量:正好贴在 reach_max 上时肘圆半径退化为 0,
        解析解会掉进 _frame_from_two_vectors 的共线分支。默认留 2 mm。

        控制环里应当"裁剪并上报",而不是直接失败 —— 失败会让上层丢一帧指令,
        在 500 Hz 关节接口上表现为可感知的顿挫。
        """
        pos = np.asarray(pos, float)
        rot = np.asarray(rot, float)
        wrist = self.target_wrist_center(pos, rot)
        vec = wrist - self.shoulder
        dist = float(np.linalg.norm(vec))
        hi, lo = self.reach_max_limited - margin, self.reach_min_limited + margin
        if lo <= dist <= hi:
            return pos, False
        if dist < 1e-9:                      # 目标就在肩心,方向无定义
            return pos, False
        target = hi if dist > hi else lo
        wrist_new = self.shoulder + vec * (target / dist)
        rot_wrist = rot @ self.model.tcp_rotation.T
        return wrist_new + rot_wrist @ (self.model.tcp_offset - self.wrist_local), True

    def _sew_basis(self, v_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """给定 S->W 单位向量,构造 SEW 角的零相位基 (n̂, b̂)。"""
        ref = self.sew_reference
        n_vec = ref - float(np.dot(ref, v_hat)) * v_hat
        if np.linalg.norm(n_vec) < 1e-6:            # 落在奇异锥内,换备用参考
            ref = self.sew_reference_alt
            n_vec = ref - float(np.dot(ref, v_hat)) * v_hat
            if np.linalg.norm(n_vec) < 1e-6:
                ref = np.array([0.0, 1.0, 0.0])
                n_vec = ref - float(np.dot(ref, v_hat)) * v_hat
        n_hat = n_vec / np.linalg.norm(n_vec)
        return n_hat, cross3(v_hat, n_hat)

    def sew_angle(self, q: np.ndarray) -> float:
        """给定关节角,反算其 SEW 角。用于诊断和轨迹上的 psi 规划。"""
        positions, rotations = self.model.joint_frames(q)
        return self._sew_from_frames(positions, rotations)

    def _sew_from_frames(self, positions, rotations) -> float:
        """已有 joint_frames 结果时直接算 SEW 角。

        拆出来是为了让 polish 的残差把"FK + SEW 角"合并成一次串联运算 ——
        原来 _residual 里 forward_kinematics 和 sew_angle 各串一遍,白算一半。
        """
        wrist = positions[6] + rotations[6] @ self.wrist_local
        elbow = positions[3]
        v_vec = wrist - self.shoulder
        norm = np.linalg.norm(v_vec)
        if norm < 1e-9:
            return 0.0
        v_hat = v_vec / norm
        n_hat, b_hat = self._sew_basis(v_hat)
        e_vec = elbow - self.shoulder
        e_perp = e_vec - float(np.dot(e_vec, v_hat)) * v_hat
        return float(np.arctan2(float(np.dot(e_perp, b_hat)),
                                float(np.dot(e_perp, n_hat))))

    def sew_angle_batch(self, q_arr: np.ndarray) -> np.ndarray:
        """sew_angle 的批量版,返回 (...)。polish 的有限差分靠它一次算完 7 个扰动。"""
        positions, rotations = self.model.joint_frames_batch(q_arr)
        return self._sew_from_frames_batch(positions, rotations)

    def _sew_from_frames_batch(self, positions: np.ndarray,
                               rotations: np.ndarray) -> np.ndarray:
        wrist = positions[..., 6, :] + rotations[..., 6, :, :] @ self.wrist_local
        elbow = positions[..., 3, :]
        v_vec = wrist - self.shoulder
        norm = np.linalg.norm(v_vec, axis=-1, keepdims=True)
        v_hat = v_vec / np.where(norm < 1e-9, 1.0, norm)
        n_hat, b_hat = self.batch._sew_basis(v_hat)
        e_vec = elbow - self.shoulder
        e_perp = e_vec - np.sum(e_vec * v_hat, axis=-1, keepdims=True) * v_hat
        out = np.arctan2(np.sum(e_perp * b_hat, axis=-1),
                         np.sum(e_perp * n_hat, axis=-1))
        return np.where(norm[..., 0] < 1e-9, 0.0, out)

    def elbow_angles(self, distance: float) -> List[float]:
        """由肩-腕距离解肘角。两支对应肘部弯曲方向。

        `distance` 先夹进 [reach_min - tol, reach_max + tol] 再反解。tol 来自
        球腕残差:X2 左臂腕三轴并不严格共点(残差 0.245 mm),所以"肩腕距离只是
        q4 的函数"这个前提本身带 ~0.47 mm 的几何误差。不给容差的话,由合法 q
        正解出来的位姿会被自己判成不可达;给了容差,先取"完全伸直"这一支,
        剩下的亚毫米残差交给 polish 的 LM 迭代收干。
        """
        if self.len1 < _EPS or self.len2_perp < _EPS:
            return []
        tol = self.reach_tol
        distance = float(np.clip(distance, self.reach_min - tol, self.reach_max + tol))
        cos_val = ((distance ** 2 - self.len1 ** 2 - self.u2_norm_sq)
                   / (2.0 * self.len1 * self.len2_perp))
        if abs(cos_val) > 1.0 + 1e-6:
            return []
        phi = float(np.arccos(np.clip(cos_val, -1.0, 1.0)))
        angles = [phi - self.elbow_offset]
        if phi > 1e-9:
            angles.append(-phi - self.elbow_offset)
        return [float(np.arctan2(np.sin(a), np.cos(a))) for a in angles]

    # ---------- 主求解 ----------

    def solve_at_psi(self, pos: np.ndarray, rot: np.ndarray, psi: float,
                     limits_only: bool = True,
                     fill_error: bool = True) -> List[IKSolution]:
        """给定目标位姿与 SEW 角,枚举全部解支(至多 8 组)。

        fill_error=False 时不为每个解支跑一遍 FK 算误差。扫 psi 时会产生上百个
        候选,而误差只对最终选中的那一个有意义 —— 这里省下的 FK 是热路径上
        最大的一笔无用开销。
        """
        pos = np.asarray(pos, float)
        rot = np.asarray(rot, float)
        wrist_t = self.target_wrist_center(pos, rot)
        v_vec = wrist_t - self.shoulder
        distance = float(np.linalg.norm(v_vec))
        # 判据必须带 reach_tol:否则由合法 q 正解出的边界位姿会被自己判成不可达
        # (左臂腕三轴不严格共点,"肩腕距离=f(q4)" 本身有 ~0.47 mm 误差)。
        if (distance < 1e-6
                or distance > self.reach_max + self.reach_tol
                or distance < self.reach_min - self.reach_tol):
            return []   # 纯几何判据;肘限位由后面的 within_limits 统一过滤
        v_hat = v_vec / distance
        n_hat, b_hat = self._sew_basis(v_hat)
        a4 = self.axes0[3]
        rot_ee_target = rot @ self.model.tcp_rotation.T

        out: List[IKSolution] = []
        for elbow_branch, q4 in enumerate(self.elbow_angles(distance)):
            rot_a4 = axis_angle_to_matrix(a4, q4)
            w_vec = self.u1 + rot_a4 @ self.u2
            w_hat = w_vec / np.linalg.norm(w_vec)
            height = float(np.dot(self.u1, w_hat))   # (E-S) 在 S->W 方向上的投影
            radius_sq = self.len1 ** 2 - height ** 2
            if radius_sq < -1e-9:
                continue
            radius = float(np.sqrt(max(radius_sq, 0.0)))
            elbow_pt = (self.shoulder + height * v_hat
                        + radius * (np.cos(psi) * n_hat + np.sin(psi) * b_hat))

            rot_s = self._shoulder_rotation(elbow_pt, w_hat, v_hat)
            if rot_s is None:
                continue
            rot_w = rot_a4.T @ rot_s.T @ rot_ee_target @ self.rot_ee0.T

            shoulder_sols = decompose_rotation(rot_s, self.axes0[0], self.axes0[1], self.axes0[2])
            wrist_sols = decompose_rotation(rot_w, self.axes0[4], self.axes0[5], self.axes0[6])
            for s_idx, (q1, q2, q3) in enumerate(shoulder_sols):
                for w_idx, (q5, q6, q7) in enumerate(wrist_sols):
                    q = np.array([q1, q2, q3, q4, q5, q6, q7])
                    ok = self.model.within_limits(q, tol=1e-9)
                    if limits_only and not ok:
                        continue
                    sol = IKSolution(q=q, psi=float(psi), elbow_branch=elbow_branch,
                                     shoulder_branch=s_idx, wrist_branch=w_idx,
                                     in_limits=ok)
                    if fill_error:
                        self._fill_error(sol, pos, rot)
                    out.append(sol)
        return out

    def _shoulder_rotation(self, elbow_pt: np.ndarray,
                           w_hat: np.ndarray, v_hat: np.ndarray) -> Optional[np.ndarray]:
        """由 (u1 -> E-S) 与 (ŵ -> v̂) 两组对应唯一确定肩部旋转。"""
        src = _frame_from_two_vectors(self.u1, w_hat)
        dst = _frame_from_two_vectors(elbow_pt - self.shoulder, v_hat)
        if src is None or dst is None:               # 手臂完全伸直/折叠,肘圆退化为点
            return None
        return dst @ src.T

    def _fill_error(self, sol: IKSolution, pos: np.ndarray, rot: np.ndarray) -> None:
        fk_pos, fk_rot = self.model.forward_kinematics(sol.q)
        sol.pos_error = float(np.linalg.norm(fk_pos - pos))
        sol.rot_error = float(np.linalg.norm(log3(rot @ fk_rot.T)))

    # ---------- 精修 ----------

    def _residual(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray,
                  target_psi: float) -> np.ndarray:
        """精修用的 7 维残差:[位置(3); 姿态(3); SEW 角(1)]。

        FK 与 SEW 角共用同一次串联(_sew_from_frames),不再各串一遍。
        """
        positions, rotations = self.model.joint_frames(q)
        rot_wrist = rotations[-1]
        fk_rot = rot_wrist @ self.model.tcp_rotation
        fk_pos = positions[-1] + rot_wrist @ self.model.tcp_offset
        err = np.empty(7)
        err[:3] = pos - fk_pos
        err[3:6] = log3(rot @ fk_rot.T)
        delta = target_psi - self._sew_from_frames(positions, rotations)
        err[6] = float(np.arctan2(np.sin(delta), np.cos(delta)))
        return err

    def _residual_batch(self, q_arr: np.ndarray, pos: np.ndarray, rot: np.ndarray,
                        target_psi: float) -> np.ndarray:
        """_residual 的批量版,(N,7) -> (N,7)。LM 的阻尼试探靠它一次算完。"""
        positions, rotations = self.model.joint_frames_batch(q_arr)
        rot_wrist = rotations[..., -1, :, :]
        fk_rot = rot_wrist @ self.model.tcp_rotation
        fk_pos = positions[..., -1, :] + rot_wrist @ self.model.tcp_offset
        err = np.empty(q_arr.shape[:-1] + (7,))
        err[..., :3] = pos - fk_pos
        err[..., 3:6] = log3_batch(rot @ np.swapaxes(fk_rot, -1, -2))
        delta = target_psi - self._sew_from_frames_batch(positions, rotations)
        err[..., 6] = np.arctan2(np.sin(delta), np.cos(delta))
        return err

    def _augmented_jacobian(self, q: np.ndarray) -> np.ndarray:
        """[几何雅可比(6x7); d psi/dq(1x7)] 拼成的 7x7 方阵。

        psi 那一行用有限差分 —— 解析式要对肘圆基向量再求一次导,
        表达式长且容易写错,而这里每帧只需 7 次 FK,实测占 track 耗时不到 15%。
        """
        step = 1e-6
        # 原点 + 7 个单关节扰动打成一批,一次串联 FK 全部算完。
        # 改前是 9 次串联(1 次 jacobian + 8 次 sew_angle),0.354 ms;
        # 改后 2 次批量调用,约 0.05 ms —— polish 的头号开销就在这里。
        stack = np.repeat(q[None, :], 8, axis=0)
        stack[1:] += step * np.eye(7)
        positions, rotations = self.model.joint_frames_batch(stack)
        jac = np.empty((7, 7))
        jac[:6, :] = self.model.jacobian_from_frames(positions[0:1], rotations[0:1])[0]
        psi_all = self._sew_from_frames_batch(positions, rotations)
        delta = psi_all[1:] - psi_all[0]
        jac[6, :] = np.arctan2(np.sin(delta), np.cos(delta)) / step
        return jac

    def polish(self, sol: IKSolution, pos: np.ndarray, rot: np.ndarray,
               psi: Optional[float] = None, iterations: int = 12,
               tol: float = 1e-12) -> IKSolution:
        """Levenberg-Marquardt 精修,吸收 URDF 的球腕建模残差(左臂 0.245 mm)。

        残差是 [位姿误差(6); SEW 角误差(1)] 的 7 维方阵系统,所以精修不会
        改变 psi —— 冗余自由度仍然由调用方掌控。

        必须带阻尼:近奇异位姿下 J 病态,无阻尼 lstsq 会给出巨大步长而震荡,
        实测会把误差长尾从 1e-13 mm 抬到 0.6 mm。这里按经典策略自适应调 lambda
        (接受则减小、拒绝则放大),并且只在残差真的下降时才接受该步,
        所以最坏情况也不会比输入更差。
        """
        target_psi = sol.psi if psi is None else float(psi)
        q = sol.q.copy()
        err = self._residual(q, pos, rot, target_psi)
        cost = float(np.linalg.norm(err))
        lam = 1e-8
        for _ in range(iterations):
            if cost < tol:
                break
            jac = self._augmented_jacobian(q)
            jtj = jac.T @ jac
            jte = jac.T @ err
            # 阻尼阶梯是 lam, 5lam, ..., 5^5 lam,升序试、遇到下降就停。
            # 实测九成以上的迭代第一个 lam 就下降,所以先单独试它;只有它不行
            # 才把剩下 5 个一次批量试完 —— 6 个全批量反而更慢(白算 5 个)。
            try:
                dq0 = np.linalg.solve(jtj + lam * np.eye(7), jte)
            except np.linalg.LinAlgError:
                break
            err0 = self._residual(q + dq0, pos, rot, target_psi)
            cost0 = float(np.linalg.norm(err0))
            if cost0 < cost:
                q, err, cost = q + dq0, err0, cost0
                lam = max(lam / 3.0, 1e-12)
                continue
            lams = lam * 5.0 ** np.arange(1, 6)
            try:
                # b 要给成 (5,7,1):numpy 2.x 的 solve 把 ndim>1 的 b 当矩阵解,
                # 直接传 (5,7) 会被误认成一个 5x7 的右端。
                dq = np.linalg.solve(jtj + lams[:, None, None] * np.eye(7),
                                     np.broadcast_to(jte[:, None], (5, 7, 1)))[..., 0]
            except np.linalg.LinAlgError:
                break
            q_try = q + dq                                        # (5,7)
            err_try = self._residual_batch(q_try, pos, rot, target_psi)
            cost_try = np.linalg.norm(err_try, axis=-1)
            hit = np.nonzero(cost_try < cost)[0]
            if hit.size == 0:           # 阻尼已加到底仍不下降,收敛到局部最好
                break
            k = int(hit[0])
            q, err, cost = q_try[k], err_try[k], float(cost_try[k])
            lam = max(lam * 5.0 ** (k + 1) / 3.0, 1e-12)
        out = IKSolution(q=q, psi=self.sew_angle(q), elbow_branch=sol.elbow_branch,
                         shoulder_branch=sol.shoulder_branch, wrist_branch=sol.wrist_branch,
                         in_limits=self.model.within_limits(q, tol=1e-9))
        self._fill_error(out, pos, rot)
        return out

    # ---------- 冗余自由度选取 ----------

    #: refine 阶段在最优 psi 邻域铺的采样点数(奇数,含中心)。
    REFINE_SAMPLES: int = 17
    #: 精修算到这个位置误差(m)就认为已到数值零,不必再试别的候选。
    #: 正常解落在 1e-13 m 量级,病态候选停在 1e-6,中间空了 7 个量级。
    POLISH_OK_TOL: float = 1e-9

    #: 精度闸门(m)。语义是**硬保证**:解析解误差不超过它就直接采用,所以最终
    #: 误差要么 <= 这个数,要么被精修压到远小于它 —— 除非位姿被关节限位顶住,
    #: 那时误差由几何决定,精修多少轮都没用(实测左臂某帧距限位 0.04 deg,
    #: 误差 0.0904 mm,加到 12 轮纹丝不动)。
    #:
    #: 指标 1 mm,这里取 0.5 mm 留 2 倍硬余量。之所以能放这么松:左臂 URDF 的
    #: 球腕三轴共点残差就有 0.245 mm(右臂 0),解析解最坏 0.327 mm —— 本来就在
    #: 1 mm 以内,再精修纯属浪费。精修在热路径上占 60% 耗时,而机械间隙、编码器
    #: 分辨率都远大于这几位小数。
    #:
    #: ⚠ 收紧回 0.1 mm 指标时把它改回 1e-5:那时左臂 0.327 mm 过不了闸门,
    #:   精修会自动重新启用(代价 +0.7 ms/帧),不需要改别的代码。
    PRECISION_TOL: float = 5e-4

    #: track() 里精修的提前退出判据。比 PRECISION_TOL 严 10 倍 —— 原因见 track()
    #: 内的实测表:早退的残差会随 q 前馈逐帧累积,放到 1e-5 就会顶破 0.1 mm 指标。
    TRACK_POLISH_TOL: float = 1e-6

    def cost_cheap_batch(self, q_arr: np.ndarray, seed: np.ndarray,
                         w_seed: float, w_margin: float) -> np.ndarray:
        """_cost_cheap 的批量版,逐元素与标量版等价。"""
        dist = np.linalg.norm(angle_delta(q_arr, seed), axis=-1)
        margin = self.model.limit_margin_batch(q_arr)
        margin_pen = np.where(margin < 0.25, 1.0 / np.maximum(margin, 1e-3), 0.0)
        return w_seed * dist + w_margin * margin_pen

    def pick_best(self, grid: BatchGrid, seed: np.ndarray,
                  w_seed: float, w_margin: float, w_manip: float,
                  top_k: int = 12,
                  branch_prev: Optional[Tuple[int, int, int]] = None
                  ) -> Optional[Tuple[IKSolution, float, int]]:
        """从一次批量扫描里挑代价最小的解。返回 (解, 代价, 目标下标) 或 None。

        两遍筛选的逻辑与标量版 solve() 完全一致:先用只看关节角的廉价代价取
        top-K,再只对这 K 个算需要雅可比的可操作度惩罚。区别仅在于两遍都是
        批量的 —— K 条雅可比也是一次 manipulability_batch 算完。
        """
        ok = grid.in_limits.copy()
        if branch_prev is not None:
            allowed = np.all(np.asarray(grid.branch) == np.asarray(branch_prev), axis=1)
            ok &= allowed[None, None, :]
        if not ok.any():
            return None
        cheap = np.where(ok, self.cost_cheap_batch(grid.q, seed, w_seed, w_margin),
                         np.inf)
        flat = cheap.ravel()
        n_ok = int(ok.sum())
        take = min(top_k, n_ok)
        if take < flat.size:
            idx = np.argpartition(flat, take - 1)[:take]
        else:
            idx = np.arange(flat.size)
        idx = idx[np.isfinite(flat[idx])]
        if idx.size == 0:
            return None
        total = flat[idx]
        if w_manip > 0.0:
            manip = self.model.manipulability_batch(grid.q.reshape(-1, 7)[idx])
            total = total + w_manip * np.where(manip < 1e-2,
                                               1.0 / np.maximum(manip, 1e-6), 0.0)
        win = int(idx[int(np.argmin(total))])
        tgt, psi_idx, branch = np.unravel_index(win, ok.shape)
        elbow, shoulder, wrist = (int(v) for v in grid.branch[branch])
        sol = IKSolution(q=grid.q[tgt, psi_idx, branch].copy(),
                         psi=float(grid.psi[tgt, psi_idx]),
                         elbow_branch=elbow, shoulder_branch=shoulder,
                         wrist_branch=wrist, in_limits=True)
        return sol, float(total[int(np.argmin(total))]), int(tgt)


    def pick_per_target(self, grid: BatchGrid, seed: np.ndarray,
                        w_seed: float, w_margin: float, w_manip: float,
                        top_k: int = 12
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
        """逐目标挑各自的最优解 —— 滚转枚举要的就是这个。

        返回 (q (R,7), cost (R,), psi (R,), branch (R,3), valid (R,))。
        valid 为 False 的目标其余字段无意义。筛选逻辑与 pick_best 相同,
        只是 argmin 沿目标维分别取。

        R 个目标(= R 个滚转角候选)共用一次 solve_grid,所以整圈滚转扫描的
        代价和单个滚转角几乎一样 —— solve_grasp 从 558 ms 掉到 10 ms 就是这么来的。
        """
        ok = grid.in_limits
        n_tgt = ok.shape[0]
        cheap = np.where(ok, self.cost_cheap_batch(grid.q, seed, w_seed, w_margin),
                         np.inf)
        flat = cheap.reshape(n_tgt, -1)
        q_flat = grid.q.reshape(n_tgt, -1, 7)
        take = min(top_k, flat.shape[1])
        idx = np.argpartition(flat, take - 1, axis=1)[:, :take]        # (R,take)
        cand_cost = np.take_along_axis(flat, idx, axis=1)
        if w_manip > 0.0:
            finite = np.isfinite(cand_cost)
            if finite.any():
                cand_q = np.take_along_axis(q_flat, idx[:, :, None], axis=1)
                manip = np.zeros_like(cand_cost)
                manip[finite] = self.model.manipulability_batch(cand_q[finite])
                pen = np.where(finite & (manip < 1e-2),
                               1.0 / np.maximum(manip, 1e-6), 0.0)
                cand_cost = np.where(finite, cand_cost + w_manip * pen, np.inf)
        win = np.argmin(cand_cost, axis=1)
        best_flat = np.take_along_axis(idx, win[:, None], axis=1)[:, 0]
        cost = np.take_along_axis(cand_cost, win[:, None], axis=1)[:, 0]
        psi_idx, branch_idx = np.unravel_index(best_flat, ok.shape[1:])
        return (q_flat[np.arange(n_tgt), best_flat], cost,
                grid.psi[np.arange(n_tgt), psi_idx],
                grid.branch[branch_idx], np.isfinite(cost))

    def _validate_request(self, pos, rot, seed, max_joint_step_rad=None):
        pos, rot, seed = (np.asarray(value, float) for value in (pos, rot, seed))
        if (pos.shape != (3,) or rot.shape != (3, 3)
                or not np.isfinite(pos).all() or not np.isfinite(rot).all()):
            raise ValueError("目标必须是有限的位置 (3,) 和旋转矩阵 (3,3)")
        if (not np.allclose(rot.T @ rot, np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(rot) - 1.0) > 1e-7):
            raise ValueError("目标旋转矩阵必须属于 SO(3)")
        if (seed.shape != (7,) or not np.isfinite(seed).all()
                or not self.model.within_limits(seed, tol=0)):
            raise ValueError("求解种子必须是限位内的七个有限关节角")
        if max_joint_step_rad is not None and (
                not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0):
            raise ValueError("单步关节变化上限必须是有限正数")
        return pos, rot, seed.copy()

    def _accept_pose(self, q, pos, rot, seed, source, *, template=None,
                     max_joint_step_rad=None):
        """按完整工具模型复核，不将越限或残差超标的候选作为成功结果。"""
        q = np.asarray(q, float)
        if (q.shape != (7,) or not np.isfinite(q).all()
                or not self.model.within_limits(q, tol=1e-9)):
            return None
        if (max_joint_step_rad is not None
                and np.max(np.abs(q - seed)) > max_joint_step_rad):
            return None
        branch = (-1, -1, -1) if template is None else branch_tuple(template)
        sol = IKSolution(q.copy(), self.sew_angle(q), *branch, True, source=source)
        self._fill_error(sol, pos, rot)
        if (not math.isfinite(sol.pos_error) or not math.isfinite(sol.rot_error)
                or sol.pos_error > self.POSITION_TOL
                or sol.rot_error > self.ROTATION_TOL):
            return None
        return sol

    def solve(self, pos: np.ndarray, rot: np.ndarray,
              q_seed: Optional[np.ndarray] = None,
              psi_samples: int = 90,
              psi_hint: Optional[float] = None,
              refine: bool = True,
              polish: bool = True,
              weight_seed: float = 1.0,
              weight_margin: float = 0.6,
              weight_manip: float = 0.15,
              fallback_numeric: bool = True, *,
              max_joint_step_rad: Optional[float] = None) -> Optional[IKSolution]:
        """单目标 IK：原位保持、局部求解、解析搜索及有限次恢复。

        位置与旋转分别按模型内残差验收，失败返回 None。保持原 IKSolution
        返回类型，数值解的解析分支标为 -1。fallback_numeric=False 禁用数值
        精修和备用种子；在线运动使用 track，不在控制循环中做全局恢复。
        """
        if max_joint_step_rad is not None and q_seed is None:
            raise ValueError("指定单步上限时必须提供 q_seed")
        pos, rot, seed = self._validate_request(
            pos, rot, self.q_ref if q_seed is None else q_seed, max_joint_step_rad)
        if isinstance(psi_samples, bool) or not isinstance(psi_samples, (int, np.integer)) or psi_samples < 1:
            raise ValueError("psi_samples 必须是正整数")
        if psi_hint is not None and not math.isfinite(psi_hint):
            raise ValueError("psi_hint 必须是有限数")
        if np.linalg.norm(pos) > self._max_reach + 1e-9:
            return None

        def accept(q, source, template=None):
            return self._accept_pose(q, pos, rot, seed, source, template=template,
                                     max_joint_step_rad=max_joint_step_rad)

        def recover(q, source):
            sol = self.numeric_solve(pos, rot, q_seed=q, iterations=120)
            return None if sol is None else accept(sol.q, source)

        result = accept(seed, "keep_seed")
        if result is not None:
            return result
        if q_seed is not None and fallback_numeric:
            result = recover(seed, "local_refine")
            if result is not None:
                return result
        candidate = self._solve_analytic(
            pos, rot, q_seed=seed, psi_samples=psi_samples, psi_hint=psi_hint,
            refine=refine, polish=polish, weight_seed=weight_seed,
            weight_margin=weight_margin, weight_manip=weight_manip,
            fallback_numeric=False)
        if candidate is not None:
            result = accept(candidate.q, "analytic", candidate)
            if result is not None:
                return result
            if fallback_numeric:
                result = recover(candidate.q, "pose_refine")
                if result is not None:
                    return result
        if fallback_numeric:
            for index, start in enumerate([seed, *self._recovery_seeds]):
                result = recover(start, f"restart_{index}")
                if result is not None:
                    return result
        return None

    def _solve_analytic(self, pos: np.ndarray, rot: np.ndarray,
              q_seed: Optional[np.ndarray] = None,
              psi_samples: int = 90,
              psi_hint: Optional[float] = None,
              refine: bool = True,
              polish: bool = True,
              weight_seed: float = 1.0,
              weight_margin: float = 0.6,
              weight_manip: float = 0.15,
              fallback_numeric: bool = True) -> Optional[IKSolution]:
        """扫 SEW 角挑最优解。

        代价 = 关节空间距离(保连续、不跳解)
             + 限位裕度惩罚(远离限位)
             + 可操作度惩罚(远离奇异)

        psi_hint 给定时会在其附近细扫一遍,用于轨迹跟踪时锁住肘部姿态。
        """
        seed = self.q_ref if q_seed is None else np.asarray(q_seed, float)
        pos = np.asarray(pos, float)
        rot = np.asarray(rot, float)
        grid = list(np.linspace(-np.pi, np.pi, psi_samples, endpoint=False))
        if psi_hint is not None:
            grid += list(psi_hint + np.linspace(-0.35, 0.35, 15))

        # 整个 psi 网格一次批量算完(见 x2_srs_batch),再批量粗筛 top-K、
        # 只对入围候选算含奇异惩罚的完整代价。数学与逐 psi 循环等价。
        picked = self.pick_best(self.batch.solve_grid(pos[None], rot[None],
                                                     np.asarray(grid, float)),
                                seed, weight_seed, weight_margin, weight_manip)
        best = None if picked is None else picked[0]
        best_cost = np.inf if picked is None else picked[1]
        if best is None:
            # 解析枚举全空:大多是完全伸直(SEW 平面退化)这类臂角参数化自身的
            # 算法奇异。退到数值解,能救则救,救不回来才真判不可达。
            if not fallback_numeric:
                return None
            rescued = self.numeric_solve(pos, rot, q_seed=seed)
            if rescued is None or not rescued.in_limits:
                return None
            if rescued.pos_error > 1e-6 or rescued.rot_error > 1e-5:
                return None
            return rescued

        coarse = best
        if refine:
            # 在最优 psi 附近铺一层密网格,一次批量算完。
            # 原来是 3 轮黄金分割(6 次逐 psi 求解,分辨率 span/8);这里 17 点
            # 覆盖 +-span 的分辨率同为 span/8,但只需一次批量调用。
            span = np.pi / max(psi_samples, 1)
            fine = best.psi + np.linspace(-span, span, self.REFINE_SAMPLES)
            # 细化这一步**不做廉价代价粗筛**(top_k 开到整格):粗筛只看关节距离
            # 和限位裕度,而奇异惩罚可以比它们大两个数量级(实测某位姿廉价项 ~4、
            # 完整项 ~356),此时按廉价项排名砍候选会把真正的最优支砍掉。
            # 原来的 3 轮黄金分割也是对每个探到的 psi 算完整代价,这里保持一致。
            cand = self.pick_best(self.batch.solve_grid(pos[None], rot[None], fine),
                                  seed, weight_seed, weight_margin, weight_manip,
                                  top_k=self.REFINE_SAMPLES * 8)
            if cand is not None and cand[1] < best_cost:
                best, best_cost = cand[0], cand[1]

        # 误差只为最终选中的解计算。按代价从低到高试,**第一个能精修到数值零的
        # 即采用** —— 代价低不等于精修得动:近奇异候选处 7x7 增广雅可比病态,
        # LM 可能停在 µm 量级下不去,此时换下一个候选。
        # 现状:细化阶段关掉廉价粗筛之后这条退路 200x2 个位姿一次都没用上
        # (误差全在 1e-13 m 量级),留着是兜底,别因为"没见它触发"就删
        # (细网格找到的候选代价更低,但只精修到 1.5 µm,粗网格那个到 1e-13 m),
        # 这种交易不划算:精度是硬指标,代价只是偏好。都到不了就取误差最小的。
        order = [best] if (not polish or coarse is best) else [best, coarse]
        chosen: Optional[IKSolution] = None
        for cand_sol in order:
            self._fill_error(cand_sol, pos, rot)
            if polish:
                polished = self.polish(cand_sol, pos, rot)
                if polished.in_limits and polished.pos_error <= cand_sol.pos_error:
                    cand_sol = polished
            if chosen is None or cand_sol.pos_error < chosen.pos_error:
                chosen = cand_sol
            if cand_sol.pos_error <= self.POLISH_OK_TOL:
                break
        return chosen


    # ---------- 位置优先:姿态可降级 ----------

    #: 姿态降级的扫描梯度(deg)。前密后疏 —— 绝大多数"拧不过去"只差十几度,
    #: 而真需要 60 deg 以上的那些多半本来就是位置不可达,不值得细扫。
    RELAX_LADDER_DEG: Tuple[float, ...] = (2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0, 90.0)
    #: 扫描期间用的 psi 采样数。比 solve() 默认的 90 稀,单次约 10 ms;
    #: 最终选中的姿态会再跑一遍完整 solve(),所以稀采样只影响"找不找得到",
    #: 不影响最终解的精度。
    RELAX_SCAN_PSI: int = 24

    def solve_hold_rotation(self, pos: np.ndarray, rot_pref: np.ndarray,
                            q_seed: Optional[np.ndarray] = None,
                            max_relax: float = np.pi / 2,
                            ladder_deg: Optional[Sequence[float]] = None,
                            bisect_steps: int = 4) -> Optional["RelaxedSolution"]:
        """位置是硬约束、姿态是软约束的求解:**能保持 rot_pref 就保持,保持不住
        就绕单轴让一点**,让位置仍然到得了。

        为什么要有这个:MDI 里敲三个数只是"把手移到这儿",姿态是上一帧继承下来的,
        并不是用户的诉求。而 X2 的腕部限位相当紧 —— 实测待机姿态 (HOME_Q =
        [0.4, 0, 0, -1.2, 0, 0, 0]) 附近 ±350 mm 的 300 个随机点里,右臂能原样
        保持姿态的只有 143 个,左臂 154 个;剩下的靠本函数降级又救回右 72 / 左 81,
        也就是说**"解不出来"里有近一半只是姿态拧不过去,位置本身够得着**。
        原来的行为是直接判 "IK 无解" 让人自己猜该改哪个 rpy;这里替他把那一下试出来。

        搜索策略:按 RELAX_LADDER_DEG 从小到大逐级放宽,每级枚举 12 个候选
        (torso 系 / TCP 系 × XYZ 三轴 × 正负两向),命中即停 —— 于是拿到的
        必然是**最小可行偏差档**。同级多个候选的姿态偏差都等于该级角度,所以
        排序先比"位置离请求点多远"、再比限位余量:位置是硬约束,不能拿它去换姿态
        (改姿态会挪腕心,连带改变 project_to_workspace 的拉回量),余量则决定
        真机上顶不顶得住 mc 的稳态偏差。命中后在 [上一级, 本级] 之间二分几次
        把偏差磨小,最后对选中的姿态跑一遍完整 solve() 出最终解。

        只放宽**单轴**是刻意的,而且是验证过的:把 32 个球面 Fibonacci 方向
        (近均匀,完全不限于坐标轴)按同一套梯度重撞一遍,本函数判"不可达"的
        85 个点里能救回的是 **0** 个。也就是说在这台臂上,单轴 6 方向 × 正负
        和搜整个三维姿态空间是等价的,多搜只是多花时间。

        (踩过的坑,留给后面改这段的人:用 psi_samples=24 的粗解做上面这个对比
        会得到"Fib 多救回 9 个"的假阳性。那 9 个点是粗解在**原姿态**就漏了解,
        完整 solve() 本来就解得出来,2 deg 扰动只是把稀网格挪到了能命中的地方。
        这类对比实验必须用完整 solve() 复核,否则量到的是采样噪声。)

        返回 None 表示连"任意降级"都救不回来,此时该报的就是真不可达。
        """
        pos = np.asarray(pos, float)
        rot_pref = np.asarray(rot_pref, float)
        seed = self.q_ref if q_seed is None else np.asarray(q_seed, float)
        tried = 0

        def attempt(rot: np.ndarray, cheap: bool,
                    psi_hint: Optional[float] = None):
            """在给定姿态下解一次,顺带把出界的位置拉回工作空间。"""
            p_c, clipped = self.project_to_workspace(pos, rot)
            if cheap:
                sol = self.solve(p_c, rot, q_seed=seed, psi_samples=self.RELAX_SCAN_PSI,
                                 refine=False, polish=False, fallback_numeric=False)
            else:
                sol = self.solve(p_c, rot, q_seed=seed, psi_hint=psi_hint)
            return sol, p_c, bool(clipped)

        def relaxed(frame: str, axis: int, sign: float, angle: float) -> np.ndarray:
            """绕单轴偏一点。torso 系是左乘(和 MDI 的 `R` 同义),TCP 系是右乘
            (和 `t` 同义)。

            两种都试**不是**因为左乘右乘能到达不同的姿态集合 —— 恰恰相反,
            R0·exp(θ m̂) = exp(θ·R0 m̂)·R0,离 R0 距离 θ 的姿态就那一个球面,
            左乘右乘扫的是同一个集合。两种都列是为了拿到 6 个**不同的方向**
            (torso 的 XYZ 与 TCP 的 XYZ 一般不重合),同时两边都有人话可讲:
            报出来的 "绕 torso 系 Y 轴让了 6 度" 是操作者能直接想象的量。
            """
            d_rot = axis_angle_to_matrix(_RELAX_AXES[axis], sign * angle)
            return d_rot @ rot_pref if frame == "torso" else rot_pref @ d_rot

        def wrap(sol: IKSolution, p_used: np.ndarray, rot_used: np.ndarray,
                 axis: Optional[int], frame: str, clipped: bool) -> "RelaxedSolution":
            # 偏差从矩阵实测,不用搜索时的名义角度 —— 顺带校验 relaxed() 没写错
            dev = float(np.linalg.norm(log3(rot_pref.T @ np.asarray(rot_used, float))))
            return RelaxedSolution(sol=sol, pos=p_used, rot=np.asarray(rot_used, float),
                                   deviation=dev, axis=axis, frame=frame,
                                   clipped=clipped, tried=tried)

        # 第 0 级:先老老实实按首选姿态解。绝大多数情况到这里就返回了。
        sol, p_c, clipped = attempt(rot_pref, cheap=False)
        tried += 1
        if sol is not None:
            return wrap(sol, p_c, rot_pref.copy(), None, "", clipped)
        if max_relax <= 0.0:
            return None

        # 梯度按 max_relax 截断:最后一级正好落在上限上,于是 "relax 20" 真的
        # 会去试 20 deg,而不是在 15 deg 就放弃。
        ladder: List[float] = []
        for deg in (self.RELAX_LADDER_DEG if ladder_deg is None else ladder_deg):
            angle = min(float(np.radians(deg)), float(max_relax))
            if not ladder or angle > ladder[-1] + 1e-9:
                ladder.append(angle)
        if not ladder:
            return None

        # 一级里的 12 个候选是 12 个**独立目标**(各自的姿态、各自被拉回的位置),
        # 正好是 solve_grid 的形状,所以一级只需一次批量调用 —— 逐个 cheap solve()
        # 是 12 x 1.1 ms,批量是 1.5 ms。判据、排序、命中即停的语义都不变。
        combos = [(frame, axis, sign)
                  for frame in ("torso", "tcp")
                  for axis in range(3)
                  for sign in (1.0, -1.0)]
        scan_grid = np.linspace(-np.pi, np.pi, self.RELAX_SCAN_PSI, endpoint=False)

        prev = 0.0
        hit = None
        for angle in ladder:
            rots = np.stack([relaxed(f, a, sg, angle) for f, a, sg in combos])
            projected = [self.project_to_workspace(pos, r) for r in rots]
            pts = np.stack([p for p, _ in projected])
            q_arr, _c, psi_arr, br_arr, valid = self.pick_per_target(
                self.batch.solve_grid(pts, rots, scan_grid), seed, 1.0, 0.6, 0.15)
            tried += len(combos)
            idx = np.nonzero(valid)[0]
            if idx.size:
                # 同一级里所有候选的姿态偏差都等于 angle,所以排序只看别的:先比
                # 位置离请求点多远 —— 位置是硬约束,不能拿它去换姿态(改姿态会挪
                # 腕心,进而改变 project_to_workspace 的拉回量,不盯着的话真会越
                # 降级越偏);同挡再比限位余量。lexsort 是稳定排序,两项都相同时
                # 仍按 combos 的原顺序,与原来的 list.sort 一致。
                dist = np.round(np.linalg.norm(pts[idx] - pos, axis=-1), 4)
                margin = self.model.limit_margin_batch(q_arr[idx])
                k = int(idx[np.lexsort((-margin, dist))[0]])
                frame, axis, sign = combos[k]
                sol = IKSolution(q=q_arr[k].copy(), psi=float(psi_arr[k]),
                                 elbow_branch=int(br_arr[k][0]),
                                 shoulder_branch=int(br_arr[k][1]),
                                 wrist_branch=int(br_arr[k][2]), in_limits=True)
                hit = (angle, frame, axis, sign, sol, rots[k], pts[k],
                       bool(projected[k][1]))
                break
            prev = angle
        if hit is None:
            return None

        angle, frame, axis, sign, sol, rot_best, p_best, clip_best = hit

        # 二分把偏差磨到梯度之间。不假设可行性对 angle 单调 —— 只保留已知可行的
        # 那一侧,中点不可行就抬下界,可行就收上界,无论单调与否结果都合法。
        lo, hi = prev, angle
        for _ in range(max(0, bisect_steps)):
            mid = 0.5 * (lo + hi)
            rot_mid = relaxed(frame, axis, sign, mid)
            sol_mid, p_mid, clip_mid = attempt(rot_mid, cheap=True)
            tried += 1
            if sol_mid is None:
                lo = mid
            else:
                hi, sol, rot_best, p_best, clip_best = mid, sol_mid, rot_mid, p_mid, clip_mid

        # 定稿:对选中的姿态跑完整 solve()。psi_hint 给粗解的 psi —— 粗扫的
        # 24 点网格并不是细扫 90 点的子集,不给提示的话完整解偶尔反而找不到。
        final, p_final, clip_final = attempt(rot_best, cheap=False, psi_hint=sol.psi)
        tried += 1
        if final is not None:
            return wrap(final, p_final, rot_best, axis, frame, clip_final)
        return wrap(sol, p_best, rot_best, axis, frame, clip_best)   # 粗解兜底

    def numeric_solve(self, pos: np.ndarray, rot: np.ndarray,
                      q_seed: Optional[np.ndarray] = None,
                      iterations: int = 120,
                      tol: float = 1e-12) -> Optional[IKSolution]:
        """6 维任务空间的阻尼最小二乘兜底解。

        为什么需要它:臂角(SEW)参数化在肩-肘-腕共线时本身退化 —— 此时肘点圆
        半径为 0,psi 没有定义,`solve_at_psi` 一个解支都构造不出来。但那些位姿
        本身是完全可达的(实测左臂 200 个随机位姿里有 3 个正好落在完全伸直附近)。
        这里不带 psi 约束,只对 6 维位姿残差做 LM 迭代,并在迭代中夹限位,
        把这类"解析法自己的奇异"救回来。返回解的 psi 字段填数值反算值。

        注意:这是兜底,不是主路径。主路径仍是解析枚举 —— 兜底解不保证是
        全局最优支,也不保证肘部姿态可控。
        """
        pos = np.asarray(pos, float)
        rot = np.asarray(rot, float)
        q = self.model.clamp(self.q_ref if q_seed is None else np.asarray(q_seed, float))

        def residual6(qq: np.ndarray) -> np.ndarray:
            fk_pos, fk_rot = self.model.forward_kinematics(qq)
            return np.concatenate((pos - fk_pos, log3(rot @ fk_rot.T)))

        err = residual6(q)
        cost = float(np.linalg.norm(err))
        lam = 1e-6
        for _ in range(iterations):
            if cost < tol:
                break
            jac = self.model.jacobian(q)
            jjt = jac @ jac.T
            improved = False
            for _try in range(8):
                try:
                    dq = jac.T @ np.linalg.solve(jjt + lam * np.eye(6), err)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                q_new = self.model.clamp(q + dq)
                err_new = residual6(q_new)
                cost_new = float(np.linalg.norm(err_new))
                if cost_new < cost:
                    q, err, cost = q_new, err_new, cost_new
                    lam = max(lam / 3.0, 1e-12)
                    improved = True
                    break
                lam *= 5.0
            if not improved:
                break

        sol = IKSolution(q=q, psi=self.sew_angle(q), elbow_branch=-1,
                         shoulder_branch=-1, wrist_branch=-1,
                         in_limits=self.model.within_limits(q))
        self._fill_error(sol, pos, rot)
        return sol

    def _cost_cheap(self, sol: IKSolution, seed: np.ndarray,
                    w_seed: float, w_margin: float) -> float:
        """代价的前两项。只用关节角,不碰雅可比。"""
        dist = float(np.linalg.norm(angle_delta(sol.q, seed)))
        margin = self.model.limit_margin(sol.q)
        # 注意这两个惩罚项都是**阶跃**的:margin 刚过 0.25 罚 0、刚不到罚 4
        # (乘 w_margin 0.6 = 跳 2.4);manip 刚过 1e-2 罚 0、刚不到罚 100
        # (乘 w_manip 0.15 = 跳 15)。而本机型 manip 的正常量级就在 1e-2 附近,
        # 阈值正落在工作区间中间,于是"代价"在那条线两侧能差 100 倍以上,而两个
        # 解的真实可操作度只差千分之几 —— 拿代价做 A/B 对比会读到假的巨大退化
        # (实测同一位姿 0.0878 vs 15.087,可操作度实际只差 0.3%)。
        # 想改成连续势垒(减去阈值处的值)是合理的,但那会改变全局选解偏好,
        # 属于独立改动,别顺手塞进性能优化里。
        margin_pen = 1.0 / max(margin, 1e-3) if margin < 0.25 else 0.0
        return w_seed * dist + w_margin * margin_pen

    def _cost(self, sol: IKSolution, seed: np.ndarray,
              w_seed: float, w_margin: float, w_manip: float) -> float:
        """完整代价。奇异惩罚需要可操作度,即一条 6x7 雅可比 —— 这是单个候选
        里最贵的一步,所以扫 psi 时先用 _cost_cheap 粗筛,只对入围的少数候选
        算完整代价。manip_pen 在远离奇异时恒为 0,粗筛因此几乎不改变结果。"""
        if w_manip <= 0.0:
            return self._cost_cheap(sol, seed, w_seed, w_margin)
        manip = self.model.manipulability(sol.q)
        manip_pen = 1.0 / max(manip, 1e-6) if manip < 1e-2 else 0.0
        return self._cost_cheap(sol, seed, w_seed, w_margin) + w_manip * manip_pen

    # ---------- 轨迹跟踪 (热启动) ----------

    def _track_branch(self, sol):
        """用候选自身的 FK/SEW 反查解析分支，无法可靠对应时拒绝跟踪。"""
        pos, rot = self.model.forward_kinematics(sol.q)
        grid = self.batch.solve_grid(pos[None], rot[None], np.array([sol.psi]))
        q = grid.q.reshape(-1, 7)
        distances = np.max(np.abs(angle_delta(q, sol.q)), axis=1)
        distances = np.where(np.isfinite(q).all(axis=1), distances, np.inf)
        index = int(np.argmin(distances))
        # 左腕轴的非共点误差会使解析反算略有偏差，偏差过大不猜分支。
        if distances[index] > self.TRACK_MAX_STEP:
            return None
        return tuple(int(v) for v in grid.branch[index % len(grid.branch)])

    def track(self, pos: np.ndarray, rot: np.ndarray,
              q_prev: np.ndarray,
              psi_prev: Optional[float] = None,
              psi_desired: Optional[float] = None,
              psi_window: float = 0.20,
              psi_samples: int = 9,
              fallback: bool = True,
              clamp_reach: bool = True,
              branch_prev: Optional[Tuple[int, int, int]] = None, *,
              max_joint_step_rad: float = TRACK_MAX_STEP) -> Optional[IKSolution]:
        """逐帧局部求解，检查位姿残差、关节步长及解析分支。

        当前姿态已满足目标时保持原位，否则先做有限迭代的六维局部精修，
        再尝试原 SEW 邻域搜索。显式指定 psi_desired 时沿用臂角搜索。
        fallback=False 时不调用全局 solve，也不使用备用种子。
        """
        if max_joint_step_rad is None:
            raise ValueError("轨迹求解必须指定单步关节变化上限")
        pos, rot, q_prev = self._validate_request(pos, rot, q_prev, max_joint_step_rad)
        if (not math.isfinite(psi_window) or psi_window <= 0
                or isinstance(psi_samples, bool)
                or not isinstance(psi_samples, (int, np.integer)) or psi_samples < 1):
            raise ValueError("轨迹臂角窗口和采样数必须为正")
        for psi in (psi_prev, psi_desired):
            if psi is not None and not math.isfinite(psi):
                raise ValueError("轨迹臂角必须为有限数")
        if branch_prev is not None:
            branch_prev = tuple(branch_prev)
            if len(branch_prev) != 3 or any(v not in (0, 1) for v in branch_prev):
                raise ValueError("branch_prev 必须是三个 0/1 解析分支编号")
        if clamp_reach:
            pos, _ = self.project_to_workspace(pos, rot)
        elif np.linalg.norm(pos) > self._max_reach + 1e-9:
            return None
        center = self.sew_angle(q_prev) if psi_prev is None else float(psi_prev)

        def accept(q, source):
            sol = self._accept_pose(q, pos, rot, q_prev, source,
                                    max_joint_step_rad=max_joint_step_rad)
            if sol is None:
                return None
            branch = self._track_branch(sol)
            if branch is None or (branch_prev is not None and branch != branch_prev):
                return None
            sol.elbow_branch, sol.shoulder_branch, sol.wrist_branch = branch
            return sol

        if psi_desired is None:
            result = accept(q_prev, "keep_seed")
            if result is not None:
                return result
            local = self.numeric_solve(pos, rot, q_seed=q_prev,
                                       iterations=self.TRACK_LOCAL_ITERATIONS)
            if local is not None:
                result = accept(local.q, "local_refine")
                if (result is not None
                        and abs(float(angle_delta(result.psi, center))) <= psi_window):
                    return result
        candidate = self._track_analytic(
            pos, rot, q_prev=q_prev, psi_prev=psi_prev, psi_desired=psi_desired,
            psi_window=psi_window, psi_samples=psi_samples, fallback=False,
            clamp_reach=False, branch_prev=branch_prev)
        if candidate is not None:
            result = accept(candidate.q, "analytic_track")
            if result is not None:
                return result
            # 保留 SEW 约束及指定分支，对邻域候选增加一轮有限精修。
            refined = self.polish(candidate, pos, rot, iterations=12,
                                   tol=self.TRACK_POLISH_TOL)
            result = accept(refined.q, "track_polish")
            if result is not None:
                return result
        if fallback:
            candidate = self.solve(pos, rot, q_seed=q_prev, psi_hint=center,
                                   max_joint_step_rad=max_joint_step_rad)
            if candidate is not None:
                return accept(candidate.q, "track_fallback")
        return None

    def _track_analytic(self, pos: np.ndarray, rot: np.ndarray,
              q_prev: np.ndarray,
              psi_prev: Optional[float] = None,
              psi_desired: Optional[float] = None,
              psi_window: float = 0.20,
              psi_samples: int = 9,
              fallback: bool = True,
              clamp_reach: bool = True,
              branch_prev: Optional[Tuple[int, int, int]] = None) -> Optional[IKSolution]:
        """原 SEW 邻域搜索，候选由 track 统一复核后返回。"""
        q_prev = np.asarray(q_prev, float)
        if clamp_reach:
            pos, _clamped = self.project_to_workspace(pos, rot)
        center = self.sew_angle(q_prev) if psi_prev is None else float(psi_prev)
        if psi_desired is not None:      # 允许上层主动挪肘(避障 / 摆姿态)
            center = float(psi_desired)

        # 邻域网格一次批量算完;代价就是关节空间距离(w_margin/w_manip 置 0),
        # 与逐 psi 循环的挑法逐字等价。
        psi_grid = np.linspace(center - psi_window, center + psi_window, psi_samples)
        picked = self.pick_best(
            self.batch.solve_grid(np.asarray(pos, float)[None],
                                  np.asarray(rot, float)[None], psi_grid),
            q_prev, 1.0, 0.0, 0.0, branch_prev=branch_prev)
        best = None if picked is None else picked[0]
        if best is None:
            return self.solve(pos, rot, q_seed=q_prev, psi_hint=center) if fallback else None

        self._fill_error(best, pos, rot)
        # 精修默认跳过。实测(500 帧连续轨迹,右臂):
        #     带精修  中位 1.662 ms  95% 1.841  最坏 3.378  最大误差 1.5e-10 mm
        #     跳精修  中位 0.695 ms  95% 0.733  最坏 0.975  最大误差 2.1e-5  mm
        # 500 Hz 环单臂预算 2 ms —— 带精修的最坏值 3.378 ms **已经超时**,
        # 跳过后双臂合计 1.95 ms 才压得进去。而 2.1e-5 mm 离 0.1 mm 指标还有
        # 5000 倍余量,所以这是"砍掉用不上的小数位",不是拿精度换速度。
        #
        # 闸门仍然留着:解析残差真超标时(近奇异位姿、或换了球腕共点残差更大的
        # 机型)照样精修,不会因为改了默认行为就失去兜底。
        if best.pos_error > self.PRECISION_TOL:
            # tol 用 TRACK_POLISH_TOL 而**不是** PRECISION_TOL:polish 的 cost 是
            # [位姿(m/rad)x6, SEW(rad)] 的 7 维范数,提前退出的误差会经 q 前馈到
            # 下一帧、逐帧累积。实测左臂同一条轨迹:
            #     tol=1e-12(跑满)  中位 1.66 ms  最坏误差 0.0904 mm
            #     tol=1e-6         中位 1.17 ms  最坏误差 0.0904 mm  <- 采用
            #     tol=1e-5         中位 1.15 ms  最坏误差 0.1294 mm  <- 超 0.1mm 指标
            # 所以这里必须比 PRECISION_TOL 再严 10 倍,不能图省事复用同一个常量。
            polished = self.polish(best, pos, rot, iterations=4,
                                   tol=self.TRACK_POLISH_TOL)
            if polished.in_limits and polished.pos_error <= best.pos_error:
                best = polished
        return best

    # ---------- 速度级 (供力矩/速度接口使用) ----------

    def inverse_velocity(self, q: np.ndarray, twist: np.ndarray,
                         psi_dot: float = 0.0) -> np.ndarray:
        """由末端 twist + SEW 角速度求关节速度。

        7 自由度臂配 6 维 twist 本来欠定;把 psi_dot 补成第 7 个方程后系统方阵,
        直接得到唯一且几何意义明确的解 —— 比 A3 那种 Jᵀ(JJᵀ+λ²I)⁻¹ 伪逆
        少一层阻尼失真,也不需要零空间增益调参。
        """
        q = np.asarray(q, float)
        jac = self._augmented_jacobian(q)
        rhs = np.concatenate([np.asarray(twist, float).reshape(6), [psi_dot]])
        dq, *_ = np.linalg.lstsq(jac, rhs, rcond=None)
        return dq


if __name__ == "__main__":
    for side in ("right", "left"):
        model = ArmModel(side)
        ik = SrsArmIK(model)
        print(f"===== {side} =====")
        print(f"  肘部常量偏置 : {np.degrees(ik.elbow_offset):+.3f} deg")
        print(f"  几何伸展范围 : {ik.reach_min * 1000:.2f} ~ {ik.reach_max * 1000:.2f} mm")
        print(f"  受限伸展范围 : {ik.reach_min_limited * 1000:.2f} ~ "
              f"{ik.reach_max_limited * 1000:.2f} mm  (肘限位 "
              f"[{model.q_min[3]:.4f}, {model.q_max[3]:.4f}] rad)")
        q_test = ik.q_ref.copy()
        p_t, r_t = model.forward_kinematics(q_test)
        sol = ik.solve(p_t, r_t, q_seed=q_test)
        if sol is None:
            print("  自检失败: 无解")
        else:
            print(f"  自检 pos_err = {sol.pos_error * 1000:.6f} mm, "
                  f"rot_err = {np.degrees(sol.rot_error):.6f} deg, psi = {np.degrees(sol.psi):+.2f} deg")
        print()
