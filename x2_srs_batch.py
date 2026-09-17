#!/usr/bin/env python3
"""解析 IK 的**批量**求解核心 —— 把 psi 扫描 / 滚转枚举整体搬到 numpy 向量化。

为什么需要这一层
----------------
`SrsArmIK.solve_at_psi` 每次调用只处理一个 psi,内部做的全是 3 维向量和 3x3
矩阵的运算。这类运算单次只有几十次浮点乘法,但每一个 numpy 调用要付 0.5~1 us
的解释器开销 —— 于是实测里 **99% 的时间花在函数调用上,不在数学上**:

    solve(psi_samples=48)  = 15.5 ms
      其中 48 次 solve_at_psi = 11.2 ms  (72%)
      单次 solve_at_psi 内约 40 次 numpy 调用 x 24 个子问题调用

而扫 psi 时,绝大多数中间量**与 psi 无关**:目标腕心、肩腕距离、肘角 q4、
w_hat、肘圆半径、以及 `_frame_from_two_vectors(u1, w_hat)` 这半边基,都只跟
目标位姿有关。真正随 psi 变的只有肘点在圆上的位置。所以整个扫描可以重排成
"一次 (R,P,...) 的广播运算",numpy 调用次数从 O(R·P) 降到 O(1)。

R 是目标位姿个数 —— 这一维是为 `solve_axis` 的滚转枚举准备的:绕工具轴转 φ
得到的每个候选姿态都是一个独立目标,过去要跑 φ 次 solve(),现在一次批量
就能把 (滚转 x psi) 的整张表算完。

数学与标量版逐字一致
--------------------
本模块不改任何公式,只改求值顺序,因此结果与 `solve_at_psi` 在机器精度内相同
(`selfcheck` 会逐解支对拉,实测最大偏差 ~1e-16 rad)。两处刻意保留的差异:

  1. 标量版在"手臂完全伸直"(phi <= 1e-9)时只返回 1 个肘角,批量版恒返回 2 个
     —— 此时两者数值相同,多出来的那支是无害的重复候选。
  2. `subproblem2` 的 gamma <= tol 退化同理:标量版返回 1 组,批量版返回 2 组
     重复。挑最优时重复候选不改变结果。

分支编号与标量版对齐:branch = elbow*4 + shoulder*2 + wrist,
其中 shoulder/wrist 的 0 支对应 subproblem2 的 sign=+1。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# 批量基元
# --------------------------------------------------------------------------

def bcross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """批量 3 维叉乘。(...,3) x (...,3) -> (...,3)。

    不用 np.cross:它为支持任意 axis / 2 维输入做了一堆前置分派,
    在这里比展开的三行乘法还贵。
    """
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def brodrigues(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """常量单位轴 + 批量角度 -> (...,3,3)。axis 必须已归一化。"""
    kx, ky, kz = (float(v) for v in axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ver = 1.0 - cos_a
    out = np.empty(np.shape(angle) + (3, 3))
    out[..., 0, 0] = cos_a + ver * kx * kx
    out[..., 0, 1] = ver * kx * ky - sin_a * kz
    out[..., 0, 2] = ver * kx * kz + sin_a * ky
    out[..., 1, 0] = ver * kx * ky + sin_a * kz
    out[..., 1, 1] = cos_a + ver * ky * ky
    out[..., 1, 2] = ver * ky * kz - sin_a * kx
    out[..., 2, 0] = ver * kx * kz - sin_a * ky
    out[..., 2, 1] = ver * ky * kz + sin_a * kx
    out[..., 2, 2] = cos_a + ver * kz * kz
    return out


def bsubproblem1(axis: np.ndarray, vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """subproblem1 的批量版。axis 为常量 (3,),vec_from/vec_to 可为 (3,) 或 (...,3)。"""
    a_par = np.asarray(vec_from @ axis)[..., None] * axis
    b_par = np.asarray(vec_to @ axis)[..., None] * axis
    a_perp = vec_from - a_par
    b_perp = vec_to - b_par
    return np.arctan2(bcross(a_perp, b_perp) @ axis,
                      np.sum(a_perp * b_perp, axis=-1))


def bframe_from_two_vectors(primary: np.ndarray, secondary: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """_frame_from_two_vectors 的批量版。返回 (基 (...,3,3), 有效 (...))。

    列向量为基向量;两输入共线时该样本标为无效(基退化)。
    """
    norm1 = np.linalg.norm(primary, axis=-1, keepdims=True)
    e1 = primary / np.where(norm1 < _EPS, 1.0, norm1)
    e2 = secondary - np.sum(secondary * e1, axis=-1, keepdims=True) * e1
    norm2 = np.linalg.norm(e2, axis=-1, keepdims=True)
    valid = (norm2[..., 0] >= 1e-9) & (norm1[..., 0] >= _EPS)
    e2 = e2 / np.where(norm2 < 1e-9, 1.0, norm2)
    e3 = bcross(e1, e2)
    e1b, e2b, e3b = np.broadcast_arrays(e1, e2, e3)
    return np.stack([e1b, e2b, e3b], axis=-1), valid


class RotationDecomposer:
    """把 R 分解为 Rot(a1,t1) Rot(a2,t2) Rot(a3,t3) 的批量版。

    三根轴在整个求解过程中是常量(零位时 base 系下的关节旋量),所以
    dot12 / denom / cross12 / probe 这些量都能在构造时算一次就固定下来。
    这是批量化能吃到的第二块收益 —— 标量版每次调用都要重算它们。
    """

    def __init__(self, axis1: np.ndarray, axis2: np.ndarray, axis3: np.ndarray):
        self.a1 = np.asarray(axis1, float)
        self.a2 = np.asarray(axis2, float)
        self.a3 = np.asarray(axis3, float)
        self.dot12 = float(self.a1 @ self.a2)
        self.denom = 1.0 - self.dot12 * self.dot12
        self.degenerate = abs(self.denom) < 1e-10        # 两轴平行,无解
        self.cross12 = bcross(self.a1, self.a2)
        self.d2n = float(self.a2 @ self.a3)
        self.from_sq = float(self.a3 @ self.a3)
        probe = np.array([1.0, 0.0, 0.0])
        if abs(float(probe @ self.a3)) > 0.9:
            probe = np.array([0.0, 1.0, 0.0])
        probe = probe - (probe @ self.a3) * self.a3
        self.probe = probe / np.linalg.norm(probe)

    def __call__(self, rot: np.ndarray, tol: float = 1e-7
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """rot (...,3,3) -> (angles (...,2,3), valid (...,2))。

        末二维的 2 是 subproblem2 的两个符号支,3 是 (t1,t2,t3)。
        """
        lead = rot.shape[:-2]
        if self.degenerate:
            return np.zeros(lead + (2, 3)), np.zeros(lead + (2,), bool)
        target = rot @ self.a3                                    # (...,3)
        d1t = target @ self.a1                                    # (...)
        alpha = (d1t - self.dot12 * self.d2n) / self.denom
        beta = (self.d2n - self.dot12 * d1t) / self.denom
        gamma_sq = (self.from_sq - alpha * alpha - beta * beta
                    - 2.0 * alpha * beta * self.dot12) / self.denom
        valid = gamma_sq >= -tol
        gamma = np.sqrt(np.maximum(gamma_sq, 0.0))

        signs = np.array([1.0, -1.0])
        # mid (...,2,3):两个符号支一起算,省掉一次 Python 层循环
        mid = (alpha[..., None, None] * self.a1
               + beta[..., None, None] * self.a2
               + (gamma[..., None] * signs)[..., None] * self.cross12)
        t2 = bsubproblem1(self.a2, self.a3, mid)                  # (...,2)
        t1 = bsubproblem1(self.a1, mid, target[..., None, :])     # (...,2)
        pre = brodrigues(self.a1, t1) @ brodrigues(self.a2, t2)   # (...,2,3,3)
        residual = np.swapaxes(pre, -1, -2) @ rot[..., None, :, :]
        t3 = bsubproblem1(self.a3, self.probe, residual @ self.probe)
        return np.stack([t1, t2, t3], axis=-1), np.broadcast_to(
            valid[..., None], t1.shape)


# --------------------------------------------------------------------------
# 批量解集
# --------------------------------------------------------------------------

@dataclass
class BatchGrid:
    """一次批量扫描的全部解支。

    q          : (R, P, 8, 7) 关节角
    ok         : (R, P, 8)    几何有效(基不退化、gamma 实数、在球壳内)
    in_limits  : (R, P, 8)    ok 且满足关节限位
    psi        : (R,P)       扫描用的 SEW 角(共用网格时每行相同)
    branch     : (8, 3)       每个 branch 对应的 (elbow, shoulder, wrist) 支号
    """
    q: np.ndarray
    ok: np.ndarray
    in_limits: np.ndarray
    psi: np.ndarray
    branch: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.in_limits.shape          # type: ignore[return-value]

    def count(self) -> int:
        return int(self.in_limits.sum())


class BatchSrsCore:
    """`SrsArmIK` 的批量后端。构造一次,复用常量;每次调用只做广播运算。"""

    #: branch = elbow*4 + shoulder*2 + wrist
    BRANCH_TABLE = np.array([(e, s, w) for e in (0, 1) for s in (0, 1) for w in (0, 1)])

    def __init__(self, ik: "object"):
        self.ik = ik
        axes = ik.axes0                                   # type: ignore[attr-defined]
        self.dec_shoulder = RotationDecomposer(axes[0], axes[1], axes[2])
        self.dec_wrist = RotationDecomposer(axes[4], axes[5], axes[6])
        self.a4 = np.asarray(axes[3], float)
        self.rot_ee0_t = np.ascontiguousarray(ik.rot_ee0.T)   # type: ignore[attr-defined]
        self.u1_hat = ik.u1 / np.linalg.norm(ik.u1)           # type: ignore[attr-defined]

    # ---------- 与 psi 无关的前置量 ----------

    def _sew_basis(self, v_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_sew_basis 的批量版:主参考共线时逐样本换备用参考。"""
        ik = self.ik
        refs = (ik.sew_reference, ik.sew_reference_alt,      # type: ignore[attr-defined]
                np.array([0.0, 1.0, 0.0]))
        n_vec = None
        for ref in refs:
            cand = ref - np.sum(ref * v_hat, axis=-1, keepdims=True) * v_hat
            if n_vec is None:
                n_vec = cand
            else:
                bad = np.linalg.norm(n_vec, axis=-1, keepdims=True) < 1e-6
                n_vec = np.where(bad, cand, n_vec)
        norm = np.linalg.norm(n_vec, axis=-1, keepdims=True)
        n_hat = n_vec / np.where(norm < _EPS, 1.0, norm)
        return n_hat, bcross(v_hat, n_hat)

    def _elbow_angles(self, distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """elbow_angles 的批量版。返回 (q4 (...,2), 有效 (...))。"""
        ik = self.ik
        tol = ik.reach_tol                                   # type: ignore[attr-defined]
        if ik.len1 < _EPS or ik.len2_perp < _EPS:            # type: ignore[attr-defined]
            return np.zeros(distance.shape + (2,)), np.zeros(distance.shape, bool)
        dist_c = np.clip(distance, ik.reach_min - tol, ik.reach_max + tol)  # type: ignore[attr-defined]
        cos_val = ((dist_c ** 2 - ik.len1 ** 2 - ik.u2_norm_sq)             # type: ignore[attr-defined]
                   / (2.0 * ik.len1 * ik.len2_perp))                        # type: ignore[attr-defined]
        valid = np.abs(cos_val) <= 1.0 + 1e-6
        phi = np.arccos(np.clip(cos_val, -1.0, 1.0))
        raw = np.stack([phi - ik.elbow_offset, -phi - ik.elbow_offset], axis=-1)  # type: ignore[attr-defined]
        return np.arctan2(np.sin(raw), np.cos(raw)), valid

    # ---------- 主入口 ----------

    def solve_grid(self, pos_arr: np.ndarray, rot_arr: np.ndarray,
                   psi: np.ndarray) -> BatchGrid:
        """对 R 个目标位姿 x P 个 SEW 角,一次算出全部 8 个解支。

        pos_arr (R,3) / rot_arr (R,3,3) / psi (P,) 或 (R,P) -> BatchGrid,q 形状 (R,P,8,7)。
        """
        ik = self.ik
        model = ik.model                                     # type: ignore[attr-defined]
        pos_arr = np.atleast_2d(np.asarray(pos_arr, float))
        rot_arr = np.asarray(rot_arr, float).reshape(-1, 3, 3)
        # psi 可以是 (P,) 全目标共用,也可以是 (R,P) 每个目标各自一套 —— 后者
        # 用于"逐滚转角各自细化自己的 psi 窗口",让 R 次细化合并成一次调用。
        psi = np.atleast_1d(np.asarray(psi, float))
        n_tgt = pos_arr.shape[0]
        if psi.ndim == 1:
            psi = np.broadcast_to(psi, (n_tgt, psi.shape[0]))
        elif psi.ndim != 2 or psi.shape[0] != n_tgt:
            raise ValueError(f"psi 必须是 (P,) 或 (R,P),R={n_tgt},实得 {psi.shape}")
        n_psi = psi.shape[1]

        # --- 1. 目标腕心与肩腕距离(与 psi 无关) ---
        rot_wrist = rot_arr @ model.tcp_rotation.T
        ee_origin = pos_arr - rot_wrist @ model.tcp_offset
        wrist_t = ee_origin + rot_wrist @ ik.wrist_local      # type: ignore[attr-defined]
        v_vec = wrist_t - ik.shoulder                         # type: ignore[attr-defined]
        dist = np.linalg.norm(v_vec, axis=-1)
        reach_ok = ((dist >= 1e-6)
                    & (dist <= ik.reach_max + ik.reach_tol)   # type: ignore[attr-defined]
                    & (dist >= ik.reach_min - ik.reach_tol))  # type: ignore[attr-defined]
        v_hat = v_vec / np.where(dist < _EPS, 1.0, dist)[:, None]
        n_hat, b_hat = self._sew_basis(v_hat)                 # (R,3) each

        # --- 2. 肘角两支 + 肘圆几何(与 psi 无关) ---
        q4, elbow_ok = self._elbow_angles(dist)               # (R,2), (R,)
        rot_a4 = brodrigues(self.a4, q4)                      # (R,2,3,3)
        w_vec = ik.u1 + rot_a4 @ ik.u2                        # type: ignore[attr-defined]
        w_norm = np.linalg.norm(w_vec, axis=-1, keepdims=True)
        w_hat = w_vec / np.where(w_norm < _EPS, 1.0, w_norm)   # (R,2,3)
        height = w_hat @ ik.u1                                # type: ignore[attr-defined]
        radius_sq = ik.len1 ** 2 - height ** 2                # type: ignore[attr-defined]
        radius_ok = radius_sq >= -1e-9
        radius = np.sqrt(np.maximum(radius_sq, 0.0))           # (R,2)
        src, src_ok = bframe_from_two_vectors(
            np.broadcast_to(ik.u1, w_hat.shape), w_hat)        # type: ignore[attr-defined]

        # --- 3. 肘点在圆上的位置(唯一随 psi 变的量) ---
        circ = (np.cos(psi)[:, :, None] * n_hat[:, None, :]
                + np.sin(psi)[:, :, None] * b_hat[:, None, :])             # (R,P,3)
        elbow_rel = (height[:, :, None, None] * v_hat[:, None, None, :]
                     + radius[:, :, None, None] * circ[:, None, :, :])     # (R,2,P,3)

        # --- 4. 肩部旋转与腕部旋转 ---
        dst, dst_ok = bframe_from_two_vectors(
            elbow_rel, np.broadcast_to(v_hat[:, None, None, :], elbow_rel.shape))
        rot_s = dst @ np.swapaxes(src, -1, -2)[:, :, None]                  # (R,2,P,3,3)
        rot_w = (np.swapaxes(rot_a4, -1, -2)[:, :, None]
                 @ np.swapaxes(rot_s, -1, -2)
                 @ rot_wrist[:, None, None] @ self.rot_ee0_t)               # (R,2,P,3,3)

        sh_ang, sh_ok = self.dec_shoulder(rot_s)              # (R,2,P,2,3)
        wr_ang, wr_ok = self.dec_wrist(rot_w)

        # --- 5. 组装 (R, E, P, S, W, 7) ---
        shape = (n_tgt, 2, n_psi, 2, 2)
        q_out = np.empty(shape + (7,))
        q_out[..., 0] = sh_ang[..., None, 0]      # (R,2,P,2,1) -> broadcast W
        q_out[..., 1] = sh_ang[..., None, 1]
        q_out[..., 2] = sh_ang[..., None, 2]
        q_out[..., 3] = q4[:, :, None, None, None]
        q_out[..., 4] = wr_ang[..., None, :, 0]   # (R,2,P,1,2) -> broadcast S
        q_out[..., 5] = wr_ang[..., None, :, 1]
        q_out[..., 6] = wr_ang[..., None, :, 2]

        ok = (reach_ok[:, None, None, None, None]
              & elbow_ok[:, None, None, None, None]
              & radius_ok[:, :, None, None, None]
              & src_ok[:, :, None, None, None]
              & dst_ok[:, :, :, None, None]
              & sh_ok[..., None]              # (R,E,P,S,1)
              & wr_ok[..., None, :])           # (R,E,P,1,W)
        ok = np.broadcast_to(ok, shape)

        # (R,E,P,S,W,*) -> (R,P,E,S,W,*) -> (R,P,8,*)
        q_out = np.ascontiguousarray(q_out.transpose(0, 2, 1, 3, 4, 5)).reshape(
            n_tgt, n_psi, 8, 7)
        ok = np.ascontiguousarray(ok.transpose(0, 2, 1, 3, 4)).reshape(
            n_tgt, n_psi, 8)
        in_limits = ok & model.within_limits_batch(q_out, tol=1e-9)
        return BatchGrid(q=q_out, ok=ok, in_limits=in_limits, psi=psi,
                         branch=self.BRANCH_TABLE)
