#!/usr/bin/env python3
"""X2 上肢运动学 / 解析 IK 验证脚本。

跑四组测试:
  1. SRS 结构验证  —— 球肩、球腕、肘轴垂直性是否成立(解析解的前提)
  2. FK->IK->FK 往返 —— 随机可达位姿的位置/姿态复现精度
  3. 冗余零空间   —— 固定位姿扫 SEW 角,确认 psi 真的只动肘不动末端
  4. 速度级 IK    —— 有限差分校验 inverse_velocity
  5. 姿态降级     —— solve_hold_rotation:保持不住当前姿态时让单轴,位置不许更偏

用法: python3 verify_x2_arm.py [样本数]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from x2_arm_model import ArmModel, log3
from x2_srs_ik import SrsArmIK, angle_delta, branch_tuple

HOME_Q = np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])   # 与 x2_sim_ros 的待机位一致


def test_srs_structure(model: ArmModel) -> None:
    print(f"[1] SRS 结构验证 ({model.side})")
    q0 = np.zeros(7)
    positions, _ = model.joint_frames(q0)
    axes = model.axes_at(q0)
    upper = positions[3] - model.shoulder_center
    fore = positions[6] - positions[3]
    a4 = axes[3]
    print(f"    球肩三轴共点残差 : {model.shoulder_axis_residual * 1e3:9.4f} mm")
    print(f"    球腕三轴共点残差 : {model.wrist_axis_residual * 1e3:9.4f} mm")
    print(f"    肘轴 · 上臂      : {float(np.dot(a4, upper / np.linalg.norm(upper))):+.2e}")
    print(f"    肘轴 · 前臂      : {float(np.dot(a4, fore / np.linalg.norm(fore))):+.2e}")
    print(f"    上臂 {np.linalg.norm(upper) * 1e3:.2f} mm / 前臂 {np.linalg.norm(fore) * 1e3:.2f} mm")


def test_roundtrip(model: ArmModel, ik: SrsArmIK, samples: int, seed: int = 0) -> None:
    print(f"[2] FK->IK->FK 往返 ({model.side}, {samples} 个随机位姿)")
    rng = np.random.default_rng(seed)
    # 采样时留 5% 限位余量,避免采到贴着限位、数值上不可复现的边界点
    span = model.q_max - model.q_min
    lo, hi = model.q_min + 0.05 * span, model.q_max - 0.05 * span

    pos_errs, rot_errs, times = [], [], []
    failures = 0
    for _ in range(samples):
        q_true = rng.uniform(lo, hi)
        p_t, r_t = model.forward_kinematics(q_true)
        t0 = time.perf_counter()
        sol = ik.solve(p_t, r_t, q_seed=q_true, psi_samples=72)
        times.append(time.perf_counter() - t0)
        if sol is None:
            failures += 1
            continue
        pos_errs.append(sol.pos_error)
        rot_errs.append(sol.rot_error)

    if not pos_errs:
        print("    全部失败")
        return
    pos_errs = np.array(pos_errs) * 1e3        # mm
    rot_errs = np.degrees(np.array(rot_errs))  # deg
    print(f"    成功 {len(pos_errs)}/{samples} (失败 {failures})")
    print(f"    位置误差  中位 {np.median(pos_errs):.3e} mm  "
          f"95% {np.percentile(pos_errs, 95):.3e} mm  最大 {pos_errs.max():.3e} mm")
    print(f"    姿态误差  中位 {np.median(rot_errs):.3e} deg "
          f"95% {np.percentile(rot_errs, 95):.3e} deg 最大 {rot_errs.max():.3e} deg")
    print(f"    单次耗时  中位 {np.median(times) * 1e3:.2f} ms  最大 {max(times) * 1e3:.2f} ms")


def test_nullspace(model: ArmModel, ik: SrsArmIK) -> None:
    """扫 psi:末端应完全不动,肘应画出一段圆弧。这是冗余自由度可用性的直接证据。"""
    print(f"[3] SEW 冗余零空间 ({model.side})")
    q0 = ik.q_ref.copy()
    p_t, r_t = model.forward_kinematics(q0)
    kept, elbows = [], []
    for psi in np.linspace(-np.pi, np.pi, 180, endpoint=False):
        sols = ik.solve_at_psi(p_t, r_t, float(psi))
        if not sols:
            continue
        best = min(sols, key=lambda s: np.linalg.norm(s.q - q0))
        # 解析解本身有 ~1 um 残差(URDF 肘轴非严格垂直),精修后才是真实精度
        best = ik.polish(best, p_t, r_t, psi=float(psi))
        if best.pos_error > 1e-9 or not best.in_limits:
            continue
        kept.append((psi, best))
        elbows.append(model.joint_frames(best.q)[0][3])
    if not kept:
        print("    无可行解")
        return
    errs = np.array([s.pos_error for _, s in kept]) * 1e3
    elbows = np.array(elbows)
    psi_span = np.degrees(max(p for p, _ in kept) - min(p for p, _ in kept))
    print(f"    可行 psi 采样点 : {len(kept)}/180 (跨度 {psi_span:.1f} deg)")
    print(f"    末端位置漂移    : 最大 {errs.max():.3e} mm  <- 应为数值零")
    print(f"    肘点移动范围    : {np.linalg.norm(elbows.max(0) - elbows.min(0)) * 1e3:.1f} mm")


def test_velocity(model: ArmModel, ik: SrsArmIK) -> None:
    """有限差分校验:按 inverse_velocity 走一小步,实际 twist 应与期望一致。"""
    print(f"[4] 速度级 IK ({model.side})")
    q = ik.q_ref.copy() + np.array([0.1, -0.05, 0.2, 0.15, -0.1, 0.05, 0.1])
    twist = np.array([0.05, -0.03, 0.02, 0.1, -0.08, 0.06])
    psi_dot = 0.2
    dq = ik.inverse_velocity(q, twist, psi_dot)

    h = 1e-6
    p0, r0 = model.forward_kinematics(q)
    p1, r1 = model.forward_kinematics(q + dq * h)
    lin = (p1 - p0) / h
    ang = log3(r1 @ r0.T) / h
    dpsi = (ik.sew_angle(q + dq * h) - ik.sew_angle(q)) / h
    print(f"    线速度残差 : {np.linalg.norm(lin - twist[:3]):.3e} m/s")
    print(f"    角速度残差 : {np.linalg.norm(ang - twist[3:]):.3e} rad/s")
    print(f"    psi 速度残差: {abs(dpsi - psi_dot):.3e} rad/s")
    print(f"    |dq|       : {np.linalg.norm(dq):.4f} rad/s  (限速 {model.dq_max.min():.2f})")


def test_relax(model: ArmModel, ik: SrsArmIK, samples: int, seed: int = 0) -> None:
    """MDI 里只敲 x y z 时走的那条路:姿态尽量保持,保持不住就让单轴。

    这里盯三件事,任何一条破了都算回归:
      * 解出来的 q 正解回去必须落在 rel.pos / rel.rot 上 —— 报告的"实际位姿"
        不能和真解出来的对不上,否则上层算的误差全是假的;
      * q 必须在限位内;
      * **降级之后位置不许比降级前更偏**。姿态一改腕心就挪,拉回量跟着变,
        不盯着的话会出现"少让 2 deg 姿态、位置多偏 30 mm"这种赔本买卖。
    """
    print(f"[5] 姿态降级 solve_hold_rotation ({model.side})")
    rng = np.random.default_rng(seed)
    p_home, r_home = model.forward_kinematics(HOME_Q)

    n_hold = n_relax = n_none = 0
    devs, worse = [], []
    err_pos = err_rot = 0.0
    t0 = time.time()
    for _ in range(samples):
        pos = p_home + rng.uniform(-0.35, 0.35, 3)
        clip_ref = float(np.linalg.norm(ik.project_to_workspace(pos, r_home)[0] - pos))
        rel = ik.solve_hold_rotation(pos, r_home, q_seed=HOME_Q)
        if rel is None:
            n_none += 1
            continue
        fk_pos, fk_rot = model.forward_kinematics(rel.sol.q)
        err_pos = max(err_pos, float(np.linalg.norm(fk_pos - rel.pos)))
        err_rot = max(err_rot, float(np.linalg.norm(log3(rel.rot.T @ fk_rot))))
        assert model.within_limits(rel.sol.q), "降级解越限位"
        if rel.axis is None:
            n_hold += 1
            assert rel.deviation < 1e-12, "未降级却报了偏差"
        else:
            n_relax += 1
            devs.append(np.degrees(rel.deviation))
            if float(np.linalg.norm(rel.pos - pos)) > clip_ref + 1e-4:
                worse.append(pos)
    dt = time.time() - t0

    print(f"    姿态保住        : {n_hold}/{samples}")
    print(f"    降级救回        : {n_relax}/{samples}"
          + (f"  偏差 中位 {np.median(devs):.1f} deg  最大 {max(devs):.1f} deg" if devs else ""))
    print(f"    真不可达        : {n_none}/{samples}  <- 位置本身够不着,换姿态也没用")
    print(f"    降级后位置更偏  : {len(worse)}  <- 必须为 0")
    print(f"    FK 回代残差     : 位置 {err_pos * 1e3:.3e} mm  姿态 {np.degrees(err_rot):.3e} deg")
    print(f"    耗时            : {dt / samples * 1e3:.0f} ms/点")
    assert not worse, "降级把位置搞得更偏了"


def test_track_rate(model: ArmModel, ik: SrsArmIK, frames: int = 200) -> None:
    """`track()` 的逐帧耗时 —— 控制环真正用的那个调用。

    单独测它、不复用 [2] 的数字,原因是两者差一个量级、用途也完全不同:
    `solve()` 全局扫 psi(几十 ms),只在首帧和大位移重规划时调;
    `track()` 热启动(几 ms),控制环每帧都调,**它才是 50 Hz 预算的分母**。

    这一项是**迁移到算力较弱的机器(如本体 PC2 / Orin NX)时的准入门槛**:
    判据见 DEPLOY_GUIDE.md §4.5。所以要报中位、95% 和最大 —— 50 Hz 环里
    偶发的一帧超时就是手臂顿一下,只看中位数会漏掉。

    **同时把 load / 核数打出来**,因为这个数对机器负载极其敏感:同一台 20 核 x86、
    同一份代码,load≈1 时中位 4.9 ms,load≈5.6 时就变 6.9 ms(+40%)。
    不带负载记录的耗时数字是不可复现的,拿它跨机对比会得出错误结论 ——
    所以判据用**绝对阈值**(见 §3.5),这里的 x86 数字只作量级参照。
    """
    try:
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 0
        env = f"   [load {load1:.2f} / {ncpu} 核]"
    except (OSError, AttributeError):
        env = ""
    print(f"[6] track() 逐帧耗时 ({model.side}, {frames} 帧直线轨迹){env}")
    q = ik.q_ref.copy()
    p0, r0 = model.forward_kinematics(q)
    psi = None
    times, fails = [], 0
    for k in range(frames):
        # 一条 6 cm 的往复直线:幅度小到不会跳解支,大到每帧都要真解一次
        p_t = p0 + np.array([0.03 * np.sin(2 * np.pi * k / frames), 0.0, 0.0])
        t0 = time.perf_counter()
        sol = ik.track(p_t, r0, q_prev=q, psi_prev=psi)
        times.append(time.perf_counter() - t0)
        if sol is None:
            fails += 1
            continue
        q, psi = sol.q, sol.psi

    t = np.array(times) * 1e3
    budget = 1000.0 / 50.0
    print(f"    失败 {fails}/{frames}")
    print(f"    逐帧耗时  中位 {np.median(t):.2f} ms  95% {np.percentile(t, 95):.2f} ms  "
          f"最大 {t.max():.2f} ms")
    print(f"    占 50 Hz 预算({budget:.0f} ms) 中位 {np.median(t) / budget * 100:.0f}%  "
          f"最大 {t.max() / budget * 100:.0f}%")
    over = int((t > budget).sum())
    print(f"    超预算帧数 : {over}/{frames}" + ("  <- 必须为 0" if over else "  (0,合格)"))
    try:
        if os.getloadavg()[0] > 1.5:
            print("    ⚠ 本机 load 偏高,这组数偏悲观。跨机对比请在空载下重测")
    except (OSError, AttributeError):
        pass


def test_track_continuity(model: ArmModel, ik: SrsArmIK, frames: int = 80) -> None:
    """验证热启动轨迹的最短角差与解析解支连续性。"""
    q = ik.q_ref.copy()
    p0, r0 = model.forward_kinematics(q)
    psi = None
    branch = None
    max_dq = 0.0
    branch_switches = 0
    fails = 0
    previous_branch = None
    for k in range(frames):
        p = p0 + np.array([0.02 * np.sin(2 * np.pi * k / frames),
                           0.01 * (1.0 - np.cos(2 * np.pi * k / frames)), 0.0])
        sol = ik.track(p, r0, q_prev=q, psi_prev=psi,
                       branch_prev=branch, fallback=False)
        if sol is None:
            fails += 1
            continue
        dq = angle_delta(sol.q, q)
        max_dq = max(max_dq, float(np.max(np.abs(dq))))
        current_branch = branch_tuple(sol)
        if previous_branch is not None and current_branch != previous_branch:
            branch_switches += 1
        q = q + dq
        psi = sol.psi
        branch = current_branch
        previous_branch = current_branch
    print(f"[7] track() 连续性 ({model.side}, {frames} 帧)")
    print(f"    失败 {fails}/{frames}  解支切换 {branch_switches}  "
          f"最大最短步长 {np.degrees(max_dq):.3f} deg")
    if fails or branch_switches:
        print("    <- 轨迹连续性不合格")


def test_angle_wrap() -> None:
    eps = 1e-4
    a = np.array([np.pi - eps])
    b = np.array([-np.pi + eps])
    delta = angle_delta(b, a)
    assert abs(float(delta[0]) - 2.0 * eps) < 1e-8
    print("[8] ±pi 最短角差 : 合格")


def main() -> None:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    test_angle_wrap()
    print()
    for side in ("right", "left"):
        model = ArmModel(side)
        ik = SrsArmIK(model)
        print("=" * 66)
        print(f"X2 {side} arm")
        print("=" * 66)
        test_srs_structure(model)
        print()
        test_roundtrip(model, ik, samples)
        print()
        test_nullspace(model, ik)
        print()
        test_velocity(model, ik)
        print()
        # 降级搜索每点 ~0.5 s(救不回来的点要走完整条梯度),比前四组贵得多,
        # 所以只取样本数的 1/10。
        test_relax(model, ik, max(12, samples // 10))
        print()
        test_track_rate(model, ik, min(200, max(50, samples)))
        print()
        test_track_continuity(model, ik)
        print()


if __name__ == "__main__":
    main()
