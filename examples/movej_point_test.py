#!/usr/bin/env python3
"""X2Arm 左右臂 MoveJ 示例：默认离线，--preflight 只读，--execute 才运动。

路线为 HOME → forward（肩 pitch 55°）→ HOME；可加 --converge 8 在每点
追加同目标 FK 位姿的闭环 MoveL。--side both 使用一个实例同步 MoveJ，
暂不支持双臂闭环 MoveL。MoveJ 本身不承诺 TCP 1 mm。
退出码：0 完成；1 连接/执行/测量失败；2 参数错误；130 用户中断。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="只读连接和配置/反馈检查")
    mode.add_argument("--execute", action="store_true", help="在已处于 URS 的机器人上执行所选臂点位")
    ap.add_argument("--side", choices=("left", "right", "both"), default="right",
                    help="left/right 使用显式左右入口；both 用同一实例同步发送双臂")
    ap.add_argument("--duration", type=float, default=8., help="每段 MoveJ 时长 s，至少 8")
    ap.add_argument("--settle", type=float, default=2., help="每段 MoveJ 保持 s，至少 2")
    ap.add_argument("--converge", type=int, choices=(0, 8), default=0,
                    help="8：MoveJ 后加闭环 MoveL；默认 0：只调用 MoveJ")
    return ap


def validate_args(args):
    if args.side == "both" and args.converge:
        raise ValueError("--side both 暂不支持 --converge；双臂同步 MoveL 尚未实现")
    for name, lower in (("duration", 8.), ("settle", 2.)):
        value = getattr(args, name)
        if not math.isfinite(value) or value < lower:
            raise ValueError(f"--{name} 必须是 >= {lower} 的有限数")


def bootstrap(argv):
    """只在线上分支调用，沿用项目的 ROS 解释器和消息包发现。"""
    import x2ik
    x2ik.load_conf()
    if os.environ.get("X2_MOVEJ_EXAMPLE_READY") == "1":
        return None
    python = x2ik.ros_python()
    if python is None:
        raise RuntimeError("找不到 ROS Python；请在机器人端先运行 ./x2ik.py doctor")
    env = x2ik.ros_env(x2ik.find_sim_home(), x2ik.find_msgs_prefix())
    env["X2_MOVEJ_EXAMPLE_READY"] = "1"
    return x2ik.run_with_ros([python, str(Path(__file__).resolve()), *argv], env)


def point_plan(arm, home):
    forward = home.copy()
    forward[0] = math.radians(55.)
    points = []
    for name, q in (("home", home.copy()), ("forward", forward), ("home", home.copy())):
        if not arm.model.within_limits(q):
            raise RuntimeError(f"{name} 目标超出当前模型关节限位")
        pos, rpy = arm.fk(q)
        points.append(dict(name=name, q=q, pos=pos, rpy=rpy))
    return points


def validate_movej_result(result):
    import numpy as np
    if result.get("stale") is not False:
        raise RuntimeError("MoveJ 没有确认新反馈，停止序列")
    for key in ("q", "err"):
        values = np.asarray(result.get(key), float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"MoveJ 返回的 {key} 不是 7 个有限数")
    error = result.get("err_max")
    if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
        raise RuntimeError("MoveJ 返回的 err_max 无效")


def validate_movel_result(result):
    if result.get("stale") is not False or result.get("trajectory_valid") is not True:
        raise RuntimeError("MoveL 反馈过期或轨迹未完整执行")
    if result.get("aborted") or any(result.get(key, 0) for key in
                                  ("ik_fails", "clipped", "step_rejects", "branch_rejects")):
        raise RuntimeError("MoveL 存在中止、无解、裁剪或跳变拒绝")
    error = result.get("pos_err")
    if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
        raise RuntimeError("MoveL 位置误差无效")
    if result.get("converged") is not True or error > .001:
        raise RuntimeError(f"闭环未达到 1 mm：reason={result.get('converge_reason')!r}, "
                           f"pos_err={error * 1000:.3f} mm")


def forbid_action(*args, **kwargs):
    raise RuntimeError("本示例禁止所有状态切换")


def run(args):
    from x2_api import HOME, X2Arm
    online = args.preflight or args.execute
    sides = ("left", "right") if args.side == "both" else (args.side,)
    default_side = "right" if args.side == "both" else args.side
    with ExitStack() as stack:
        # 双臂也只创建一个在线实例，左侧离线对象仅用于对应模型的 FK。
        arm = stack.enter_context(X2Arm(default_side, connect=online))
        planning_arms = {default_side: arm}
        if args.side == "both":
            planning_arms["left"] = stack.enter_context(X2Arm("left", connect=False))
        if online:
            # API 自身只读确认 URS；示例再禁止实例发出任何状态切换请求。
            arm.cli.set_action = forbid_action
            arm.cli.enter_control = forbid_action
            print("连接配置：" + json.dumps(arm.connection_config, ensure_ascii=False), flush=True)
            current = {default_side: arm.joints().tolist()}  # 此调用已验证双臂完整新反馈。
            if args.side == "both":
                current["left"] = arm.cli.q("left").tolist()
            print("当前关节 rad：" + json.dumps(current), flush=True)
        points = {side: point_plan(planning_arms[side], HOME) for side in sides}
        print(f"{args.side} 点位计划（q/RPY 为 rad，XYZ 为 m）：", flush=True)
        for index in range(3):
            for side in sides:
                point = points[side][index]
                print(json.dumps(dict(index=index + 1, side=side, name=point["name"],
                                       q=point["q"].tolist(), xyz=point["pos"].tolist(),
                                       rpy=point["rpy"].tolist()), ensure_ascii=False), flush=True)
        if not args.execute:
            print("只读预检完成，未下发运动。" if online else "离线计算完成，未连接 ROS 或机器人。", flush=True)
            return 0
        for index in range(3):
            if args.side == "both":
                results = arm.move_j_both(q_left=points["left"][index]["q"],
                                          q_right=points["right"][index]["q"],
                                          duration=args.duration, settle=args.settle)
            else:
                move_j = arm.R_move_J if args.side == "right" else arm.L_move_J
                results = {args.side: move_j(points[args.side][index]["q"],
                                             duration=args.duration, settle=args.settle)}
            for side in sides:
                try:
                    validate_movej_result(results[side])
                except (KeyError, TypeError, RuntimeError) as exc:
                    raise RuntimeError(f"{side} MoveJ 结果无效：{exc}") from exc
            for side in sides:
                point = points[side][index]
                print(f"#{index + 1} {side}/{point['name']} MoveJ："
                      f"最大关节误差 {math.degrees(results[side]['err_max']):.3f} deg", flush=True)
            if args.converge:
                point = points[args.side][index]
                result = arm.move_l(point["pos"], point["rpy"], duration=2., settle=2.,
                                    converge=args.converge, converge_tol=.001,
                                    converge_step=math.radians(.5), converge_total=math.radians(3.))
                validate_movel_result(result)
                print(f"#{index + 1} {args.side}/{point['name']} 闭环：反馈 FK 误差 "
                      f"{result['pos_err'] * 1000:.3f} mm，"
                      f"修正 {result.get('converge_iterations')} 轮", flush=True)
        print("点位序列完成。MoveJ 关节误差与闭环反馈 FK 误差按上述字段判读。", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    args = ap.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    try:
        if args.preflight or args.execute:
            status = bootstrap(argv)
            if status is not None:
                return status
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        return run(args)
    except KeyboardInterrupt:
        print("用户中断；停止序列，未执行恢复动作。", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[失败] {exc}；停止序列，未执行恢复动作。", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
