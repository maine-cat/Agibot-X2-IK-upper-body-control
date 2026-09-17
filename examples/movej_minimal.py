#!/usr/bin/env python3
"""最小业务封装示例；安装 x2ik 并加载机器人 ROS 环境后运行。

python3 movej_minimal.py             只打印目标，不连接机器人。
python3 movej_minimal.py --execute   右臂 MoveJ 到 HOME。
python3 movej_minimal.py --tcp-mode gripper   选择夹爪 TCP，仍不连接。

只在机器人已处于 URS、路径净空且其他控制入口已退出时使用 --execute。
"""
from __future__ import annotations

import argparse


def move_right_arm(robot, joints):
    """业务侧只需要这一条调用；关节角为 7 个弧度数。"""
    return robot.moveJ(joints, side="right", duration=8.0, settle=2.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="实际执行右臂 HOME 运动")
    parser.add_argument("--tcp-mode", choices=("none", "hand", "gripper", "custom"), default="none")
    parser.add_argument("--tcp-file", help="custom 模式的 TCP JSON；由运行本程序的机器读取")
    args = parser.parse_args(argv)
    if (args.tcp_mode == "custom") != bool(args.tcp_file):
        parser.error("custom 需要 --tcp-file；其他模式不接受文件")

    from x2ik import HOME, Robot

    target = HOME.copy()  # 替换为业务目标：[肩 pitch, roll, yaw, 肘, 腕 yaw, pitch, roll]
    if not args.execute:
        print("离线目标（rad）：", target.tolist())
        print("TCP：", args.tcp_mode, args.tcp_file or "")
        print("机器人上执行时保留所选参数，追加 --execute")
        return 0
    with Robot(tcp_mode=args.tcp_mode, tcp_file=args.tcp_file) as robot:
        result = move_right_arm(robot, target)
        print("结束瞬间最大关节误差（rad）：", result["err_max"])
        print("TCP 反馈位置（torso，m）：", result["position_m"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
