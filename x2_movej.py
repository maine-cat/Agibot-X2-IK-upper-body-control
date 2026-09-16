"""面向业务程序的 MoveJ 薄封装；公开入口为 ``from x2ik import Robot, HOME``。

所有运动复用 X2Arm 的限位、URS、反馈、发布独占和发送间隔保护。
连接与关闭不发送运动，也不切换运控状态。
"""
from __future__ import annotations

from typing import Optional, Sequence

from x2_api import HOME, X2Arm as _X2Arm

__all__ = ["Robot", "HOME"]


class Robot:
    """一条连接，显式选择单臂或同时控制双臂。

    机器人须已处于 URS。补偿固定为 40 N·m/rad / 12 deg / pelvis；
    robot_sn 仅记录机器身份，不读取或要求 SN 标定文件。
    方法顺序阻塞，返回后不后台保位。
    """

    def __init__(self, *, robot_sn: Optional[str] = None, verbose: bool = True):
        self._arm = _X2Arm("right", robot_sn=robot_sn, verbose=verbose,
                          _fixed_compensation=True)

    def moveJ(self, q: Optional[Sequence[float]] = None, *,
              side: Optional[str] = None,
              q_left: Optional[Sequence[float]] = None,
              q_right: Optional[Sequence[float]] = None,
              duration: float = 8.0, settle: float = 2.0) -> dict:
        """执行一次关节运动，单位 rad / s。

        单臂：moveJ(q, side="right" 或 "left")，省略 side 默认右臂。
        双臂：moveJ(q_left=..., q_right=...)，两臂共用时长与轨迹进度。
        q 不给时单臂目标为 HOME；双臂必须显式给齐两侧目标。

        目标顺序：肩 pitch / roll / yaw、肘、腕 yaw / pitch / roll。
        单臂返回 {q, err, err_max, tau, stale}；双臂返回 {left: ..., right: ...}。
        err_max 单位 rad。MoveJ 不做碰撞规划，也不包含 TCP 1 mm 闭环。
        异常直接传给调用者，不自动重试、回 HOME 或切换状态。
        """
        if q_left is not None or q_right is not None:
            if q is not None or side is not None:
                raise ValueError("双臂 q_left / q_right 不能与 q / side 混用")
            if q_left is None or q_right is None:
                raise ValueError("双臂 MoveJ 必须同时提供 q_left 和 q_right")
            return self._arm.move_j_both(q_left=q_left, q_right=q_right,
                                         duration=duration, settle=settle)
        if side not in (None, "left", "right"):
            raise ValueError("side 必须是 'left' 或 'right'；双臂使用 q_left / q_right")
        move = self._arm.L_move_J if side == "left" else self._arm.R_move_J
        return move(q, duration=duration, settle=settle)

    def close(self) -> None:
        """关闭连接；不追加运动或状态切换。"""
        self._arm.close()

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
