#!/usr/bin/env python3
"""关节轨迹 CSV 录制与跳变分析。

两件事,互相独立:

1. **录制** —— `JointRecorder` 挂到 `X2ArmClient` 上,每下发一帧就落一行
   CSV:给定位置、实际写进消息的位置(含重力偏置)、反馈位置、反馈速度、
   反馈力矩,两条臂 14 个关节全量。要 ROS 环境。

2. **分析** —— `analyze()` / `python3 x2_record.py <csv>` 只吃 CSV,
   只依赖 numpy,不需要 ROS。找出帧间跳变最大的位置并按关节排名。

分开是因为:录的时候人在真机旁边,分析的时候人在办公桌上,后者不该被
rclpy / aimdk_msgs 的环境卡住。

用法:
    # 边动边录(任意运动子命令加 --record)
    ./x2ik.py ros --record run.csv pose 0.35 -0.25 -0.10
    # 只录不动(另一个进程在动,这边纯旁听)
    ./x2ik.py ros record --seconds 20 --output run.csv
    # 事后分析
    ./x2ik.py jump run.csv
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# 关节短名。CSV 列头用短名,免得每列都拖一个 `_joint` 后缀。
JOINT_SHORT = ("sh_pitch", "sh_roll", "sh_yaw", "elbow",
               "wr_yaw", "wr_pitch", "wr_roll")
JOINT_SUFFIX = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                "elbow_joint", "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")
SIDES = ("left", "right")

#: 五组逐关节通道。q_cmd 与 q_sent 的差就是重力偏置实际生效的量 ——
#: 跳变出在 IK 还是出在偏置夹子上,全靠这两列分开看。
CHANNELS = ("q_cmd", "q_sent", "q_meas", "dq_meas", "tau_meas")

#: 判跳变的默认阈值。goto_cartesian 内部的拒绝阈值也是 0.05 rad/帧,
#: 这里取同一个数,于是"分析报出来的跳变"和"运动时被拒绝的候选"可比。
DEFAULT_JUMP_RAD = 0.05


def csv_header() -> List[str]:
    """CSV 列头。顺序固定,analyze 按名字取列,不按下标。"""
    cols = ["t", "frame", "phase", "side_active"]
    for ch in CHANNELS:
        for side in SIDES:
            for name in JOINT_SHORT:
                cols.append(f"{ch}_{side}_{name}")
    # TCP 位姿由反馈位置正解出来,方便直接看末端有没有跳。
    for side in SIDES:
        cols += [f"tcp_{side}_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")]
    return cols


class JointRecorder:
    """挂在 X2ArmClient 上的逐帧记录器。

    `X2ArmClient.send()` 每发一帧就回调一次 `tick()`。所以采样率天然等于
    控制环下发率(方案一 50 Hz / 方案二 500 Hz),不需要另开线程 ——
    另开线程反而会和 `_pace` 里的 `spin_once` 抢 GIL,把节拍搅乱。
    """

    def __init__(self, path: Path, tcp: bool = True, flush_every: int = 200):
        self.path = Path(path)
        self.tcp = tcp
        self.flush_every = int(flush_every)
        self.frame = 0
        self.phase = ""
        self.side_active = ""
        self.t0: Optional[float] = None
        self._fh = self.path.open("w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(csv_header())

    # ---- 给运动函数打标记用。哪一段是插值、哪一段是 settle,事后分析要分开看 ----
    def mark(self, phase: str, side: str = "") -> None:
        self.phase = phase
        self.side_active = side

    def tick(self, cli, q_cmd: Dict[str, np.ndarray],
             q_sent: Dict[str, np.ndarray]) -> None:
        """落一行。cli 只用来读反馈,不改它任何状态。"""
        import time
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        row: List[object] = [f"{now - self.t0:.6f}", self.frame,
                             self.phase, self.side_active]
        meas = {s: (cli.q(s), cli.dq(s), cli.tau(s)) for s in SIDES}
        for ch in CHANNELS:
            for side in SIDES:
                if ch == "q_cmd":
                    vec = q_cmd[side]
                elif ch == "q_sent":
                    vec = q_sent[side]
                elif ch == "q_meas":
                    vec = meas[side][0]
                elif ch == "dq_meas":
                    vec = meas[side][1]
                else:
                    vec = meas[side][2]
                row += [f"{float(v):.6f}" for v in vec]
        for side in SIDES:
            if self.tcp:
                pos, rot = cli.models[side].forward_kinematics(meas[side][0])
                from .x2_arm_model import matrix_to_rpy
                rpy = matrix_to_rpy(rot)
                row += [f"{float(v):.6f}" for v in (*pos, *rpy)]
            else:
                row += [""] * 6
        self._w.writerow(row)
        self.frame += 1
        if self.flush_every and self.frame % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
        print(f"[rec] {self.frame} 帧 -> {self.path}")


# --------------------------------------------------------------------------
# 分析:只吃 CSV,只要 numpy
# --------------------------------------------------------------------------

def load(path: Path) -> Dict[str, np.ndarray]:
    """读 CSV 成 {列名: 数组}。空串成 nan,字符串列原样留着。"""
    with Path(path).open(newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise SystemExit(f"{path}: 只有表头,没有数据")
    head, body = rows[0], rows[1:]
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(head):
        col = [r[i] if i < len(r) else "" for r in body]
        if name in ("phase", "side_active"):
            out[name] = np.array(col, dtype=object)
            continue
        vals = []
        for v in col:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        out[name] = np.array(vals, float)
    return out


def _matrix(data: Dict[str, np.ndarray], channel: str, side: str) -> np.ndarray:
    """把某个通道的 7 列拼成 (N, 7)。缺列就返回全 nan,不抛异常。"""
    cols = []
    n = len(next(iter(data.values())))
    for name in JOINT_SHORT:
        key = f"{channel}_{side}_{name}"
        cols.append(data.get(key, np.full(n, np.nan)))
    return np.column_stack(cols)


def analyze(path: Path, threshold_rad: float = DEFAULT_JUMP_RAD,
            top: int = 10) -> Dict[str, object]:
    """找跳变。

    判据是**帧间差分**,分别对给定位置和反馈位置各算一遍:
      - q_cmd 跳 -> 上游(IK / 插值)自己给出了跳变的指令
      - q_sent 跳但 q_cmd 不跳 -> 重力偏置在跳(偏置夹子或刚度估计的问题)
      - q_meas 跳但 q_cmd/q_sent 不跳 -> 机器人执行侧在跳(掉齿、丢步、mc 内部)
    三种成因的处置完全不同,所以必须分开报。
    """
    data = load(path)
    t = data["t"]
    dt = np.diff(t) if len(t) > 1 else np.array([np.nan])
    out: Dict[str, object] = {
        "path": str(path), "frames": len(t),
        "duration": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "dt_median": float(np.nanmedian(dt)), "dt_max": float(np.nanmax(dt)),
        "threshold_rad": threshold_rad, "channels": {},
    }
    for channel in ("q_cmd", "q_sent", "q_meas"):
        per_channel = {}
        for side in SIDES:
            mat = _matrix(data, channel, side)
            if np.all(np.isnan(mat)):
                continue
            d = np.abs(np.diff(mat, axis=0))          # (N-1, 7)
            if d.size == 0:
                continue
            worst_per_joint = np.nanmax(d, axis=0)
            frame_peak = np.nanmax(d, axis=1)
            hits = int(np.count_nonzero(frame_peak > threshold_rad))
            idx = np.argsort(-np.nan_to_num(frame_peak))[:top]
            events = []
            for i in idx:
                if not np.isfinite(frame_peak[i]) or frame_peak[i] <= threshold_rad:
                    continue
                j = int(np.nanargmax(d[i]))
                events.append(dict(
                    frame=int(i + 1), t=float(t[i + 1]),
                    joint=JOINT_SHORT[j],
                    dq_deg=float(math.degrees(frame_peak[i])),
                    dt=float(t[i + 1] - t[i]),
                    phase=str(data["phase"][i + 1]) if "phase" in data else "",
                ))
            per_channel[side] = dict(
                peak_deg=float(math.degrees(np.nanmax(frame_peak))),
                hits=hits,
                per_joint_deg={n: float(math.degrees(v))
                               for n, v in zip(JOINT_SHORT, worst_per_joint)},
                events=events,
            )
        if per_channel:
            out["channels"][channel] = per_channel
    # 力矩峰值:跳变往往伴随一记力矩尖峰,一起报出来省得再翻一遍 CSV。
    tau = {}
    for side in SIDES:
        mat = _matrix(data, "tau_meas", side)
        if np.all(np.isnan(mat)):
            continue
        tau[side] = {n: float(np.nanmax(np.abs(mat[:, i])))
                     for i, n in enumerate(JOINT_SHORT)}
    out["tau_peak"] = tau
    return out


def report(res: Dict[str, object]) -> str:
    lines = [
        f"文件      {res['path']}",
        f"帧数      {res['frames']}   时长 {res['duration']:.2f} s",
        f"帧间隔    中值 {res['dt_median'] * 1000:.2f} ms   最大 {res['dt_max'] * 1000:.2f} ms",
        f"跳变阈值  {math.degrees(res['threshold_rad']):.2f} deg/帧",
        "",
    ]
    if res["dt_max"] > 3 * res["dt_median"]:
        lines += ["⚠ 帧间隔有明显毛刺(最大 > 3x 中值)。这时候的 q_meas 跳变可能只是",
                  "  控制环落后导致的采样错位,不一定是机器人真的跳,先看 q_cmd。", ""]
    label = {"q_cmd": "给定位置(IK/插值出来的)",
             "q_sent": "实发位置(含重力偏置)",
             "q_meas": "反馈位置(机器人实际走的)"}
    for channel, per_side in res["channels"].items():
        lines.append(f"── {channel}  {label.get(channel, '')}")
        for side, info in per_side.items():
            lines.append(f"   {side:5s} 峰值 {info['peak_deg']:7.3f} deg/帧   "
                         f"超阈值 {info['hits']} 帧")
            worst = sorted(info["per_joint_deg"].items(), key=lambda kv: -kv[1])[:3]
            lines.append("         逐关节峰值 top3: " +
                         "  ".join(f"{n}={v:.3f}" for n, v in worst))
            for e in info["events"][:5]:
                lines.append(f"         [!] 帧{e['frame']:5d} t={e['t']:6.3f}s "
                             f"{e['joint']:9s} {e['dq_deg']:7.3f} deg "
                             f"(dt={e['dt'] * 1000:.1f}ms, phase={e['phase']})")
        lines.append("")
    if res.get("tau_peak"):
        lines.append("── 力矩峰值 |tau| (N·m)")
        for side, per_joint in res["tau_peak"].items():
            top = sorted(per_joint.items(), key=lambda kv: -kv[1])[:4]
            lines.append(f"   {side:5s} " + "  ".join(f"{n}={v:.2f}" for n, v in top))
        lines.append("")
    # 结论:三个通道谁在跳,处置完全不同,直接把判读写出来。
    cmd = res["channels"].get("q_cmd", {})
    sent = res["channels"].get("q_sent", {})
    meas = res["channels"].get("q_meas", {})
    def peak(d):
        return max((v["peak_deg"] for v in d.values()), default=0.0)
    thr = math.degrees(res["threshold_rad"])
    p_cmd, p_sent, p_meas = peak(cmd), peak(sent), peak(meas)
    lines.append("── 判读")
    if p_cmd > thr:
        lines.append(f"   给定位置本身在跳({p_cmd:.2f} deg/帧)。是上游 IK 或插值的问题,")
        lines.append("   与机器人无关 —— 拿这段 CSV 找解支切换的那一帧。")
    elif p_sent > thr:
        lines.append(f"   给定不跳、实发在跳({p_sent:.2f} deg/帧)。是重力偏置在跳,")
        lines.append("   查 --stiffness 估值和 --bias-limit 夹子。")
    elif p_meas > thr:
        lines.append(f"   指令是连续的、反馈在跳({p_meas:.2f} deg/帧)。指令侧清白,")
        lines.append("   问题在机器人执行侧(掉齿/丢步/mc 内部限幅),这就是要给厂家的证据。")
    else:
        lines.append(f"   三个通道都没超过 {thr:.2f} deg/帧,这一段没录到跳变。")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="分析 record 出来的关节 CSV,找跳变")
    ap.add_argument("csv", help="x2ik.py ros record / --record 生成的 CSV")
    ap.add_argument("--threshold", type=float, default=math.degrees(DEFAULT_JUMP_RAD),
                    help="跳变阈值 deg/帧(默认 2.86 = 0.05 rad,与运动时的拒绝阈值同值)")
    ap.add_argument("--top", type=int, default=10, help="每个通道列出前几个跳变点")
    ap.add_argument("--json", default=None, help="同时把结果写成 JSON")
    args = ap.parse_args(argv)
    res = analyze(Path(args.csv), math.radians(args.threshold), args.top)
    print(report(res))
    if args.json:
        import json
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                              default=float))
        print(f"[ok] JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
