#!/usr/bin/env python3
"""X2 上肢到点控制的 ROS 2 客户端与运动原语。

当前板卡流程仅使用 upper_body：在已经处于 UPPERBODY_REMOTE_SPLIT（URS）
的机器人上发点位，禁止任何状态切换及运控服务操作。配置应为
X2_ROBOT_SN 为当前机器真实序列号、X2_URS_ONLY=1；由 ./x2ik.py ros 加载 x2ik.conf。
不要把本文件保留的历史 HAL 直控、状态机或仿真功能作为当前操作指南。

先看 README.md / QUICK_START.md。标准测试与闭环参数见
CONVERGE_TEST_GUIDE.md；对外封装见 x2_api.py / API_INTERFACE.md。
部署环境见 DEPLOY_GUIDE.md；真实硬件配置不从历史仿真示例推定。

只读入口（在机器人板卡终端、运动开始前单独运行）：

    ./x2ik.py ros state                     # 关节反馈和当前 action
    ./x2ik.py ros imu                       # IMU 与重力估计
    python3 x2_converge_test.py --preflight # URS、独占发布、反馈和 IMU 预检

离线入口：

    ./x2ik.py api                           # FK / IK 自检
    python3 x2_converge_test.py --offline   # 标准测试点与路径检查

当前接口与下发顺序：
    /mc/upper_body_command，UpperBodyCommandArray，50 Hz。
    左 7 + 右 7 个关节，固定槽位见 DEFAULT_ARM_ORDER；两个实例不能独立占用
    两条臂。反馈来自 /aima/hal/joint/arm/state，按关节名读取位置、速度、力矩。
    当前 SN 配置为 40 N·m/rad / 12 deg / pelvis；它是测试起始参数，
    不代表新机器的实测静态标定，旧的 17～42 刚度表已作废。

运动原语：
    goto_joint 做关节空间五次多项式插值，没有 1 mm 闭环。
    goto_cartesian 做直线路径跟踪；converge > 0 在尾部增加关节空间闭环。
    pose / cartesian 的 --converge 默认 0；启用时等反馈仍持续发送保持输入。
    内部容差用 m、关节修正用 rad；CLI 对应参数用 mm / deg。
    标准验收使用 MoveJ 接近 + 短 MoveL 精定位，不代表任意长 MoveL 路径已验收。

口径：
    torso 系，X 前 / Y 左 / Z 上；位置 m，CLI 姿态 deg。
    RX / RY / RZ 为固定轴 X-Y-Z 欧拉角，R = Rz @ Ry @ Rx。
    --rpy 是绝对姿态，--drpy 是待机姿态加 torso 系旋转增量。
    默认 TCP 为 wrist_roll_link；手或夹爪外参需单独测量。
    pos_err 是反馈关节角经 URDF FK 的 TCP 位置误差，不含装配误差；
    真实世界 1 mm 需要外部测量。方法返回后不后台持续保位。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ⚠ 必须在 import numpy **之前**:BLAS 读线程数是在加载时一次性定的,
# 进程跑起来再改环境变量没有任何效果。
# 本项目矩阵都是 7x7 量级,多线程只增负载不增速度(实测墙钟不变、CPU 减半),
# 板卡上尤其要压。正常路径由 x2ik.py 的 ros_env() 提前设好,这里是直接
# `python3 x2_sim_ros.py` 绕过启动器时的兜底。setdefault 保证外面能覆盖。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from x2_arm_model import (ARM_JOINT_SUFFIX, ArmModel, log3, axis_angle_to_matrix,
                          rpy_to_matrix, matrix_to_rpy, rpy_deg)
from x2_arm_dynamics import ArmDynamics
from x2_srs_ik import SrsArmIK, angle_delta, branch_tuple
from x2_frames import (HOME_Q, GravityEstimator, WaistChain, lateral_raise_q,
                       quat_to_matrix, matrix_to_quat,
                       CHEST_IMU_TOPIC, PELVIS_IMU_TOPIC)

ARM_STATE_TOPIC = "/aima/hal/joint/arm/state"
ARM_COMMAND_TOPIC = "/aima/hal/joint/arm/command"
UPPER_BODY_TOPIC = "/mc/upper_body_command"
SET_ACTION_SRV = "/aimdk_5Fmsgs/srv/SetMcAction"
GET_ACTION_SRV = "/aimdk_5Fmsgs/srv/GetMcAction"
WAIST_STATE_TOPIC = "/aima/hal/joint/waist/state"

HERE = Path(__file__).resolve().parent

CALIB_DIR = Path(os.environ.get("X2IK_CALIB_DIR", str(
    Path(os.environ["X2IK_CONFIG"]).expanduser().parent / "calibration"
    if os.environ.get("X2IK_CONFIG") else HERE / "calibration")))


def _calib_path(sn: str) -> Path:
    return CALIB_DIR / f"{sn}.json"


def load_calibration(sn: Optional[str]) -> Optional[dict]:
    """按 SN 读标定文件;没设 SN、文件不存在、或解析失败都返回 None(不炸主流程)。"""
    if not sn:
        return None
    p = _calib_path(sn)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] 标定文件 {p} 读取失败: {exc},忽略", file=sys.stderr)
        return None


def cmd_calibrate_save(args) -> int:
    sn = args.sn or os.environ.get("X2_ROBOT_SN")
    if not sn:
        print("[err] 必须指定 --sn,或者在 x2ik.conf 里设置 X2_ROBOT_SN", file=sys.stderr)
        return 2
    CALIB_DIR.mkdir(exist_ok=True)
    data = {
        "sn": sn,
        "stiffness": args.stiffness,
        "bias_limit_deg": args.bias_limit,
        "gravity_source": args.gravity_source,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": args.note,
    }
    path = _calib_path(sn)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"[ok] 已写入 {path}")
    return 0


def cmd_calibrate_show(args) -> int:
    sn = args.sn or os.environ.get("X2_ROBOT_SN")
    if not sn:
        print("[err] 必须指定 --sn,或者在 x2ik.conf 里设置 X2_ROBOT_SN", file=sys.stderr)
        return 2
    calib = load_calibration(sn)
    if calib is None:
        print(f"[info] SN={sn} 没有标定文件,当前会使用内置默认值"
              f"(stiffness=mc默认40 / bias_limit=8deg / gravity_source=chest)")
        return 0
    print(json.dumps(calib, ensure_ascii=False, indent=2))
    return 0

# 可视化侧的三个约定。都在 torso 系下 —— IK 用的就是这个系,rviz 的 Fixed Frame
# 也设成它,这样 rviz 里量出来的坐标和你敲进 pose/mdi 的数字是同一套。
VIZ_ROOT_FRAME = "x2_torso"
MARKER_TOPIC = "/x2ik/markers"        # viz 发出:骨架 / TCP 文字 / 可达球 / 目标点
TARGET_TOPIC = "/x2ik/target"         # mdi、pose 发出,viz 订阅(PoseStamped)
#: mdi 里"只给位置"的指令(裸 xyz 和 d)允许的最大姿态降级量 deg。姿态在那两条
#: 里是从上一帧继承的、不是用户的诉求,所以拧不过去时让一点把位置保住;显式给了
#: 姿态的指令一律不降级。设 0 = 关掉,回到"拧不过去就报无解"。
RELAX_DEFAULT_DEG = 90.0

# **两种方案都是按顺序下发的**:每条臂 7 个关节,按固定顺序排成一段,
# 左臂 7 + 右臂 7 = 14。名字只是标签,接口消费的是顺序 ——
# 所以 arm_order 是这两条接口唯一的语义约定,写错了就是装错关节。
#
# 默认顺序取 HAL default.yaml 的 active_joint(腕部 yaw, pitch, roll)。
# 但 mc 另有配置(upper_body_external.yaml / planner_upper.yaml)把腕部写成
# yaw, roll, pitch。四份 yaml 自相矛盾,所以**上机第一件事是跑 `order` 子命令实测**,
# 不要相信任何一份 yaml,实测出来不一致就用 --arm-order 传回来。
DEFAULT_ARM_ORDER: Tuple[str, ...] = tuple(
    [f"left_{s}" for s in ARM_JOINT_SUFFIX] + [f"right_{s}" for s in ARM_JOINT_SUFFIX]
)

UPPER_RATE = 50.0        # 方案一发布频率
JOINT_RATE = 500.0       # 方案二发布频率,对齐 mc.yaml 里 arm_rt_pub 的 500
# 方案二的关节增益。官方 py_examples joint_control 示例对全部 14 个臂关节
# 统一取 kp=20 / kd=2 —— 这是"保证不出事"的保守起点,不是精度最优点。
# 本目录 MuJoCo 实测(右臂回 HOME_Q,gravity_ff=1.0,躯干直立):
#     kp=20  kd=2    角误差 1.346 deg / 末端 3.04 mm      <- 官方值
#     kp=60  kd=3    角误差 0.374 deg / 末端 1.30 mm
#     kp=120 kd=6    角误差 0.048 deg / 末端 0.26 mm
# 上真机从官方的 20/2 起步逐步往上加,不要直接上 120。
OFFICIAL_KP = 20.0
OFFICIAL_KD = 2.0

IK_RATE = 50.0           # 推理频率。IK 出 50 Hz 路点,再插值到下发频率
                         # (方案二 50->500 插 10 拍;方案一 50->50 直接发)

# mc 内部手臂 PD(upper_body_external.yaml)。方案一改不了,只作为位置偏置折算的分母。
MC_ARM_KP = 40.0


# ---------------------------------------------------------------- 轨迹小工具

def quintic(s: float) -> Tuple[float, float]:
    """五次多项式的位置比例与(归一化)速度比例。起终点速度加速度均为 0。"""
    return (10 * s**3 - 15 * s**4 + 6 * s**5,
            30 * s**2 - 60 * s**3 + 30 * s**4)


def expm3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    return np.eye(3) if th < 1e-12 else axis_angle_to_matrix(w / th, th)


# ---------------------------------------------------------------- 节点

class X2ArmClient:
    """一个节点同时承载两种接口方案。差异只在 _send_*,其余共用。"""

    def __init__(self, mode: str, arm_order: Sequence[str] = DEFAULT_ARM_ORDER,
                 gravity_ff_scale: float = 0.8, kp: float = OFFICIAL_KP, kd: float = OFFICIAL_KD,
                 joint_stiffness: Optional[np.ndarray] = None,
                 payload: float = 0.0, gravity_source: str = "chest",
                 hand_mode: int = 1, bias_limit: float = math.radians(8.0),
                 verbose: bool = True, read_only: bool = False):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               DurabilityPolicy)
        from aimdk_msgs.msg import (JointCommand, JointCommandArray, JointStateArray,
                                    MessageHeader, UpperBodyCommandArray)

        self.rclpy = rclpy
        self.mode = mode
        self.verbose = verbose
        self.arm_order = list(arm_order)
        if len(self.arm_order) != 14:
            raise ValueError("arm_order 必须是 14 项(左 7 + 右 7)")
        self.hand_mode = int(hand_mode)
        self.bias_limit = float(bias_limit)

        self.msg = dict(JointCommand=JointCommand, JointCommandArray=JointCommandArray,
                        MessageHeader=MessageHeader,
                        UpperBodyCommandArray=UpperBodyCommandArray)

        if not rclpy.ok():
            rclpy.init()
        self.node = Node("x2_arm_ik_client")

        # QoS 必须与机器人侧对齐,这一条错了是"静默收不到",不会报错:
        #   HAL 关节指令/状态 —— BEST_EFFORT / KEEP_LAST(10) / VOLATILE
        #     (官方 py_examples joint_control 示例原文:"matches the robot-side
        #      joint command/state subscribers")
        #   /mc/upper_body_command —— 官方示例用默认 QoS,即 RELIABLE / KEEP_LAST(10)
        # BEST_EFFORT 的发布者匹配不上 RELIABLE 的订阅者,所以关节状态订阅
        # 一旦写成 RELIABLE,表现就是话题存在、有发布者、但回调一次都不进。
        qos_hal = QoSProfile(depth=10,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.VOLATILE)
        qos_mc = QoSProfile(depth=10,
                            reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST)
        self.node.create_subscription(JointStateArray, ARM_STATE_TOPIC,
                                      self._on_state, qos_hal)

        # ---- 重力估计:胸腔 IMU 挂在 torso_link 上,就是 IK 基座系本身 ----
        # 注意话题名的坑:/imu/chest/state 挂在 torso_link,/imu/torso/state 挂在 pelvis。
        from sensor_msgs.msg import Imu
        from rclpy.qos import qos_profile_sensor_data
        self.grav = GravityEstimator(source=gravity_source)
        self.waist_chain = WaistChain()
        self.q_waist = np.zeros(3)
        self.imu_count = {"chest": 0, "pelvis": 0}
        self.imu_has_orientation = {"chest": None, "pelvis": None}
        if gravity_source != "static":
            self.node.create_subscription(Imu, CHEST_IMU_TOPIC,
                                          lambda m: self._on_imu("chest", m),
                                          qos_profile_sensor_data)
            self.node.create_subscription(Imu, PELVIS_IMU_TOPIC,
                                          lambda m: self._on_imu("pelvis", m),
                                          qos_profile_sensor_data)
            self.node.create_subscription(JointStateArray, WAIST_STATE_TOPIC,
                                          self._on_waist, qos_hal)

        self._command_type, self._command_qos = UpperBodyCommandArray, qos_mc
        self.pub = None
        if read_only and mode == "upper_body":
            self.rate = UPPER_RATE
        elif mode == "upper_body":
            self.pub = self.node.create_publisher(UpperBodyCommandArray,
                                                  UPPER_BODY_TOPIC, qos_mc)
            self.rate = UPPER_RATE
        elif mode == "joint_direct":
            self.pub = self.node.create_publisher(JointCommandArray,
                                                  ARM_COMMAND_TOPIC, qos_hal)
            self.rate = JOINT_RATE
        else:
            raise ValueError("mode 必须是 upper_body 或 joint_direct")

        self.seq = 0
        self.state_pos: Dict[str, float] = {}
        self.state_vel: Dict[str, float] = {}
        self.state_eff: Dict[str, float] = {}
        self.state_names: List[str] = []
        self.state_count = 0
        self._imu_last: Dict[str, tuple] = {}

        self.models = {s: ArmModel(s) for s in ("left", "right")}
        self.iks = {s: SrsArmIK(self.models[s]) for s in ("left", "right")}
        self.dyns = {s: ArmDynamics(self.models[s], payload_mass=payload)
                     for s in ("left", "right")}
        self.gravity_ff_scale = gravity_ff_scale
        self.kp, self.kd = kp, kd
        self.k_eff = (np.full(7, MC_ARM_KP) if joint_stiffness is None
                      else np.asarray(joint_stiffness, float))
        self.head_pos = [0.0, 0.0]
        # 录制钩子。挂上以后每 send 一帧落一行 CSV,见 x2_record.JointRecorder。
        # 默认 None = 不录,零开销。
        self.recorder = None
        self._last_sent: Dict[str, np.ndarray] = {"left": np.zeros(7),
                                                  "right": np.zeros(7)}

    # ---- 反馈 ----
    def _on_state(self, msg) -> None:
        names = []
        for j in msg.joints:
            names.append(j.name)
            self.state_pos[j.name] = float(j.position)
            self.state_vel[j.name] = float(j.velocity)
            self.state_eff[j.name] = float(j.effort)
        self.state_names = names
        self.state_count += 1

    def _on_imu(self, which: str, msg) -> None:
        self.imu_count[which] += 1
        q = msg.orientation
        a = msg.linear_acceleration
        # ROS 约定:orientation_covariance[0] == -1 表示该 IMU 不提供姿态,
        # 这时只能退到加速度计(静止时可信,运动时会被真实加速度污染)。
        has_ori = float(msg.orientation_covariance[0]) != -1.0
        norm2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if norm2 < 1e-9:            # 全零四元数 = 该 IMU 其实没接上
            has_ori = False
        self.imu_has_orientation[which] = has_ori
        self._imu_last[which] = ((q.x, q.y, q.z, q.w) if has_ori else None,
                                 (a.x, a.y, a.z))

    def _on_waist(self, msg) -> None:
        pos = {j.name: float(j.position) for j in msg.joints}
        self.q_waist = np.array([pos.get(n, 0.0) for n in
                                 ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")])

    def refresh_gravity(self) -> np.ndarray:
        """把最新 IMU 折成 torso 系重力,推给两条臂的动力学模型。

        每次下发前调一次。source=chest 时不经过任何关节,腰部编码器误差和
        腰部传动间隙都不进来;source=pelvis 时要多串一次腰链。
        """
        src = self.grav.source
        if src == "chest":
            frame = self._imu_last.get("chest")
            if frame is not None:
                self.grav.update(quat_xyzw=frame[0], accel=frame[1])
        elif src == "pelvis":
            frame = self._imu_last.get("pelvis")
            if frame is not None and frame[0] is not None:
                from x2_frames import quat_to_matrix
                rot_wp = quat_to_matrix(*frame[0])
                self.grav.update(g_direct=self.grav.from_waist(
                    rot_wp, self.q_waist, self.waist_chain))
        g = self.grav.g
        for dyn in self.dyns.values():
            dyn.gravity = g
        return g

    def spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.001)

    def wait_state(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.state_count > 0:
                return True
        return False

    def fresh_state(self, timeout: float = 0.5) -> bool:
        """等一帧**新的**反馈再测量;拿不到就返回 False。

        测量前必须过这一道:只要控制环有过一次落后于实时,q() 就可能是动作中途的
        旧值,于是"手臂没到位"这个结论完全是假的(详见 _pace 的注释)。
        用 state_count 单调递增来判断"新",不看时间戳 —— 仿真的 header.stamp
        未必可靠,而计数是本进程自己数的,不会骗人。
        """
        n0 = self.state_count
        end = time.time() + timeout
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)
            if self.state_count > n0:
                return True
        return False

    def q(self, side: str) -> np.ndarray:
        """按**关节名**取反馈,绝不按下标 —— 反馈里名字是齐的,没有歧义。"""
        return np.array([self.state_pos.get(f"{side}_{s}", 0.0) for s in ARM_JOINT_SUFFIX])

    def dq(self, side: str) -> np.ndarray:
        return np.array([self.state_vel.get(f"{side}_{s}", 0.0) for s in ARM_JOINT_SUFFIX])

    def tau(self, side: str) -> np.ndarray:
        return np.array([self.state_eff.get(f"{side}_{s}", 0.0) for s in ARM_JOINT_SUFFIX])

    # ---- 下发 ----
    def send(self, q_left: np.ndarray, q_right: np.ndarray,
             dq_left: Optional[np.ndarray] = None,
             dq_right: Optional[np.ndarray] = None) -> None:
        if os.environ.get("X2_URS_ONLY") == "1" and self.mode != "upper_body":
            raise RuntimeError("X2_URS_ONLY=1：只允许 upper_body 点位指令")
        if self.mode == "upper_body":
            self._send_upper(q_left, q_right)
        else:
            self._send_joint(q_left, q_right, dq_left, dq_right)
        # 录制放在发布之后:_send_* 会把加过重力偏置、夹过限位的最终值写进
        # _last_sent,这时 q_cmd 与 q_sent 才是同一帧的一对。
        if self.recorder is not None:
            self.recorder.tick(self,
                               {"left": np.asarray(q_left, float),
                                "right": np.asarray(q_right, float)},
                               self._last_sent)

    def _header(self, frame_id: str):
        h = self.msg["MessageHeader"]()
        h.stamp = self.node.get_clock().now().to_msg()
        h.frame_id = frame_id
        h.sequence = self.seq
        self.seq += 1
        return h

    def _send_upper(self, q_left: np.ndarray, q_right: np.ndarray) -> None:
        """方案一。只有位置通道,重力用 position_bias 折算进给定位置。"""
        m = self.msg["UpperBodyCommandArray"]()
        m.header = self._header("mc_upper_body")
        m.source = "remote_teleop_pc"
        # ⚠ hand_sub_mode 决定 mc 读哪一段,不只是"手怎么动":
        #   0 (NONE)  -> mc 只读 head_pos,**arm_pos 整段丢弃**(配 HEAD_ONLY 用)
        #   1/2/3     -> mc 只读 arm_pos + hand_pos,head_pos 丢弃(配 URS 用)
        # 实测(生态课程 SDK, URS): 填 0 时 /aima/hal/joint/arm/command 恒为 0;
        # 填 1 时立刻变成我们给的值。所以想动手臂就**不能**填 0。
        m.hand_sub_mode = self.hand_mode
        m.head_pos = list(self.head_pos)
        self.refresh_gravity()
        by_name = {}
        for side, q in (("left", q_left), ("right", q_right)):
            q = self.models[side].clamp(np.asarray(q, float))
            # limit 是保守夹子,不是物理量:模型错了它兜住;但姿态真的需要大偏置时
            # 它会**静默吃掉**差值。实测直臂侧平举 shoulder_roll 需要 7.65/40 = 11.0 deg,
            # 8 deg 的夹子刚好吃掉 3.0 deg —— 就是仿真里看到的那 3 deg 下垂。
            bias = self.dyns[side].position_bias(q, self.k_eff, limit=self.bias_limit)
            # --bias-limit 0 是"纯位置控制、完全不补重力"的基线工况(真机第一测要跑它),
            # 这时 bias 恒为 0,除数也是 0 —— 不加这个判就是 ZeroDivisionError。
            if self.bias_limit > 1e-12:
                self.bias_clipped = max(getattr(self, "bias_clipped", 0.0),
                                        float(np.max(np.abs(bias))) / self.bias_limit)
            q = self.models[side].clamp(q + bias)
            self._last_sent[side] = q.copy()
            for s, v in zip(ARM_JOINT_SUFFIX, q):
                by_name[f"{side}_{s}"] = float(v)
        # 按 arm_order 装配 —— 接口消费的是顺序,不是名字,歧义全集中在这一行
        m.arm_pos = [by_name[n] for n in self.arm_order]
        # hand_pos 长度由 hand_sub_mode 决定;本脚本不控手,一律给"保持不动"的值。
        m.hand_pos = {1: [0.0, 0.0],            # 夹爪: 左右张合度,0=闭
                      2: [0.0] * 20,            # 灵巧手: 左右各 10 关节
                      3: [0.0, 0.0, 0.0, 0.0],  # 手势: [左ID,左张合,右ID,右张合]
                      }.get(self.hand_mode, [])
        self.pub.publish(m)

    def _send_joint(self, q_left: np.ndarray, q_right: np.ndarray,
                    dq_left: Optional[np.ndarray], dq_right: Optional[np.ndarray]) -> None:
        """方案二。MIT 式:tau = stiffness(q_d-q) + damping(dq_d-dq) + effort。

        effort 里放重力前馈。stiffness/damping 由本脚本给,不再受 mc 的 40/2 约束。
        """
        JointCommand = self.msg["JointCommand"]
        m = self.msg["JointCommandArray"]()
        m.header = self._header("x2_arm_ik")
        self.refresh_gravity()
        # 先按名字算好每个关节的五个通道,再**按 arm_order 排成数组** ——
        # 和方案一用的是同一份顺序约定,`order` 子命令测出来的结论两边通用。
        # jc.name 照填,但只当标签用,不指望下游按它寻址。
        by_name: Dict[str, tuple] = {}
        for side, q_d, dq_d in (("left", q_left, dq_left), ("right", q_right, dq_right)):
            model = self.models[side]
            q_d = model.clamp(np.asarray(q_d, float))
            dq_d = np.zeros(7) if dq_d is None else np.asarray(dq_d, float)
            ff = self.gravity_ff_scale * self.dyns[side].gravity_torque(q_d)
            ff = np.clip(ff, -model.tau_max_safe, model.tau_max_safe)
            # 方案二没有位置偏置,q_sent 就是夹过限位的 q_d 本身。
            self._last_sent[side] = q_d.copy()
            for i, s in enumerate(ARM_JOINT_SUFFIX):
                by_name[f"{side}_{s}"] = (
                    float(q_d[i]),
                    float(np.clip(dq_d[i], -model.dq_max[i], model.dq_max[i])),
                    float(ff[i]))
        joints = []
        for name in self.arm_order:
            pos, vel, eff = by_name[name]
            jc = JointCommand()
            jc.name = name
            jc.position = pos
            jc.velocity = vel
            jc.effort = eff
            jc.stiffness = float(self.kp)
            jc.damping = float(self.kd)
            joints.append(jc)
        m.joints = joints
        self.pub.publish(m)

    # ---- mc 状态机 ----
    def set_action(self, action: str, timeout: float = 3.0) -> bool:
        if os.environ.get("X2_URS_ONLY") == "1":
            self._log(f"[err] X2_URS_ONLY=1：禁止所有状态切换，拒绝 {action}")
            return False
        from aimdk_msgs.srv import SetMcAction
        from aimdk_msgs.msg import RequestHeader, CommonState, McActionCommand
        cli = self.node.create_client(SetMcAction, SET_ACTION_SRV)
        if not cli.wait_for_service(timeout_sec=timeout):
            self._log(f"[err] 服务 {SET_ACTION_SRV} 不可用")
            return False
        req = SetMcAction.Request()
        req.header = RequestHeader()
        req.source = "node"
        cmd = McActionCommand()
        cmd.action_desc = action
        req.command = cmd
        fut = None
        for _ in range(8):        # 官方 example 的重试逻辑,remote peer 首帧常丢
            req.header.stamp = self.node.get_clock().now().to_msg()
            fut = cli.call_async(req)
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=0.25)
            if fut.done():
                break
        resp = fut.result() if fut is not None else None
        if resp is None:
            self._log(f"[err] SetMcAction({action}) 超时")
            return False
        ok = resp.response.status.value == CommonState.SUCCESS
        if ok:
            self._log(f"SetMcAction({action}) -> OK")
        else:
            # ⚠ mc 的拒绝原因在 header.code 里,**不在** message 里 —— message 官方
            #   不保证填,实测常是空串。只打 message 的话,"摔倒进了安全保护(6)"、
            #   "姿态不允许(4)"、"当前状态无可达路径(10)"会显示成同一句空话,而这三种
            #   正是站立首测最可能撞上的。官方 py_examples/set_mc_action.py 自己也打 code。
            code = None
            try:
                code = int(resp.response.header.code)
            except Exception:
                pass
            why = MC_ACTION_REJECT.get(code)
            msg = (resp.response.message or "").strip()
            parts = [f"SetMcAction({action}) -> 被拒"]
            if code is not None:
                parts.append(f"code={code}" + (f"({why})" if why else ""))
            if msg:
                parts.append(f"message={msg!r}")
            self._log("  ".join(parts))
        return ok

    def get_action(self, timeout: float = 3.0) -> Optional[str]:
        """读 mc 当前 action。

        注意请求体和 SetMcAction 不是一个形状:SetMcAction.Request 直接带
        `header`,而 GetMcAction.Request 只有一个 `request`(CommonRequest),
        header 套在里面,也没有 `source` 字段。写错了是 AttributeError,不是超时。
        """
        try:
            from aimdk_msgs.srv import GetMcAction
        except ImportError:
            return None
        cli = self.node.create_client(GetMcAction, GET_ACTION_SRV)
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        req = GetMcAction.Request()
        stamp = self.node.get_clock().now().to_msg()
        if hasattr(req, "request"):
            req.request.header.stamp = stamp
        elif hasattr(req, "header"):
            req.header.stamp = stamp
        fut = cli.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        resp = fut.result()
        if resp is None:
            return None
        # 响应是 McActionInfo:current_action(枚举) + action_desc(字符串) + status
        info = getattr(resp, "info", None)
        if info is None:
            for attr in ("action", "current_action", "state"):
                info = getattr(resp, attr, None)
                if info is not None:
                    break
        if info is None:
            return str(resp)
        desc = getattr(info, "action_desc", "")
        if desc:
            return desc
        cur = getattr(info, "current_action", None)
        return self._action_name(getattr(cur, "value", cur)) if cur is not None else str(info)

    @staticmethod
    def _action_name(value) -> str:
        """枚举值反查名字 —— action_desc 为空时(mc 不一定填)至少给个可读结果。"""
        try:
            from aimdk_msgs.msg import McAction
        except ImportError:
            return str(value)
        for k in dir(McAction):
            if k.isupper() and getattr(McAction, k, None) == value:
                return f"{k}({value})"
        return str(value)

    # 方案一想进的目标 action,按优先级试。旧 SDK 是 UPPERBODY_REMOTE_SPLIT,
    # 1.1 改成了 WHOLE_BODY_TELEOP —— 具体哪个存在由 action_ruler.yaml 说话。
    UPPER_TARGETS = ("UPPERBODY_REMOTE_SPLIT", "WHOLE_BODY_TELEOP",
                     "VR_REMOTE_CONTROLLER")

    def enter_control(self) -> bool:
        """把 mc 带到能接受外部上肢指令的状态。

        跳转序列不写死:从 action_ruler.yaml 里按当前 action 现算一条最短路径。
        写死 `JD -> SD -> US` 只在旧 SDK 上成立,1.1 里 US 根本不存在,
        表现是前两步 OK、第三步 `can not find action`,而且机器人已经站起来了。
        """
        if self.mode != "upper_body":
            return True

        # 先确认 mc 真的在听这条话题,早点在这里断掉,比把机器人推到站立再失败要好。
        # 注意:这条话题的订阅者在另一个节点(mc_ros2_node...)上,是独立于
        # /aima 状态话题的 DDS 端点发现过程,wait_state() 只保证状态话题已发现,
        # 不保证这条也发现完了 —— 一次性判断 0 会有假阴性,这里改成带超时的重试。
        deadline = time.time() + 3.0
        while self.node.count_subscribers(UPPER_BODY_TOPIC) == 0 and time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.node.count_subscribers(UPPER_BODY_TOPIC) == 0:
            self._log(f"[err] 3s 内没有节点订阅 {UPPER_BODY_TOPIC}。")
            self._log("      可能是本 SDK 确实不支持方案一(上肢遥操走 /aima/mc/joint/retargeting),")
            self._log("      也可能是发现还没完成(跨主机 DDS 有时更慢)——先用")
            self._log("      `ros2 topic info /mc/upper_body_command --verbose` 手动确认一次。")
            self._log("      确认没有订阅者再改用 --mode joint_direct(按关节名下发)。")
            return False

        if os.environ.get("X2_URS_ONLY") == "1":
            current = self.get_action()
            name = (current or "").split("(", 1)[0].strip()
            if name not in ("UPPERBODY_REMOTE_SPLIT", "URS", "US"):
                self._log(f"[err] 仅允许在现有 URS 下做点位，当前 {current!r}；不切换状态")
                return False
            self._log(f"[info] 已确认 {current}，X2_URS_ONLY=1，保持当前状态")
            return True

        graph, always = action_graph()
        # ⚠ "读不到" != "处在 PASSIVE_DEFAULT"。历史代码是
        #   `cur = self.get_action() or "PASSIVE_DEFAULT"`,于是服务超时、消息包缺失
        #   都会被打印成"当前 PASSIVE_DEFAULT" —— 那是**卸力态**的名字,首次上真机
        #   看到这行足以让人以为机器人要摔,而真实状态其实只是未知。更糟的是有
        #   action_ruler.yaml 时,它会拿这个编造的起点去算最短路 —— 从错误起点算出
        #   的路径"图上通、物理上摔一次"。
        #   读不到就如实说读不到:强制重进(安全方向,最多多切一次 SD->URS),
        #   有状态图时则直接拒绝执行、交给人。
        cur = self.get_action()
        if cur is None:
            self._log("[warn] 读不到 mc 当前 action(服务超时/未响应/消息包缺失)。")
            self._log("      真实状态未知,不做任何假设 —— 不会按某个猜测的起点算跳转路径。")
            cur = "<未知>"

        # ⚠ "action 名字是 URS" != "上肢分体遥操真的 armed"。
        #   mc 摔倒过、或上一次跑完进程直接退出,action 名会**留在** URS,
        #   但内部退化成 idle 托管:话题照订、消息照收、arm_pos 一律不执行。
        #   症状就是本命令全程无报错,末了两条臂的稳态误差 = 整个目标角度。
        #   见 DEPLOY_GUIDE.md §13.4「action 名是 URS 但手臂不动」。
        #   所以默认**每次都重新走一遍**进入序列(哪怕已经在 URS)—— 这也是
        #   9-14 合并之前一直在用、现场验证过能动的行为。
        #   ⚠ 中转站只能是 STAND_DEFAULT(REARM_VIA),全程站立、不卸力。
        #   不想重走的场合用 --no-reenter 跳过。
        reenter = getattr(self, "reenter", True)
        if cur in self.UPPER_TARGETS and not reenter:
            seq = []
        elif not graph:
            # 没有 X2_MC_HOME/action_ruler.yaml 时的兜底路径。
            # ⚠ 这里**绝不能**带 JOINT_DEFAULT:JD 会把关节电机全部卸力,站着的
            #   机器人直接摔。历史版本写的是 JD -> SD -> URS,那是错的,已删。
            #   只中转站立态;mc 如果不允许从当前态直接进 SD,set_action 会报错停住,
            #   由人自己决定怎么走 —— 比脚本擅自穿过卸力态安全得多。
            seq = [REARM_VIA, self.UPPER_TARGETS[0]]
        else:
            # 已经在目标态又要强制 re-arm:光 path(URS -> URS) 是空的,
            # 得先退到 STAND_DEFAULT 再回来,否则等于没重进。
            start, prefix = cur, []
            if cur in self.UPPER_TARGETS:
                back = action_path(graph, always, cur, REARM_VIA)
                if back:
                    start, prefix = REARM_VIA, back
            seq = None
            for goal in self.UPPER_TARGETS:
                if goal not in graph and goal not in always:
                    continue
                seq = action_path(graph, always, start, goal)
                if seq is not None:
                    break
            if seq is None:
                have = "  ".join(sorted(set(graph) | set(always)))
                self._log(f"[err] 从 {cur} 走不到任何上肢遥操 action。本 SDK 可选:{have}")
                return False
            seq = prefix + seq

        # 最后一道闸:不管上面怎么算的,自动序列里出现卸力/阻尼态就直接拒绝执行。
        # action_ruler.yaml 的最短路有可能穿过 DD/JD —— 图上通、物理上是摔一次。
        bad = [a for a in seq if a in UNSAFE_ACTIONS]
        if bad:
            self._log(f"[err] 自动跳转序列里出现卸力/阻尼态 {bad},已拒绝执行。")
            self._log(f"      完整序列: {cur} -> " + " -> ".join(seq))
            self._log("      这类 action 会让关节卸力、机器人失稳,不能由脚本自动穿过。")
            self._log("      请人工确认现场安全(有吊架/有人扶)后,用 `ros action <名字>` 逐跳走。")
            return False
        if cur in self.UPPER_TARGETS and seq:
            self._log(f"[info] 已在 {cur},仍经 {REARM_VIA} 重走一遍以确保 mc 真的 armed"
                      "(全程保持站立、不卸力;不想重走加 --no-reenter)。")
        self._log(f"状态机路径: {cur} -> " + " -> ".join(seq) if seq else f"已在 {cur}")
        for act in seq:
            if not self.set_action(act):
                return False
            self.spin(1.5)
        return True

    def check_contention(self) -> None:
        """查同一条指令话题上有没有别的发布者 —— 两种方案都要查。

        方案一:官方示例 upper_body_control 的 arm_pos 全 0,没停掉就会把手臂按回零位。
        方案二:mc 自己也在 500 Hz 发 /aima/hal/joint/arm/command。

        mc.yaml 里 arm_rt_pub 是无条件发布的,和外部发布者会在同一个话题上打架 ——
        表现为手臂高频抖动或指令时灵时不灵。这里只做检测和告警,
        真要独占这条话题需要停掉 mc(或让 mc 停在不出臂指令的 action),
        具体以现场实测为准,不要凭配置文件推断。
        """
        if self.mode == "upper_body":
            # 方案一同样会被抢。最常见的肇事者是官方 py_examples/examples 的
            # upper_body_control —— 它的 arm_pos 是**14 个 0**,还在后台跑的话
            # 会以 50 Hz 把手臂一直按回零位,和我们的指令交替生效。
            # 症状正好是"全程不报错、两条臂都停在 ~0、稳态误差 = 整个目标角度"。
            n = self.node.count_publishers(UPPER_BODY_TOPIC)
            self._log(f"{UPPER_BODY_TOPIC} 上的发布者数量 = {n}(含本节点)")
            if n > 1:
                self._log("[warn] 还有别的节点在发 /mc/upper_body_command!")
                self._log("[warn] 两个发布者交替生效 —— 如果对方是官方示例 "
                          "upper_body_control(arm_pos 全 0),")
                self._log("[warn] 手臂会被一直按在零位,表现成'指令发出去了但一动不动'。")
                self._log("[warn] 先 `ros2 node list | grep upper_body` 找出来停掉。")
            return
        if self.mode != "joint_direct":
            return
        n = self.node.count_publishers(ARM_COMMAND_TOPIC)
        self._log(f"{ARM_COMMAND_TOPIC} 上的发布者数量 = {n}(含本节点)")
        if n > 1:
            self._log("[warn] 还有别的节点在发同一条话题(几乎肯定是 mc 的 arm_rt_pub)。")
            self._log("[warn] 两个发布者会交替生效,手臂会抖。先确认 mc 是否已停止发臂指令。")

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def enable_commands(self):
        """Explicitly acquire the command endpoint for a desktop MDI session."""
        if self.mode != "upper_body":
            raise RuntimeError("Desktop MDI requires upper_body")
        if self.pub is None:
            self.pub = self.node.create_publisher(self._command_type, UPPER_BODY_TOPIC,
                                                  self._command_qos)

    def disable_commands(self):
        if self.pub is not None:
            self.node.destroy_publisher(self.pub)
            self.pub = None

    def close(self) -> None:
        self.node.destroy_node()


# ---------------------------------------------------------------- 动作原语

def goto_joint(cli: X2ArmClient, q_target: Dict[str, np.ndarray],
               duration: float = 3.0, settle: float = 1.5) -> Dict[str, Dict]:
    """两臂同时用五次多项式插值走到目标关节角。"""
    dt = 1.0 / cli.rate
    q0 = {s: cli.q(s) for s in ("left", "right")}
    qf = {s: cli.models[s].clamp(np.asarray(q_target[s], float)) for s in ("left", "right")}
    n = max(1, int(round(duration / dt)))
    if cli.recorder is not None:
        cli.recorder.mark("movej", "both")
    t0 = time.time()
    for i in range(n):
        a, ad = quintic((i + 1) / n)
        ql = q0["left"] + (qf["left"] - q0["left"]) * a
        qr = q0["right"] + (qf["right"] - q0["right"]) * a
        cli.send(ql, qr,
                 (qf["left"] - q0["left"]) * ad / duration,
                 (qf["right"] - q0["right"]) * ad / duration)
        _pace(cli, t0, (i + 1) * dt)
    t1 = time.time()
    if cli.recorder is not None:
        cli.recorder.mark("movej_settle", "both")
    for i in range(int(round(settle / dt))):
        cli.send(qf["left"], qf["right"])
        _pace(cli, t1, (i + 1) * dt)
    out = {}
    for s in ("left", "right"):
        q = cli.q(s)
        out[s] = dict(q=q, err=q - qf[s], err_max=float(np.max(np.abs(q - qf[s]))),
                      tau=cli.tau(s))
    return out


def goto_cartesian(cli: X2ArmClient, side: str, pos: np.ndarray, rot: np.ndarray,
                   duration: float = 3.0, settle: float = 2.0, *,
                   converge: int = 0, converge_tol: float = 0.001,
                   converge_step: float = math.radians(0.5),
                   converge_total: float = math.radians(3.0)) -> Dict:
    """单臂笛卡尔直线插值 + 连续解支跟踪。

    IK 候选先经过解支/单帧步长检查，异常候选被拒绝并保持上一条已接受
    指令；绝不把静默限幅后的跳变伪装成正常轨迹。
    converge > 0 时额外做关节空间闭环；容差用 m，修正上限用 rad。
    q_cmd 始终表示原 IK 目标，q_hold 才是加过闭环修正的 send() 输入。
    """
    _validate_converge(converge, converge_tol, converge_step, converge_total)
    if converge and (not math.isfinite(duration) or duration <= 0
                     or not math.isfinite(settle) or settle < 0):
        raise ValueError("闭环要求 duration 为有限正数，settle 为有限非负数")
    if converge and (not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot))):
        raise ValueError("闭环目标位置/姿态必须全部为有限数")
    if converge and not _complete_arm_feedback(cli):
        raise ValueError("闭环要求最新反馈帧包含完整双臂 14 关节且角度全部有限；未下发运动")
    dt = 1.0 / cli.rate
    sub = max(1, int(round(cli.rate / IK_RATE)))
    model, ik = cli.models[side], cli.iks[side]
    other = "left" if side == "right" else "right"
    q_other = cli.q(other).copy()

    q_d = cli.q(side).copy()
    p0, r0 = model.forward_kinematics(q_d)
    pos = np.asarray(pos, float)
    rot = np.asarray(rot, float)
    dr = log3(r0.T @ rot)
    psi = ik.sew_angle(q_d)
    q_prev = q_d.copy()
    branch_prev = None

    n_ik = max(1, int(round(duration * IK_RATE)))
    # 与 ArmController 的安全层保持同一量级；这里是拒绝阈值，不是限幅值。
    max_joint_step_rad = 0.05
    max_consecutive_rejects = 5
    fails = clipped = 0
    step_violations = step_rejects = branch_rejects = 0
    peak_raw_dq = peak_accepted_dq = 0.0
    consecutive_rejects = 0
    trajectory_valid = True
    aborted = False
    if cli.recorder is not None:
        cli.recorder.mark("movel", side)
    t0 = time.time()
    frame = 0
    for k in range(n_ik):
        a, _ = quintic((k + 1) / n_ik)
        p_t = p0 + (pos - p0) * a
        r_t = r0 @ expm3(dr * a)
        p_c, was_clipped = ik.project_to_workspace(p_t, r_t)
        clipped += int(was_clipped)
        sol = ik.track(p_c, r_t, q_prev=q_d, psi_prev=psi,
                       branch_prev=branch_prev, fallback=False)
        if sol is None:
            fails += 1
            if branch_prev is not None:
                branch_rejects += 1
            consecutive_rejects += 1
        else:
            # 先展开到最接近上一条已接受指令的等价角，再做步长检查。
            dq_raw = angle_delta(sol.q, q_d)
            peak = float(np.max(np.abs(dq_raw)))
            peak_raw_dq = max(peak_raw_dq, peak)
            if peak > max_joint_step_rad:
                step_violations += 1
                step_rejects += 1
                consecutive_rejects += 1
                print(f"  [!] {side} pose IK 跳变候选已拒绝: "
                      f"第{k + 1}/{n_ik}帧 max|dq|={math.degrees(peak):.2f} deg")
            else:
                q_d = q_d + dq_raw
                peak_accepted_dq = max(peak_accepted_dq, peak)
                psi = sol.psi
                branch_prev = branch_tuple(sol)
                consecutive_rejects = 0
        if consecutive_rejects >= max_consecutive_rejects:
            trajectory_valid = False
            aborted = True
            print(f"  [!] {side} pose 连续 {consecutive_rejects} 帧无法接受 IK 解，停止推进")
            break
        dq_d = (q_d - q_prev) * IK_RATE
        for i in range(sub):
            w = (i + 1) / sub
            q_i = q_prev + (q_d - q_prev) * w
            args = ((q_i, q_other, dq_d, np.zeros(7)) if side == "left"
                    else (q_other, q_i, np.zeros(7), dq_d))
            cli.send(*args)
            frame += 1
            _pace(cli, t0, frame * dt)
        q_prev = q_d.copy()

    t1 = time.time()
    if cli.recorder is not None:
        cli.recorder.mark("movel_settle", side)
    if not converge:
        # 默认路径逐条保留：发送数量、节拍、fresh_state 和返回字段不变。
        for i in range(int(round(settle / dt))):
            args = ((q_d, q_other, None, None) if side == "left" else (q_other, q_d, None, None))
            cli.send(*args)
            _pace(cli, t1, (i + 1) * dt)

        fresh = cli.fresh_state()
        q_meas = cli.q(side)
        p_meas, r_meas = model.forward_kinematics(q_meas)
    else:
        q_hold = q_d.copy()
        hist, state_counts = [], []
        iterations = step_clips = 0
        for it in range(converge + 1):
            for _ in range(int(round(settle / dt))):
                # 每拍从发送时刻计时，等待反馈/计算耗时不变成下一轮瞬时补发。
                sent_at = time.time()
                args = ((q_hold, q_other, None, None) if side == "left"
                        else (q_other, q_hold, None, None))
                cli.send(*args)
                _pace(cli, sent_at, dt)
            fresh = _fresh_hold(cli, side, q_hold, q_other)
            q_meas = cli.q(side)
            p_meas, r_meas = model.forward_kinematics(q_meas)
            pe = float(np.linalg.norm(p_meas - pos))
            if not fresh:
                reason = "stale_feedback" if _complete_arm_feedback(cli) else "invalid_feedback"
                break                 # 旧值可用于诊断，绝不能用于修正/判定达标。
            if (not np.all(np.isfinite(q_meas)) or not math.isfinite(pe)
                    or not np.all(np.isfinite(r_meas))):
                reason = "invalid_feedback"
                break
            hist.append(pe)
            state_counts.append(cli.state_count)
            if (not trajectory_valid or fails or clipped or step_rejects
                    or branch_rejects):
                reason = "invalid_trajectory"
                break
            if len(hist) >= 2 and pe > hist[-2] * 1.5:
                reason = "diverging"
                break
            if pe <= converge_tol:
                reason = "tolerance"
                break
            if it >= converge:
                reason = "max_iterations"
                break
            error = angle_delta(q_d, q_meas)
            step = np.clip(error, -converge_step, converge_step)
            step_clips += int(np.count_nonzero(np.abs(error) > converge_step))
            candidate = q_hold + step
            accum = angle_delta(candidate, q_d)
            if (np.any(np.abs(accum) > converge_total + 1e-12)
                    or np.any(np.abs(candidate - q_d) > converge_total + 1e-12)):
                reason = "total_limit"
                break
            bounded = model.clamp(candidate)
            if np.any(np.abs(bounded - candidate) > 1e-12):
                reason = "joint_limit"
                break                 # 不发送被静默夹过的闭环候选。
            if np.max(np.abs(step)) <= 1e-12:
                reason = "no_joint_correction"
                break
            q_hold = bounded
            iterations += 1
            if cli.recorder is not None:
                cli.recorder.mark("movel_converge", side)
    result = dict(target=pos, reached=p_meas, stale=not fresh,
                pos_err=float(np.linalg.norm(p_meas - pos)),
                pos_err_xyz=p_meas - pos,
                rot_err=float(np.linalg.norm(log3(rot.T @ r_meas))),
                q_track_err=float(np.max(np.abs(angle_delta(q_meas, q_d)))),
                ik_fails=fails, clipped=clipped, q_cmd=q_d, q_meas=q_meas,
                step_violations=step_violations, step_rejects=step_rejects,
                branch_rejects=branch_rejects, implicit_fallbacks=0,
                peak_raw_dq=peak_raw_dq, peak_accepted_dq=peak_accepted_dq,
                max_joint_step_rad=max_joint_step_rad,
                trajectory_valid=trajectory_valid, aborted=aborted,
                branch=branch_prev, sent_frames=frame)
    if converge:
        result.update(converge_hist=hist, converge_state_counts=state_counts,
                      converge_iterations=iterations, converge_reason=reason,
                      converge_step_clips=step_clips, converged=reason == "tolerance",
                      q_hold=q_hold, q_correction=angle_delta(q_hold, q_d))
    return result


def _validate_converge(rounds: int, tol: float, step: float, total: float) -> None:
    """在创建 ROS 客户端/下发前拒绝非法闭环参数；内部单位 m/rad。"""
    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds < 0:
        raise ValueError("--converge 必须是非负整数")
    for name, value in (("tol", tol), ("step", step), ("total", total)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--converge-{name} 必须是有限正数")
    if step > total:
        raise ValueError("--converge-step 不能大于 --converge-total")
    if total > math.pi:
        raise ValueError("--converge-total 不能大于 180 deg")


def _complete_arm_feedback(cli: X2ArmClient) -> bool:
    """只接受同一最新帧的完整双臂反馈，不能拼接 q() 留存的旧关节。"""
    expected = {f"{side}_{suffix}" for side in ("left", "right")
                for suffix in ARM_JOINT_SUFFIX}
    if cli.state_count <= 0 or not expected.issubset(set(cli.state_names)):
        return False
    for side in ("left", "right"):
        q = cli.q(side)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return False
    return True


def _fresh_hold(cli: X2ArmClient, side: str, q_hold: np.ndarray,
                q_other: np.ndarray, timeout: float = 0.5) -> bool:
    """持续保持并等待完整新反馈；不完整/非有限新帧立即拒绝，禁止用于修正。"""
    dt = 1.0 / cli.rate
    n0 = cli.state_count
    end = time.time() + timeout
    while time.time() < end:
        sent_at = time.time()
        args = ((q_hold, q_other, None, None) if side == "left"
                else (q_other, q_hold, None, None))
        cli.send(*args)
        _pace(cli, sent_at, dt)
        if cli.state_count > n0:
            return _complete_arm_feedback(cli)
    return False


def _pace(cli: X2ArmClient, t0: float, target_elapsed: float) -> None:
    """按目标节拍等待,期间继续处理反馈回调。

    `remain <= 0`(已经落后于实时)那一支**也要 spin 一次**,这不是可省的优化。
    早先那版直接 return,于是控制环一旦落后 —— CPU 被别的进程抢(rviz、另一个
    python 占满一个核)、一次 GC、DDS 抖动都够 —— 整段动作就一次回调都不处理,
    `cli.q()` 返回的是动作**中途**的旧值。报出来的样子是"手臂只走了一半、
    误差方向恰好和运动方向相反",而 IK 失败和裁剪计数全是 0,极难往测量上去想。
    实测中它表现为同一组目标点反复跑、每次坏的点不一样(随机落后的时刻不同)。

    这里用 timeout_sec=0.0 而不是给个正的超时:已经落后了就不能再阻塞,
    非阻塞地取一条就绪的回调既能保证反馈推进,又不会让落后继续放大。
    """
    spun = False
    while True:
        remain = t0 + target_elapsed - time.time()
        if remain <= 0:
            if not spun:
                cli.rclpy.spin_once(cli.node, timeout_sec=0.0)
            return
        cli.rclpy.spin_once(cli.node, timeout_sec=min(remain, 0.002))
        spun = True


# ---------------------------------------------------------------- 子命令

def cmd_state(cli: X2ArmClient, args) -> int:
    if not cli.wait_state():
        # RMW 不一致是这里最常见的原因,而且症状是"话题在、发布者在、回调一次都不进",
        # 所以把当前值打出来,让人能直接和仿真 start_sim.sh 里的那个对。
        print(f"[err] {ARM_STATE_TOPIC} 收不到数据。检查:仿真是否已启动?"
              f"RMW_IMPLEMENTATION(当前 {os.environ.get('RMW_IMPLEMENTATION', '未设')})"
              f"是否和仿真 start_sim.sh 里的一致?")
        return 1
    print(f"反馈话题 {ARM_STATE_TOPIC}  已收 {cli.state_count} 帧")
    print(f"关节数 {len(cli.state_names)}")
    print("  idx  name                       position     velocity     effort")
    for i, n in enumerate(cli.state_names):
        print(f"   {i:2d}   {n:<26s} {cli.state_pos[n]:+9.5f}  {cli.state_vel[n]:+9.5f}  "
              f"{cli.state_eff[n]:+9.4f}")
    print()
    same = list(cli.state_names) == list(DEFAULT_ARM_ORDER)
    print(f"反馈顺序 == HAL default.yaml 假设顺序(左7+右7,腕 yaw/pitch/roll): {same}")
    if not same:
        print("  实测顺序与假设不一致 —— 方案一的 arm_pos[14] 请用 --arm-order 指定实测顺序。")
    act = cli.get_action()
    print(f"当前 mc action: {act if act else '(GetMcAction 不可用)'}")
    for side in ("left", "right"):
        q = cli.q(side)
        p, _ = cli.models[side].forward_kinematics(q)
        print(f"[{side}] q = {np.round(q, 4).tolist()}")
        print(f"        TCP(torso) = {np.round(p, 5).tolist()}  "
              f"离 HOME 最大 {np.degrees(np.max(np.abs(q - HOME_Q))):.2f} deg")
    cli.check_contention()
    return 0


def cmd_order(cli: X2ArmClient, args) -> int:
    """实测 arm_pos[14] 的下标语义。方案一上机第一件事。

    做法:把 14 维指令整体设为当前反馈值,只把第 i 位加一个小量,
    看反馈里**哪个名字**动了。这是唯一能穿透所有 yaml 口径矛盾的判据。
    """
    if cli.mode != "upper_body":
        print("[info] joint_direct 按关节名下发,没有下标歧义,无需 order 测试。")
        return 0
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    if not cli.enter_control():
        return 1

    base = {s: cli.q(s).copy() for s in ("left", "right")}

    # 先量本底漂移:不改指令、只保持基准,看关节自己会漂多少。
    # 这一步不是可选的 —— 如果漂移和 Δ 同量级,后面"哪个关节动了"取的就是
    # 漂得最厉害的那个,和下标语义毫无关系,而结果看起来却像一张正经的映射表。
    _hold(cli, base, 1.5)
    drift_ref = {sd: cli.q(sd).copy() for sd in ("left", "right")}
    _hold(cli, base, 2.0)
    drift, drift_name = 0.0, ""
    for sd in ("left", "right"):
        d = cli.q(sd) - drift_ref[sd]
        for j, suf in enumerate(ARM_JOINT_SUFFIX):
            if abs(d[j]) > drift:
                drift, drift_name = abs(float(d[j])), f"{sd}_{suf}"
    print(f"本底漂移(2s 内不加指令): 最大 {drift:.5f} rad @ {drift_name or 'n/a'}")
    if drift > 0.3 * args.delta:
        print(f"[warn] 漂移达 Δ({args.delta}) 的 {drift / args.delta:.0%},本次结果不可信。")
        print("       常见原因:mc 控制周期超时(日志里 'missed wakeup time')、")
        print("       机器人没站稳(日志里 'Detect Falling Action')、本机不是实时内核。")
        print(f"       可以加大 --delta(比如 {max(0.3, drift * 6):.1f})再试,或先让它站稳。")

    print(f"逐位激励 Δ={args.delta} rad,观察反馈中实际变化的关节名")
    print("  arm_pos 下标  实际动的关节           实测位移 rad   与假设顺序")
    mismatch = 0
    low_conf: List[bool] = []
    for idx in range(14):
        # 保持基准 2 秒,取一个干净的零点
        _hold(cli, base, 1.0)
        ref = {s: cli.q(s).copy() for s in ("left", "right")}
        target = {s: v.copy() for s, v in base.items()}
        name_assumed = cli.arm_order[idx]
        side_a = "left" if name_assumed.startswith("left") else "right"
        j_a = list(ARM_JOINT_SUFFIX).index(name_assumed.split("_", 1)[1])
        target[side_a][j_a] += args.delta
        _hold(cli, target, 2.0)
        moved = []
        for s in ("left", "right"):
            d = cli.q(s) - ref[s]
            for j, suf in enumerate(ARM_JOINT_SUFFIX):
                if abs(d[j]) > args.threshold:
                    moved.append((f"{s}_{suf}", float(d[j])))
        moved.sort(key=lambda x: -abs(x[1]))
        actual = moved[0][0] if moved else "(无响应)"
        disp = moved[0][1] if moved else 0.0
        ok = actual == name_assumed
        mismatch += int(not ok)
        # 信噪比不够就明确标出来,不要让它冒充一条结论
        weak = drift > 0 and abs(disp) < 3 * drift
        note = "一致" if ok else "X 假设是 " + name_assumed
        if weak:
            note += f"  [?信噪比低: |位移| {abs(disp):.4f} < 3x漂移 {3 * drift:.4f}]"
        low_conf.append(weak)
        print(f"      {idx:2d}      {actual:<22s} {disp:+.5f}      {note}")
    _hold(cli, base, 1.0)
    print()
    n_weak = sum(low_conf)
    print(f"不一致项数 = {mismatch};信噪比不足的行 = {n_weak}/14")
    if n_weak:
        print("[warn] 有行的位移压不过本底漂移,这张表不能直接当 --arm-order 用。")
        print("       先解决漂移(见上面的本底漂移一行),再重跑。")
        return 1
    if mismatch:
        print("  => 用实测出来的名字顺序作为 --arm-order 传入,再跑 home / cartesian。")
    else:
        print("  => arm_pos[14] 顺序 = 左臂7 + 右臂7,腕部 yaw, pitch, roll。确认。")
    return 0


def _hold(cli: X2ArmClient, q: Dict[str, np.ndarray], seconds: float) -> None:
    dt = 1.0 / cli.rate
    t0 = time.time()
    for i in range(max(1, int(round(seconds / dt)))):
        cli.send(q["left"], q["right"])
        _pace(cli, t0, (i + 1) * dt)



def _gravity_calibration_poses(cli: X2ArmClient, args) -> Dict[str, Dict[str, np.ndarray]]:
    """返回按侧区分、且避开肘部硬限位的标定姿态；角度单位内部均为 rad。"""
    poses = {"home": {s: HOME_Q.copy() for s in ("left", "right")}}
    if args.pose in ("lateral", "all"):
        poses["lateral"] = {
            s: lateral_raise_q(s, math.radians(args.elbow), cli.models[s])
            for s in ("left", "right")}
    if args.pose in ("forward", "all"):
        poses["forward"] = {}
        for s in ("left", "right"):
            q = HOME_Q.copy()
            q[0] = math.radians(args.forward_pitch)
            poses["forward"][s] = q
    return poses


def _robust_stats(values: Sequence[float]) -> Dict[str, object]:
    a = np.asarray(values, float)
    if a.size == 0:
        return {"n": 0, "median": None, "mad": None}
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {"n": int(a.size), "median": med, "mad": mad,
            "min": float(np.min(a)), "max": float(np.max(a))}


def _calibration_report(samples: List[Dict[str, object]], side: str,
                        mode: str) -> Dict[str, object]:
    out: Dict[str, object] = {"side": side, "mode": mode, "joints": {}}
    for j, name in enumerate(ARM_JOINT_SUFFIX):
        rows = [r for r in samples if r["side"] == side]
        ks, residuals = [], []
        for r in rows:
            err = float(r["q_target"][j] - r["q_meas"][j])
            tau_g = float(r["tau_model"][j])
            tau_m = float(r["tau_meas"][j])
            residuals.append(tau_m - tau_g)
            # k=tau_g/(q_d-q),只接收重力与稳态误差同号且分母足够大的样本。
            if abs(err) >= math.radians(0.3) and tau_g * err > 0:
                k = tau_g / err
                if math.isfinite(k) and 0 < k < 2000:
                    ks.append(k)
        item = {"stiffness": _robust_stats(ks),
                "gravity_residual": _robust_stats(residuals)}
        out["joints"][name] = item
    return out


def cmd_gravity_calibrate(cli: X2ArmClient, args) -> int:
    """多姿态采样，估计方案一 k_eff，同时报告方案二重力残差。"""
    if args.side == "both":
        sides = ("left", "right")
    else:
        sides = (args.side,)
    poses = _gravity_calibration_poses(cli, args)
    plan = {
        "mode": cli.mode, "gravity_source": cli.grav.source,
        "sides": list(sides), "poses": list(poses),
        "repeats": args.repeats, "samples": args.samples,
        "duration": args.duration, "settle": args.settle,
        "elbow_deg": args.elbow, "execute": bool(args.execute),
    }
    print("重力标定计划:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("[安全] 仅加 --execute 才会运动；标定要求 --bias-limit 0，且先处理滑齿/限位问题。")
    if not args.execute:
        print("dry-run：未下发任何运动指令。确认姿态和安全后，重新加 --execute。")
        return 0
    if cli.bias_limit > 1e-12:
        print("[err] 标定必须使用 --bias-limit 0，避免已有补偿污染测量。")
        return 2
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    cli.check_contention()
    if not cli.enter_control():
        return 1
    samples: List[Dict[str, object]] = []
    try:
        for rep in range(args.repeats):
            print(f"--- 第 {rep + 1}/{args.repeats} 轮")
            target = {s: HOME_Q.copy() for s in ("left", "right")}
            goto_joint(cli, target, args.duration, args.settle)
            for pose_name, pose_targets in poses.items():
                target = {s: HOME_Q.copy() for s in ("left", "right")}
                for s in sides:
                    target[s] = pose_targets[s].copy()
                print(f"  姿态 {pose_name}: 到位并采样")
                goto_joint(cli, target, args.duration, args.settle)
                for _ in range(args.samples):
                    if not cli.fresh_state(timeout=1.0):
                        print("[warn] 采样超时，跳过该帧")
                        continue
                    cli.refresh_gravity()
                    for s in sides:
                        q_meas = cli.q(s)
                        tau_meas = cli.tau(s)
                        tau_model = cli.dyns[s].gravity_torque(target[s])
                        samples.append({
                            "repeat": rep + 1, "pose": pose_name, "side": s,
                            "q_target": target[s].tolist(),
                            "q_meas": q_meas.tolist(),
                            "tau_meas": tau_meas.tolist(),
                            "tau_model": tau_model.tolist(),
                        })
                    if args.sample_interval > 0:
                        time.sleep(args.sample_interval)
        report = {"plan": plan, "samples": samples,
                  "report": {s: _calibration_report(samples, s, cli.mode)
                             for s in sides}}
        print("标定统计:")
        for s in sides:
            print(f"[{s}]")
            for name, item in report["report"][s]["joints"].items():
                k = item["stiffness"]
                r = item["gravity_residual"]
                print(f"  {name}: k_eff={k['median']} N·m/rad "
                      f"(n={k['n']}, MAD={k['mad']}); "
                      f"residual={r['median']} N·m (n={r['n']})")
        if args.output:
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"已保存: {args.output}")
        return 0
    finally:
        # 无论中断还是完成，都回 HOME；保持当前 URS/UPPERBODY_REMOTE_SPLIT，
        # 不自动切到 DAMPING_DEFAULT，也不自动写入或启用标定参数。
        goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, args.duration, 0.5)


def cmd_home(cli: X2ArmClient, args) -> int:
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    cli.check_contention()
    if not cli.enter_control():
        return 1
    print(f"回待机位 HOME_Q = {HOME_Q.tolist()}  用时 {args.duration}s")
    res = goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, args.duration, args.settle)
    for s in ("left", "right"):
        r = res[s]
        p, _ = cli.models[s].forward_kinematics(r["q"])
        print(f"[{s}] 稳态关节误差 max = {math.degrees(r['err_max']):.4f} deg")
        print(f"      逐关节 = {np.round(np.degrees(r['err']), 4).tolist()} deg")
        print(f"      TCP(torso) = {np.round(p, 5).tolist()}")
        print(f"      反馈力矩   = {np.round(r['tau'], 3).tolist()} N·m")
    return 0


# ------------------------------------------------- 姿态(RX/RY/RZ)的口径

# 全脚本只认一种姿态表示:**torso 系固定轴 X-Y-Z 欧拉角**,即 R = Rz(rz) @ Ry(ry) @ Rx(rx)。
# 和 URDF 的 <origin rpy="..."> 完全同一个约定(x2_arm_model.rpy_to_matrix),
# 这样"从 URDF 抄一个角度过来"不需要做任何转换。命令行一律用 **deg**,
# 内部一律 rad —— 边界就在 parse 这一层,别在别处再换算。
#
# 绝对 vs 相对:
#   --rpy  RX RY RZ    绝对姿态。TCP 系三根轴相对 torso 系摆成这个欧拉角。
#   --drpy RX RY RZ    在**待机姿态**基础上、绕 **torso 系**的轴再转:R = ΔR @ R_home。
# 想绕 TCP 自身的轴转(手腕拧一下),那是右乘 R_home @ ΔR —— 只有 mdi 里提供,
# 见 mdi 的 `t` 命令。CLI 不给,是因为两种"相对"混在一个 flag 里必然记错。

def publish_target(cli: X2ArmClient, pos: np.ndarray, rot: np.ndarray) -> None:
    """把目标位姿发到 TARGET_TOPIC,给 `viz` 画。没人订阅也无害,所以无条件发 ——
    这样"先开 viz 还是先开 mdi"就不重要了。发布器第一次用时才建。"""
    if getattr(cli, "_target_pub", None) is None:
        from geometry_msgs.msg import PoseStamped
        cli._PoseStamped = PoseStamped
        cli._target_pub = cli.node.create_publisher(PoseStamped, TARGET_TOPIC, 10)
    m = cli._PoseStamped()
    m.header.frame_id = VIZ_ROOT_FRAME
    m.header.stamp = cli.node.get_clock().now().to_msg()
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
    qx, qy, qz, qw = matrix_to_quat(rot)
    m.pose.orientation.x, m.pose.orientation.y = qx, qy
    m.pose.orientation.z, m.pose.orientation.w = qz, qw
    cli._target_pub.publish(m)


def rot_from_args(model: ArmModel, args) -> Tuple[np.ndarray, str]:
    """按 --rpy / --drpy 解出目标姿态矩阵,返回 (R, 一句话说明)。都不给就用待机姿态。"""
    _, r_home = model.forward_kinematics(HOME_Q)
    rpy = getattr(args, "rpy", None)
    drpy = getattr(args, "drpy", None)
    if rpy is not None and drpy is not None:
        raise SystemExit("--rpy 和 --drpy 只能给一个")
    if rpy is not None:
        rot = rpy_to_matrix(np.radians(rpy))
        return rot, f"绝对姿态 rpy = {list(rpy)} deg"
    if drpy is not None:
        rot = rpy_to_matrix(np.radians(drpy)) @ r_home
        return rot, (f"待机姿态 + torso 系 {list(drpy)} deg  "
                     f"-> 绝对 {np.round(rpy_deg(rot), 2).tolist()} deg")
    return r_home, f"待机姿态 rpy = {np.round(rpy_deg(r_home), 2).tolist()} deg"


def preflight(cli: X2ArmClient, side: str, pos: np.ndarray, rot: np.ndarray,
              q_seed: Optional[np.ndarray] = None) -> Dict:
    """**动之前**先在离线模型上解一次:够不够得着、解出来的关节角压不压限位。

    这不是多余的一步 —— goto_cartesian 走的是直线插值 + 逐帧 track,IK 失败时它
    "保持上一帧、绝不丢帧",于是不可达的目标表现成"手臂走一半停住",错误信息
    要翻 ik_fails 计数才看得出来。先解一次就能直接说"这个点差 63 mm 够不着"。
    """
    model, ik = cli.models[side], cli.iks[side]
    pos = np.asarray(pos, float)
    p_c, clipped = ik.project_to_workspace(pos, rot)
    sol = ik.solve(p_c, rot, q_seed=cli.q(side) if q_seed is None else q_seed)
    out = dict(clipped=bool(clipped), clip_mm=float(np.linalg.norm(p_c - pos)) * 1000.0,
               ok=sol is not None, q=None, margin=0.0, pos_err=0.0, rot_err=0.0)
    if sol is not None:
        out.update(q=sol.q, margin=float(model.limit_margin(sol.q)),
                   pos_err=float(sol.pos_error), rot_err=float(sol.rot_error))
    return out


def preflight_line(pf: Dict) -> str:
    """把 preflight 的结论压成一行,mdi 和 pose 共用。"""
    if not pf["ok"]:
        return "  [x] IK 无解 —— 位置够得着但这个姿态拧不过去(腕部限位),换个 rpy 试试"
    bits = [f"  预检: 关节角 {np.round(np.degrees(pf['q']), 2).tolist()} deg",
            f"        限位余量 {math.degrees(pf['margin']):.2f} deg"]
    if pf["clipped"]:
        bits.append(f"  [!] 目标在工作空间外,已沿肩->腕方向拉回 {pf['clip_mm']:.1f} mm —— "
                    f"实际到位点不是你输的点")
    if pf["margin"] < math.radians(2.0):
        bits.append("  [!] 有关节贴着限位,mc 的稳态偏差会把它顶出去,误差表会难看")
    return "\n".join(bits)


def default_targets(side: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """位置 7 点 + 纯姿态 3 点 + 位姿复合 2 点。姿态点位置不动,单看 RX/RY/RZ 的到位精度。"""
    arm = ArmModel(side)
    p0, r0 = arm.forward_kinematics(HOME_Q)
    d = np.array([[0.10, 0, 0], [-0.06, 0, 0], [0, 0.08, 0], [0, -0.08, 0],
                  [0, 0, 0.10], [0, 0, -0.04], [0.06, 0.05, 0.05]])
    names = ["+X 前伸 100mm", "-X 后收 60mm", "+Y 左移 80mm", "-Y 右移 80mm",
             "+Z 上抬 100mm", "-Z 下压 40mm", "斜向复合"]
    out = [(n, p0 + v, r0) for n, v in zip(names, d)]

    # 纯姿态:位置钉死在待机点,只转姿态。这一段才是"带 RX/RY/RZ 的到点测试" ——
    # 位置误差应当仍是亚毫米,姿态误差单独成列,两者不再互相掩盖。
    for label, rv in (("RX +30 deg", [30, 0, 0]), ("RY -30 deg", [0, -30, 0]),
                      ("RZ +40 deg", [0, 0, 40])):
        out.append((f"姿态 {label}", p0, rpy_to_matrix(np.radians(rv)) @ r0))
    # 位姿同时变:检验位置/姿态插值是不是真的同步走完(直线 + expm3 测地线)。
    out.append(("位姿复合 A", p0 + np.array([0.08, 0.0, 0.05]),
                rpy_to_matrix(np.radians([0, -25, 20])) @ r0))
    out.append(("位姿复合 B", p0 + np.array([0.05, 0.06, -0.03]),
                rpy_to_matrix(np.radians([25, 15, -20])) @ r0))
    return out


def _converge_options(args) -> Dict:
    """CLI 的 mm/deg 转为动作原语的 m/rad；关闭时不改变原调用参数。"""
    if not getattr(args, "converge", 0):
        return {}
    return dict(converge=args.converge, converge_tol=args.converge_tol / 1000.0,
                converge_step=math.radians(args.converge_step),
                converge_total=math.radians(args.converge_total))


def _print_convergence(result: Dict) -> None:
    if "converge_hist" not in result:
        return
    curve = " -> ".join(f"{error * 1000:.3f}" for error in result["converge_hist"])
    correction = float(np.max(np.abs(np.degrees(result["q_correction"]))))
    print(f"       闭环 TCP 误差: {curve or '无新鲜测量'} mm  "
          f"修正 {result['converge_iterations']} 轮  "
          f"停止原因 {result['converge_reason']}  "
          f"最大累计修正 {correction:.3f} deg")


def cmd_cartesian(cli: X2ArmClient, args) -> int:
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    if getattr(args, "converge", 0) and not _complete_arm_feedback(cli):
        print("[err] 闭环要求最新反馈帧包含完整双臂 14 关节且角度全部有限；未下发运动")
        return 1
    cli.check_contention()
    if not cli.enter_control():
        return 1
    side = args.side
    model = cli.models[side]
    print("先回待机位 ...")
    goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, args.duration, 1.0)

    if args.target is not None:
        rot, why = rot_from_args(model, args)
        print(f"  姿态: {why}")
        targets = [("自定义", np.array(args.target, float), rot)]
    else:
        targets = default_targets(side)

    print()
    # 关节跟踪误差单独成列:没有它就分不清"我们算错了要去哪"和"算对了但手臂没走到"。
    # 前者是 IK/口径的问题,后者是 mc 增益、实时性或机器人本体状态的问题,
    # 修法完全不同 —— 只看位置/姿态误差这一列会把两者混为一谈。
    print("  目标点            目标 xyz(torso)             目标 rpy(deg)         "
          "位置误差 mm  分量误差 mm                姿态误差 deg  关节跟踪 deg  IK失败 裁剪")
    perrs, rerrs = [], []
    convergence_ok = True
    for name, pt, rt in targets:
        res = goto_cartesian(cli, side, pt, rt, args.duration, args.settle,
                             **_converge_options(args))
        perrs.append(res["pos_err"]); rerrs.append(res["rot_err"])
        print(f"  {name:<16s} {np.round(pt, 4).tolist()!s:<27s} "
              f"{np.round(rpy_deg(rt), 1).tolist()!s:<21s} {res['pos_err']*1000:9.3f}  "
              f"{np.round(res['pos_err_xyz']*1000, 2).tolist()!s:<26s} "
              f"{math.degrees(res['rot_err']):8.3f}  {math.degrees(res['q_track_err']):10.3f}  "
              f"{res['ik_fails']:5d} {res['clipped']:4d}"
              f"  步长拒绝 {res.get('step_rejects', 0):3d}"
              f"  解支拒绝 {res.get('branch_rejects', 0):3d}"
              f"{'  <- 反馈是旧的,这一行不可信' if res.get('stale') else ''}")
        _print_convergence(res)
        convergence_ok = convergence_ok and res.get("converged", True)
        if not convergence_ok:
            print("  [!] 闭环未完成，停止点位序列；保持 URS，不再发回 HOME 指令。")
            break
        goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, 2.0, 0.5)
    print()
    print(f"  位置误差: 中位 {np.median(perrs)*1000:.3f} mm  最大 {np.max(perrs)*1000:.3f} mm")
    print(f"  姿态误差: 中位 {math.degrees(np.median(rerrs)):.3f} deg  "
          f"最大 {math.degrees(np.max(rerrs)):.3f} deg")
    print("  注:两项误差都是用**反馈关节角做正解**得到的 TCP 位姿与目标的差 ——")
    print("      仿真里没有 TCP 话题,这是唯一诚实的量法;它不含 URDF 与真机的装配偏差。")
    print("      姿态误差 = log3(R_target^T @ R_measured) 的模,即等效轴角,和欧拉角分量无关。")
    print("      关节跟踪 deg = max|q_measured - q_commanded|。它大 = 手臂没走到我们要求的地方")
    print("      (mc 增益/实时性/本体状态),它小而误差大 = 我们要求的地方本身就不对(IK/口径)。")
    return 0 if convergence_ok else 1


def cmd_pose(cli: X2ArmClient, args) -> int:
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    if getattr(args, "converge", 0) and not _complete_arm_feedback(cli):
        print("[err] 闭环要求最新反馈帧包含完整双臂 14 关节且角度全部有限；未下发运动")
        return 1
    cli.check_contention()
    if not cli.enter_control():
        return 1
    side = args.side
    model = cli.models[side]
    pos = np.array(args.target, float)
    rot, why = rot_from_args(model, args)
    print(f"[{side}] 目标 xyz = {np.round(pos, 5).tolist()} m")
    print(f"       目标 rpy = {np.round(rpy_deg(rot), 3).tolist()} deg   ({why})")
    print(preflight_line(preflight(cli, side, pos, rot)))
    publish_target(cli, pos, rot)
    res = goto_cartesian(cli, side, pos, rot, args.duration, args.settle,
                         **_converge_options(args))
    r_meas = model.forward_kinematics(res["q_meas"])[1]
    print(f"       实到 xyz = {np.round(res['reached'], 5).tolist()} m")
    print(f"       实到 rpy = {np.round(rpy_deg(r_meas), 3).tolist()} deg")
    print(f"       位置误差 {res['pos_err']*1000:.3f} mm  "
          f"分量 {np.round(res['pos_err_xyz']*1000, 3).tolist()} mm")
    print(f"       姿态误差 {math.degrees(res['rot_err']):.4f} deg  "
          f"(等效轴角;分量 {np.round(np.degrees(log3(rot.T @ r_meas)), 3).tolist()} deg)")
    print(f"       IK 失败 {res['ik_fails']}  裁剪 {res['clipped']}  "
          f"关节跟踪误差 {math.degrees(res['q_track_err']):.4f} deg")
    print(f"       轨迹检查: 步长拒绝 {res['step_rejects']}  解支拒绝 {res['branch_rejects']}  "
          f"最大候选跳变 {math.degrees(res['peak_raw_dq']):.3f} deg  "
          f"最大接受步长 {math.degrees(res['peak_accepted_dq']):.3f} deg")
    if not res['trajectory_valid']:
        print("       [!] 轨迹未完成：已因连续异常停止，未到达目标。")
    if res.get("stale"):
        print("  [!] 测量前没等到新的反馈帧 —— 上面的实到位姿/误差都是旧值,别当结论。")
    _print_convergence(res)
    return 0 if res.get("converged", True) else 1


def cmd_lateral(cli: X2ArmClient, args) -> int:
    """侧平举:手臂向左右抬起,肩/肘/腕保持同高。

    "平不平" 不看下发值,看**反馈关节角做正解**得到的肩/肘/腕/TCP 高度差 ——
    重力下垂、mc 内部 PD 的稳态偏差都只会体现在反馈里。
    """
    if not cli.wait_state():
        print("[err] 没有反馈")
        return 1
    cli.check_contention()
    if not cli.enter_control():
        return 1

    elbow = math.radians(args.elbow)
    q_raise = {s: lateral_raise_q(s, elbow, cli.models[s]) for s in ("left", "right")}
    # --side left/right 时,另一条臂留在待机位
    if args.side != "both":
        other = "left" if args.side == "right" else "right"
        q_raise[other] = HOME_Q.copy()

    print(f"侧平举  肘屈 {args.elbow:.1f} deg  "
          f"({'直臂 T 位' if abs(elbow) < 1e-9 else '上臂侧平 + 前臂朝前'})  "
          f"侧: {args.side}")
    for s in ("left", "right"):
        m = cli.models[s]
        margin = math.degrees(m.limit_margin(q_raise[s]))
        print(f"  [{s}] 目标 q = {np.round(np.degrees(q_raise[s]), 3).tolist()} deg"
              f"   限位余量 {margin:.1f} deg")
        if margin < 1.0:
            print("        [warn] 有关节顶在硬限位上(elbow 上限就是 0)。"
                  "上真机用 --elbow -5 留余量。")
    # 侧平举是重力力矩最大的姿态之一(直臂时 shoulder_roll 上 7.65 N·m)。
    # 8 deg 的默认夹子不够用,不抬的话会残 3 deg 下垂 —— 这不是 mc 跟不上,
    # 是我们自己把偏置切掉了。用户没显式指定就自动抬到够用,并打出来。
    # 注意:"没显式指定"包含标定文件 —— 标定文件里写了 bias_limit_deg 就算用户
    # 指定过,不能再自动抬,否则现场标出来的值会被这里悄悄改掉。
    need = max(float(np.max(np.abs(cli.dyns[s].gravity_torque(q_raise[s]) / cli.k_eff)))
               for s in ("left", "right"))
    explicit = getattr(cli, "bias_limit_explicit", args.bias_limit is not None)
    if not explicit and need > cli.bias_limit:
        was = math.degrees(cli.bias_limit)
        cli.bias_limit = need * 1.1
        print(f"  重力偏置夹子 {was:.2f} -> {math.degrees(cli.bias_limit):.2f} deg"
              f"(该姿态需要 {math.degrees(need):.2f} deg;夹在 {was:.2f} 会残约 "
              f"{math.degrees(need) - was:.2f} deg 下垂)")
    print(f"  抬起 {args.duration:.1f}s -> 保持 {args.hold:.1f}s"
          f"{' -> 回待机位' if not args.no_return else ''}  x{args.cycles}")
    print()

    for c in range(args.cycles):
        if args.cycles > 1:
            print(f"--- 第 {c + 1}/{args.cycles} 轮")
        print("  先回待机位 ...")
        goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, args.duration, 0.5)
        print("  抬起 ...")
        res = goto_joint(cli, q_raise, args.duration, args.hold)
        _report_flat(cli, res, q_raise, args.side)
        if not args.no_return or c + 1 < args.cycles:
            print("  放下 ...")
            goto_joint(cli, {"left": HOME_Q, "right": HOME_Q}, args.duration, 0.5)
    return 0


def _report_flat(cli: X2ArmClient, res: Dict[str, Dict],
                 q_target: Dict[str, np.ndarray], side_sel: str) -> None:
    """按反馈关节角正解,报"实际到底平不平"。"""
    for s in ("left", "right"):
        if side_sel != "both" and s != side_sel:
            continue
        m = cli.models[s]
        q_fb = res[s]["q"]
        pos = m.joint_frames(q_fb)[0]
        tcp = m.forward_kinematics(q_fb)[0]
        # 肩 roll 原点 / 肘 / 腕 yaw / TCP —— 侧平举时这四个点应当同高
        z = np.array([pos[1][2], pos[3][2], pos[4][2], tcp[2]])
        spread = float(z.max() - z.min())
        print(f"  [{s}] 稳态关节误差 max = {math.degrees(res[s]['err_max']):.4f} deg"
              f"   逐关节 = {np.round(np.degrees(res[s]['err']), 3).tolist()} deg")
        print(f"        实际高度 z: 肩 {z[0]:.4f}  肘 {z[1]:.4f}  腕 {z[2]:.4f}  "
              f"TCP {z[3]:.4f} m")
        print(f"        水平度 = 最大高差 {spread * 1000:.2f} mm"
              f"   (TCP 相对肩 {(z[3] - z[0]) * 1000:+.2f} mm,"
              f"倾角 {math.degrees(math.atan2(z[3] - z[0], max(abs(tcp[1] - pos[1][1]), 1e-6))):+.2f} deg)")
        need = np.max(np.abs(cli.dyns[s].gravity_torque(q_target[s]) / cli.k_eff))
        print(f"        重力偏置: 该姿态需要 {math.degrees(need):.2f} deg,"
              f"夹子 {math.degrees(cli.bias_limit):.2f} deg"
              f"{'  <- 被夹住了,残余下垂就是这么来的' if need > cli.bias_limit * 1.001 else ''}")
        p_cmd = m.forward_kinematics(q_target[s])[0]
        print(f"        TCP 反馈 {np.round(tcp, 4).tolist()}  "
              f"指令 {np.round(p_cmd, 4).tolist()}  "
              f"差 {np.linalg.norm(tcp - p_cmd) * 1000:.2f} mm")


MDI_HELP = """\
可输入(位置单位默认 m,姿态一律 deg):
  x y z                走到绝对位置,姿态**尽量**不变(拧不过去会让一点,见下)
  x y z rx ry rz       走到绝对位姿(rpy = torso 系固定轴 X-Y-Z,和 URDF 同一约定)
  d  dx dy dz          相对当前 TCP 平移(torso 系),姿态**尽量**不变
  R  rx ry rz          绕 **torso 系** 的轴转,位置不动     R = dR @ R_now
  t  rx ry rz          绕 **TCP 自身** 的轴转,位置不动     R = R_now @ dR   (拧手腕)
  rpy rx ry rz         直接给绝对姿态,位置不动
  j  q1..q7            直接给 7 个关节角 deg(绕过 IK,用来验证 IK 对不对)
  home                 两臂回待机位          side left|right   换手
  w | where            打印当前位姿          mm                位置输入在 m / mm 间切换
  dur 2.5              设置运动用时 s        settle 1.0        设置到位后稳定时间 s
  dry                  开关 dry(运动/保持零发送)  ?            本帮助
  relax 20             姿态最多让 20 deg     relax 0           关掉降级
  q                    先回待机位再退出(dry 直接退出)  q!      原地退出
  Ctrl-C / EOF         退出会话,不自动回 HOME;异常同样终止会话

只给位置时姿态是软约束:`x y z` 和 `d` 里的 rx/ry/rz 是上一帧继承来的,不是你提的
要求,所以那个姿态在目标点拧不过去时,会绕单轴让**最小的一点**把位置保住,并打印
让了多少度、绕哪条轴。显式给了姿态的(6 个数 / R / t / rpy)一律不降级 —— 那是你
的要求,悄悄改掉比报 "IK 无解" 更坏。位置永远优先:同偏差档里先挑离你输的点最近的。
"""


def _mdi_show(cli: X2ArmClient, side: str) -> Tuple[np.ndarray, np.ndarray]:
    model = cli.models[side]
    q = cli.q(side)
    pos, rot = model.forward_kinematics(q)
    print(f"  [{side}] TCP xyz = {np.round(pos, 5).tolist()} m"
          f"  ({np.round(pos * 1000, 1).tolist()} mm)")
    print(f"        TCP rpy = {np.round(rpy_deg(rot), 3).tolist()} deg")
    print(f"        关节角  = {np.round(np.degrees(q), 2).tolist()} deg")
    print(f"        限位余量 {math.degrees(model.limit_margin(q)):.2f} deg   "
          f"可操作度 {model.manipulability(q):.5f}")
    return pos, rot


def cmd_mdi(cli: X2ArmClient, args) -> int:
    """借用原客户端进入 MDI；主线程保持发送，后台只处理输入和纯 IK。"""
    import x2_mdi
    return x2_mdi.run(cli, args, sys.modules[__name__])


def _rot_angle(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """两个姿态之间的等效轴角(deg)。"""
    return math.degrees(float(np.linalg.norm(log3(np.asarray(rot_a).T @ np.asarray(rot_b)))))


def _imu_field_report(cli: X2ArmClient) -> None:
    """逐字段体检。存在的理由:话题有数据 != 字段都对。

    生态课程那套仿真里,胸腔 IMU 的 angular_velocity 和 linear_acceleration
    填的是**四元数的前三个分量**(wxyz 的 w,x,y),不是真的陀螺/加计 ——
    所以 |acc| 会是 1 附近而不是 9.81,四元数-加计夹角会是一百多度。
    只有 orientation 是可用的。基座 IMU 的三个字段都是真的。
    """
    print()
    print("  IMU 逐字段体检(|acc| 应 ≈ 9.81;差太远说明该字段不是真的加速度)")
    for which, topic in (("chest", CHEST_IMU_TOPIC), ("pelvis", PELVIS_IMU_TOPIC)):
        frame = cli._imu_last.get(which)
        if frame is None:
            print(f"    {which:<7} 无帧")
            continue
        quat, accel = frame
        acc = np.asarray(accel, float)
        line = f"    {which:<7} |acc| = {np.linalg.norm(acc):7.4f} m/s^2"
        if quat is not None:
            g_q = cli.grav.from_orientation(quat)
            g_a = cli.grav.from_accel(acc)
            na, nb = np.linalg.norm(g_q), np.linalg.norm(g_a)
            if na > 1e-6 and nb > 1e-6:
                cos = float(np.clip(np.dot(g_q, g_a) / (na * nb), -1.0, 1.0))
                line += f"   ∠(四元数重力, 加计重力) = {math.degrees(math.acos(cos)):7.2f} deg"
        if np.linalg.norm(acc) < 5.0 or np.linalg.norm(acc) > 15.0:
            line += "   <-- 该 IMU 的加计字段不可信"
        print(line)


def _imu_source_verdict(cli: X2ArmClient) -> None:
    """判定胸腔 IMU 报的到底是 torso 的姿态,还是只是 pelvis 姿态的副本。

    为什么必须判:整套重力补偿建立在"胸腔 IMU = R_torso_in_world"这一条上。
    要是它其实是 pelvis 的姿态被复制过来,那用 chest 就等于**把腰部转角整个丢掉**,
    比老老实实走 pelvis + 腰链还差 —— 而且这个错误在话题层面完全看不出来:
    发布者有、频率对、四元数是单位的、covariance 也不是 -1。

    判据:腰链转角非零时,
        ∠(chest, pelvis)          ≈ 0  -> chest 是 pelvis 的副本,**不能用**
        ∠(chest, pelvis·腰链)     ≈ 0  -> chest 名副其实,首选
    腰链转角本身接近 0 时两个判据等价,判不出来 —— 得先把腰转一点再测。
    """
    chest = cli._imu_last.get("chest")
    pelvis = cli._imu_last.get("pelvis")
    print()
    if not chest or chest[0] is None:
        print("  胸腔 IMU 没有可用的姿态四元数 —— 用 --gravity-source pelvis")
        return
    if not pelvis or pelvis[0] is None:
        print("  基座 IMU 没有姿态四元数,无法交叉判定 chest 是否名副其实。")
        return

    rot_c = quat_to_matrix(*chest[0])
    rot_p = quat_to_matrix(*pelvis[0])
    _, rot_tp = cli.waist_chain.torso_pose_in_pelvis(cli.q_waist)
    waist_ang = _rot_angle(np.eye(3), rot_tp)
    d_copy = _rot_angle(rot_c, rot_p)
    d_true = _rot_angle(rot_c, rot_p @ rot_tp)

    print(f"  胸腔 IMU 挂点交叉判定    (腰链转角 {waist_ang:.3f} deg)")
    print(f"    ∠(chest, pelvis)      = {d_copy:8.4f} deg")
    print(f"    ∠(chest, pelvis·腰链) = {d_true:8.4f} deg")
    if waist_ang < 2.0:
        print("    [判不出来] 腰几乎没转,两个判据等价。先让腰转一点(≥5 deg)再跑本命令。")
    elif d_copy < 0.5 <= d_true:
        print("    [结论] chest 报的是 **pelvis 的姿态**,只是换了个话题名 ——")
        print("           用 chest 会把整个腰部转角丢掉。改用 --gravity-source pelvis。")
    elif d_true < 0.5 <= d_copy:
        print("    [结论] chest 名副其实,报的是 torso_link 姿态。首选 chest。")
    else:
        print("    [结论] 两个判据都不干净 —— 可能腰部在动(两条话题不同时刻)、"
              "或者安装转角没标。")
        print("           静止下重跑一次;仍不干净就先用 pelvis。")


def cmd_imu(cli: X2ArmClient, args) -> int:
    """IMU 通路自检 + 量化"用不用 IMU"到底差多少力矩"。

    这个子命令回答的是一个可证伪的问题:当前躯干姿态下,按 IMU 实测重力算的
    前馈,和按 [0,0,-9.81] 硬算的前馈,差多少 N·m、折成关节角是多少度。
    差得不明显就说明躯干确实是正的,不用纠结;差得明显就必须接 IMU。
    """
    if not cli.wait_state():
        print(f"[err] {ARM_STATE_TOPIC} 收不到数据。")
        return 1
    cli.spin(args.seconds)

    print(f"胸腔 IMU {CHEST_IMU_TOPIC}")
    print(f"  发布者 {cli.node.count_publishers(CHEST_IMU_TOPIC)}  "
          f"收到 {cli.imu_count['chest']} 帧  "
          f"提供姿态四元数: {cli.imu_has_orientation['chest']}")
    print(f"基座 IMU {PELVIS_IMU_TOPIC}   (注意:名字叫 torso,挂点其实是 pelvis)")
    print(f"  发布者 {cli.node.count_publishers(PELVIS_IMU_TOPIC)}  "
          f"收到 {cli.imu_count['pelvis']} 帧  "
          f"提供姿态四元数: {cli.imu_has_orientation['pelvis']}")
    print(f"腰关节 {WAIST_STATE_TOPIC}  q_waist = {np.round(cli.q_waist, 5).tolist()}")

    _imu_field_report(cli)
    _imu_source_verdict(cli)

    g = cli.refresh_gravity()
    print()
    print(cli.grav.describe())
    print()
    print("  臂     关节        IMU重力前馈    写死重力前馈      差值      折算关节角(kp=40)")
    for side in ("left", "right"):
        q = cli.q(side)
        cli.dyns[side].gravity = g
        tau_imu = cli.dyns[side].gravity_torque(q)
        cli.dyns[side].gravity = np.array([0.0, 0.0, -9.81])
        tau_flat = cli.dyns[side].gravity_torque(q)
        cli.dyns[side].gravity = g
        d = tau_imu - tau_flat
        k = int(np.argmax(np.abs(d)))
        for i, suf in enumerate(ARM_JOINT_SUFFIX):
            mark = "  <-- 最大" if i == k else ""
            print(f"  {side:<6} {suf.replace('_joint',''):<12} "
                  f"{tau_imu[i]:+9.4f}   {tau_flat[i]:+9.4f}   {d[i]:+9.4f}   "
                  f"{math.degrees(d[i] / MC_ARM_KP):+8.3f} deg{mark}")
    return 0


# mc 状态机的枚举名 —— 服务里跑的是这些全名,缩写只是命令行图省事。
# 不同版本 SDK 的 py_examples 缩写表不一样(老版把 UPPERBODY_REMOTE_SPLIT 写成 US,
# 1.1 写成 URS),所以这里两个都收,全名也照收。
ACTION_ALIASES = {
    "PD": "PASSIVE_DEFAULT", "DD": "DAMPING_DEFAULT", "JD": "JOINT_DEFAULT",
    "SD": "STAND_DEFAULT", "LD": "LOCOMOTION_DEFAULT",
    "WBT": "WHOLE_BODY_TELEOP", "VR": "VR_REMOTE_CONTROLLER", "BM": "BMIMIC",
    # 下面两个只存在于旧 SDK(生态课程那版),1.1 的 mc 不认。留着是为了给出准确的报错。
    "HO": "HEAD_ONLY", "US": "UPPERBODY_REMOTE_SPLIT", "URS": "UPPERBODY_REMOTE_SPLIT",
}


# ⚠⚠ 会卸力/失稳的 action,**任何自动序列都不许碰**,只能由人显式指定并二次确认。
#   JOINT_DEFAULT(JD):关节电机全部卸力 —— 站着的机器人会直接摔。
#   DAMPING_DEFAULT(DD)/PASSIVE_DEFAULT(PD):阻尼/被动,同样撑不住自重。
#   SIT_JOINT_DEFAULT:坐姿版的 joint-default,一样是**卸力**。名字里带 SIT 容易
#     被当成"坐下所以安全",但站立中切过去等于原地卸力,后果和 JD 相同。
#   现场规矩:所有动作跑完就**停在 URS**,不要"收尾切阻尼"。
UNSAFE_ACTIONS = ("JOINT_DEFAULT", "DAMPING_DEFAULT", "PASSIVE_DEFAULT",
                  "SIT_JOINT_DEFAULT")

# ⚠ 不卸力,但会让**整机**大幅运动(坐下/躺下/起身/行走)。上肢 IK 测试期间没有一个
#   该被下发:机器人站着、旁边有人、手臂可能正举在半空。它们不像 JD 那样"一按就摔",
#   所以不强制 --yes-unsafe,但必须打印一条醒目确认,别让 Tab 补全一滑就走。
#   名单取自 SIM_GUIDE.md:353-356 记的 1.1 实测可选值。
WHOLE_BODY_ACTIONS = (
    "SIT_DOWN_DEFAULT", "SIT_UP_DEFAULT",
    "LIE_DOWN_DEFAULT", "LIE_UP_DEFAULT", "PRONE_UP_DEFAULT",
    "GET_UP_DEFAULT", "GROUNDUP",
    "LOCOMOTION_DEFAULT", "LOCOMOTION_STEP",
    "BMIMIC", "BMIMIC_STEP", "WAIST_ROTATION",
    "WHOLE_BODY_TELEOP", "VR_REMOTE_CONTROLLER",
    "TEST_FOURIER", "TEST_PLATFORM", "FOUNDATION",
)

# 已经站着时,重新 arm 上肢遥操的中转站。只能用站立态,不能用上面那三个。
REARM_VIA = "STAND_DEFAULT"

# SetMcAction 被拒时 response.header.code 的含义。取自板卡文档 modeswitch.html
# ("response.header.code is the action switch result code: 0 means success,
#  non-0 means rejected")。0 不在表里 —— 那是成功,走不到这张表。
MC_ACTION_REJECT = {
    2: "请求非法(InvalidRequest)",
    3: "mc 不认识这个 action(UnknownAction) —— 多半是 SDK 版本对不上",
    4: "当前姿态不允许切过去(InvalidPosture)",
    6: "安全保护中,除 PD/DD 外一律拒绝(SecureLevelForbid) —— 机器人是不是摔过/被急停过?",
    8: "正在运动中(MovingBusy)",
    9: "有动作在执行(MotionBusy)",
    10: "从当前状态没有可达路径(NoTransitionPath) —— 需要先切到中转态",
    104: "被 VR 遥操占用(VrTeleop)",
    110: "被全身遥操占用(WholeBodyTeleop)",
}


def action_graph() -> Tuple[Dict[str, List[str]], List[str]]:
    """从 mc 的 action_ruler.yaml 读真实的状态机。

    可选 action 和允许的跳转随 SDK 版本变 —— 1.1 里既没有 UPPERBODY_REMOTE_SPLIT
    也没有 HEAD_ONLY,硬编码一张表只会给出错误的提示。读不到就返回空,调用方退回旧行为。
    路径由 x2ik.py 通过 X2_MC_HOME 传进来。
    """
    mc = os.environ.get("X2_MC_HOME")
    if not mc:
        return {}, []
    param = Path(mc) / "bin" / "mc_param" / "src" / "robot"
    # 仿真跑的是 sim/ 那份,优先用它
    cands = sorted(param.glob("*/sim/action_ruler.yaml")) + \
            sorted(param.glob("*/action_ruler.yaml"))
    for f in cands:
        try:
            import yaml
            doc = yaml.safe_load(f.read_text()) or {}
        except Exception:
            continue
        rules = doc.get("action_rules") or {}
        if not isinstance(rules, dict) or not rules:
            continue
        graph = {k: list(v.get("next_actions") or []) if isinstance(v, dict) else []
                 for k, v in rules.items()}
        always = list((doc.get("common_rules") or {}).get("allow_all_switch_to") or [])
        return graph, always
    return {}, []


def action_path(graph: Dict[str, List[str]], always: List[str],
                start: str, goal: str) -> Optional[List[str]]:
    """状态机里 start -> goal 的最短跳转序列(不含 start)。走不通返回 None。"""
    if start == goal:
        return []
    seen, queue = {start}, [(start, [])]
    while queue:
        cur, path = queue.pop(0)
        for nxt in list(graph.get(cur) or []) + always:
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


# ---------------------------------------------------------------- rviz 可视化

# 帧名前缀带 x2_,免得和仿真/mc 自己可能发的 TF 撞名。rviz 的 Fixed Frame 用 VIZ_ROOT_FRAME。
VIZ_WORLD_FRAME = "x2_world"
JOINT_FRAME_NAMES = tuple(f"j{i+1}_{n}" for i, n in enumerate(ARM_JOINT_SUFFIX))

RVIZ_CONFIG = HERE / "x2ik.rviz"


def _rviz_config_text() -> str:
    """最小可用的 rviz2 配置。手写 YAML 而不是引 yaml 库:内容是固定的,
    而且这样能在注释里说明每一项为什么这么设。"""
    return f"""Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 0.1
      Plane Cell Count: 12
      Color: 160; 160; 164
    # TF 显示会给**每个**帧画一组坐标三轴 —— torso 轴、7 个关节轴、TCP 轴、目标轴
    # 全都来自这一项,所以 Marker 那边不用再重复画轴。
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Marker Scale: 0.12
      Show Names: true
      Show Axes: true
      Show Arrows: false
    - Class: rviz_default_plugins/MarkerArray
      Name: X2 markers
      Enabled: true
      Topic:
        Value: {MARKER_TOPIC}
        Depth: 5
        Reliability Policy: Reliable
        Durability Policy: Volatile
  Global Options:
    # Fixed Frame 必须是 torso —— IK 的全部坐标都在这个系里,换成 world/pelvis
    # 看到的数字就和你输进去的对不上了。
    Fixed Frame: {VIZ_ROOT_FRAME}
    Background Color: 48; 48; 48
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
      Distance: 1.8
      Focal Point: {{X: 0.1, Y: 0, Z: 0.05}}
      Pitch: 0.35
      Yaw: 2.4
      Target Frame: {VIZ_ROOT_FRAME}
Window Geometry:
  Height: 900
  Width: 1400
  Displays:
    collapsed: false
"""


class VizPublisher:
    """把 torso 系、7 个关节系、TCP 位姿、目标点发成 TF + MarkerArray 给 rviz2。

    只读:订阅关节反馈,自己算正解,**不下发任何指令**。所以可以和 mdi / lateral /
    cartesian 同时开着跑,一个终端看、一个终端动。
    """

    def __init__(self, cli: X2ArmClient, reach: bool = True):
        from geometry_msgs.msg import Point, PoseStamped, TransformStamped
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
        from tf2_ros import TransformBroadcaster
        self._Point, self._Color = Point, ColorRGBA
        self._Marker, self._MarkerArray = Marker, MarkerArray
        self._Tf = TransformStamped
        self.cli, self.reach = cli, reach
        self.tf = TransformBroadcaster(cli.node)
        self.pub = cli.node.create_publisher(MarkerArray, MARKER_TOPIC, 5)
        self.target: Optional[Tuple[np.ndarray, np.ndarray]] = None
        cli.node.create_subscription(PoseStamped, TARGET_TOPIC, self._on_target, 10)

    def _on_target(self, msg) -> None:
        o = msg.pose.orientation
        self.target = (np.array([msg.pose.position.x, msg.pose.position.y,
                                 msg.pose.position.z]),
                       quat_to_matrix(o.x, o.y, o.z, o.w))

    # ------------------------------------------------------------ TF
    def _tf_msg(self, stamp, child: str, pos, rot, parent: str = None):
        t = self._Tf()
        t.header.stamp = stamp
        t.header.frame_id = parent or VIZ_ROOT_FRAME
        t.child_frame_id = child
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        qx, qy, qz, qw = matrix_to_quat(rot)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        return t

    # ------------------------------------------------------------ Marker
    def _marker(self, stamp, ns: str, mid: int, mtype: int, scale, rgba):
        m = self._Marker()
        m.header.frame_id = VIZ_ROOT_FRAME
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, self._Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color = self._Color(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        return m

    def publish(self) -> None:
        stamp = self.cli.node.get_clock().now().to_msg()
        tfs = [self._tf_msg(stamp, VIZ_ROOT_FRAME, np.zeros(3), np.eye(3),
                            parent=VIZ_WORLD_FRAME)]
        markers = []
        # 左臂暖橙、右臂青,和 x2_viz.py 的离线图保持同一套配色。
        colors = {"left": (1.0, 0.5, 0.05, 1.0), "right": (0.09, 0.75, 0.81, 1.0)}
        for k, side in enumerate(("left", "right")):
            model = self.cli.models[side]
            q = self.cli.q(side)
            pos, rot = model.joint_frames(q)
            tcp_p, tcp_r = model.forward_kinematics(q)
            for i, name in enumerate(JOINT_FRAME_NAMES):
                tfs.append(self._tf_msg(stamp, f"x2_{side}_{name}", pos[i], rot[i]))
            tfs.append(self._tf_msg(stamp, f"x2_{side}_tcp", tcp_p, tcp_r))

            line = self._marker(stamp, "skeleton", k, self._Marker.LINE_STRIP,
                                (0.012, 0, 0), colors[side])
            line.points = [self._Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                           for p in list(pos) + [tcp_p]]
            markers.append(line)
            dots = self._marker(stamp, "joints", k, self._Marker.SPHERE_LIST,
                               (0.024, 0.024, 0.024), colors[side])
            dots.points = list(line.points)
            markers.append(dots)
            txt = self._marker(stamp, "tcp_text", k, self._Marker.TEXT_VIEW_FACING,
                               (0, 0, 0.028), (1.0, 1.0, 1.0, 0.95))
            txt.pose.position.x, txt.pose.position.y = float(tcp_p[0]), float(tcp_p[1])
            txt.pose.position.z = float(tcp_p[2]) - 0.05
            txt.text = (f"{side} TCP\n"
                        f"xyz {np.round(tcp_p, 4).tolist()}\n"
                        f"rpy {np.round(rpy_deg(tcp_r), 1).tolist()} deg")
            markers.append(txt)
            if self.reach:
                ik = self.cli.iks[side]
                ball = self._marker(stamp, "reach", k, self._Marker.SPHERE,
                                    (2 * ik.reach_max_limited,) * 3,
                                    (colors[side][0], colors[side][1],
                                     colors[side][2], 0.06))
                ball.pose.position.x = float(ik.shoulder[0])
                ball.pose.position.y = float(ik.shoulder[1])
                ball.pose.position.z = float(ik.shoulder[2])
                markers.append(ball)

        if self.target is not None:
            tp, tr = self.target
            tfs.append(self._tf_msg(stamp, "x2_target", tp, tr))
            ball = self._marker(stamp, "target", 0, self._Marker.SPHERE,
                                (0.03, 0.03, 0.03), (1.0, 0.0, 1.0, 0.85))
            ball.pose.position.x, ball.pose.position.y = float(tp[0]), float(tp[1])
            ball.pose.position.z = float(tp[2])
            markers.append(ball)
            txt = self._marker(stamp, "target", 1, self._Marker.TEXT_VIEW_FACING,
                               (0, 0, 0.03), (1.0, 0.4, 1.0, 0.95))
            txt.pose.position.x, txt.pose.position.y = float(tp[0]), float(tp[1])
            txt.pose.position.z = float(tp[2]) + 0.06
            txt.text = (f"target\nxyz {np.round(tp, 4).tolist()}\n"
                        f"rpy {np.round(rpy_deg(tr), 1).tolist()} deg")
            markers.append(txt)

        ax_txt = self._marker(stamp, "label", 0, self._Marker.TEXT_VIEW_FACING,
                              (0, 0, 0.03), (0.8, 0.8, 0.8, 0.9))
        ax_txt.pose.position.z = -0.06
        ax_txt.text = "torso (IK 基座): X 前 / Y 左 / Z 上"
        markers.append(ax_txt)

        self.tf.sendTransform(tfs)
        arr = self._MarkerArray(); arr.markers = markers
        self.pub.publish(arr)


def cmd_viz(cli: X2ArmClient, args) -> int:
    """把 torso 坐标轴 + 关节骨架 + TCP 位姿 + 目标点实时发给 rviz2。

    **只读**,不下发、不切 mc 模式,所以开着它再另开一个终端跑 mdi / lateral 都行。
    坐标轴是 TF 显示画的:torso、7 个关节、TCP、目标点各一组三轴。
    """
    if not cli.wait_state():
        print("[err] 没有反馈,仿真起了吗?")
        return 1
    RVIZ_CONFIG.write_text(_rviz_config_text(), encoding="utf-8")
    viz = VizPublisher(cli, reach=not args.no_reach)
    if args.target is not None:
        rot, why = rot_from_args(cli.models[args.side], args)
        viz.target = (np.array(args.target, float), rot)
        print(f"静态目标点已设:{np.round(args.target, 4).tolist()}  ({why})")

    proc = None
    if args.rviz:
        import subprocess
        print(f"[viz] 启动 rviz2 -d {RVIZ_CONFIG}")
        proc = subprocess.Popen(["rviz2", "-d", str(RVIZ_CONFIG)])
    else:
        print(f"另开一个终端,先 eval \"$(./x2ik.py env)\" 再:")
        print(f"    rviz2 -d {RVIZ_CONFIG}")
        print("  (或者直接给本命令加 --rviz,由它替你起)")
    print(f"TF: {VIZ_ROOT_FRAME} + x2_<side>_j1..j7 + x2_<side>_tcp"
          f"{' + x2_target' if viz.target else ''}")
    print(f"Marker: {MARKER_TOPIC}     目标点输入: {TARGET_TOPIC} (PoseStamped)")
    print("mdi / pose 子命令会自动把目标位姿发到上面那个话题。Ctrl-C 结束。")

    dt = 1.0 / max(1.0, args.rate)
    t0, n = time.time(), 0
    try:
        while cli.rclpy.ok():
            viz.publish()
            n += 1
            _pace(cli, t0, n * dt)
    except KeyboardInterrupt:
        print(f"\n共发 {n} 帧")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    return 0


def cmd_action(cli: X2ArmClient, args) -> int:
    """读或写 mc 状态机。不带参数只读当前 action 并列出可选值。"""
    if args.name is None:
        # 节点刚建好就问服务,DDS 参与者发现可能还没走完,首次会空手而归。
        # cmd_state 是因为先 wait_state() 转了一会儿才没这问题。
        cli.spin(0.5)
        cur = cli.get_action(timeout=5.0)
        print(f"当前 action: {cur if cur else '(读不到,mc 起了吗?)'}")
        graph, always = action_graph()
        if graph:
            print("本 SDK 可选:" + "  ".join(sorted(set(graph) | set(always))))
            if cur:
                nxt = sorted(set(graph.get(cur) or []) | set(always))
                print(f"从 {cur} 可直接切到:" + ("  ".join(nxt) or "(无)"))
        else:
            print("可选(读不到 mc 配置,下面是旧 SDK 的表):"
                  + "  ".join(sorted(set(ACTION_ALIASES.values()))))
        print("缩写:" + "  ".join(f"{k}={v}" for k, v in ACTION_ALIASES.items()))
        return 0 if cur else 1
    target = ACTION_ALIASES.get(args.name.upper(), args.name.upper())
    # 显式点名卸力/阻尼态也要拦一道:这条命令本身很容易手滑(JD 和 SD 就差一个字母)。
    if target in UNSAFE_ACTIONS and not getattr(args, "yes_unsafe", False):
        print(f"[err] {target} 会让关节电机卸力/进阻尼,站立中的机器人会失稳摔倒,已拦下。")
        print("      现场规矩:所有动作跑完停在 UPPERBODY_REMOTE_SPLIT 即可,不要切阻尼收尾。")
        print("      确实需要(机器已挂吊架/有人扶/已坐下)再加 --yes-unsafe 重跑。")
        return 1
    # 不卸力、但会让整机大幅运动的 action。上肢 IK 测试期间一个都不该下发 ——
    # 机器人站着、手臂可能正举在半空、旁边有人。不强制 --yes-unsafe(它们不是"一按就摔"),
    # 但要求当面敲一次全名,挡掉 Tab 补全手滑。非交互环境下一律拒绝。
    if target in WHOLE_BODY_ACTIONS and not getattr(args, "yes_unsafe", False):
        print(f"[warn] {target} 会让**整机**大幅运动(坐下/躺下/起身/行走一类),"
              f"不是上肢遥操作 action。")
        print("       上肢 IK 测试期间不需要它。手臂若正举在半空,整机动作会把它一起带走。")
        if not sys.stdin.isatty():
            print("[err] 非交互环境,已拒绝。确认现场安全后加 --yes-unsafe 重跑。")
            return 1
        try:
            typed = input(f"       确认请完整敲入 {target} (其它任何输入都取消): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            return 1
        if typed != target:
            print("已取消。")
            return 1
    graph, always = action_graph()
    if graph and target not in graph and target not in always:
        print(f"[err] 本 SDK 的 mc 没有 action `{target}`。"
              f"可选:{'  '.join(sorted(set(graph) | set(always)))}")
        return 1
    # 状态机不允许跨级(PASSIVE 直接跳 URS 会被拒),所以按 action_ruler 现算路径逐步走。
    seq = [target]
    if graph:
        cli.spin(0.5)
        cur = cli.get_action(timeout=5.0)
        if cur:
            found = action_path(graph, always, cur, target)
            if found is None:
                print(f"[err] 从 {cur} 走不到 {target}。"
                      f"可直接切到:{'  '.join(sorted(set(graph.get(cur) or []) | set(always)))}")
                return 1
            if not found:
                print(f"已经在 {target}")
                return 0
            seq = found
            if len(seq) > 1:
                print(f"路径: {cur} -> " + " -> ".join(seq))
    mid = [a for a in seq[:-1] if a in UNSAFE_ACTIONS]
    if mid and not getattr(args, "yes_unsafe", False):
        print(f"[err] 去 {target} 的最短路要穿过卸力/阻尼态 {mid},已拦下。")
        print(f"      完整路径: {' -> '.join(seq)}")
        print("      图上通不代表物理上安全。确认现场安全后加 --yes-unsafe,或自己逐跳走。")
        return 1
    for act in seq:
        if not cli.set_action(act):
            return 1
        cli.spin(1.5)
    print(f"当前 action: {cli.get_action()}")
    return 0


def cmd_passive(cli: X2ArmClient, args) -> int:
    """切阻尼态。⚠ 这**不是**常规收尾动作。

    DAMPING_DEFAULT 撑不住自重,站立中的机器人切过去会失稳。现场规矩是
    所有动作跑完就停在 UPPERBODY_REMOTE_SPLIT,不做"收尾切阻尼"。
    这条命令保留下来只为真的需要卸力的场合(已挂吊架 / 已坐下 / 有人扶)。
    """
    if not getattr(args, "yes_unsafe", False):
        print("[err] DAMPING_DEFAULT 是阻尼态,撑不住自重,站立中的机器人会失稳摔倒,已拦下。")
        print("      现场规矩:动作跑完停在 UPPERBODY_REMOTE_SPLIT 就行,不需要切阻尼收尾。")
        print("      确实要卸力(已挂吊架/已坐下/有人扶)再加 --yes-unsafe 重跑。")
        return 1
    cli.set_action("DAMPING_DEFAULT")
    return 0


def cmd_record(cli: X2ArmClient, args) -> int:
    """只录不动:旁听反馈,落 CSV。另一个进程在驱动运动时用这个。"""
    from x2_record import JointRecorder
    if not cli.wait_state(timeout=10.0):
        print("[err] 等了 10s,反馈一帧都没收到,检查机器人连接", file=sys.stderr)
        return 1
    print(f"[rec] 录 {args.seconds}s -> {args.output}   (只旁听,不下发)")
    rec = JointRecorder(Path(args.output), tcp=args.tcp)
    rec.mark("record_idle", "")
    t0 = time.time()
    dt = 1.0 / args.hz
    frame = 0
    try:
        while time.time() - t0 < args.seconds:
            cli.fresh_state(timeout=0.5)
            # 不下发,所以 q_cmd = q_sent = 当前反馈,表示"保持"。
            # 实际消息是另一个进程在发,这边只是为了录制格式统一而填个占位。
            hold = {"left": cli.q("left"), "right": cli.q("right")}
            rec.tick(cli, hold, hold)
            frame += 1
            _pace(cli, t0, frame * dt)
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()
    return 0


def _add_converge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--converge", type=int, default=0, metavar="N",
                        help="最多额外关节闭环轮数，默认 0 关闭")
    parser.add_argument("--converge-tol", type=float, default=1.0, metavar="MM",
                        help="反馈关节角正解的 TCP 位置容差，默认 1.0 mm")
    parser.add_argument("--converge-step", type=float, default=0.5, metavar="DEG",
                        help="单轮单关节修正上限，默认 0.5 deg")
    parser.add_argument("--converge-total", type=float, default=3.0, metavar="DEG",
                        help="相对原 IK 目标的单关节累计修正上限，默认 3.0 deg")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X2 上肢 IK —— 两种关节接口方案的 ROS2 客户端")
    ap.add_argument("--mode", default="upper_body", choices=["upper_body", "joint_direct"],
                    help="upper_body=方案一(mc 位置接口) / joint_direct=方案二(关节接口)")
    ap.add_argument("--arm-order", default=None,
                    help="实测出的 14 个槽位对应的关节顺序,逗号分隔;两种方案通用,默认 HAL 顺序")
    ap.add_argument("--kp", type=float, default=OFFICIAL_KP,
                    help=f"方案二的 stiffness(默认 {OFFICIAL_KP} = 官方示例值)")
    ap.add_argument("--kd", type=float, default=OFFICIAL_KD,
                    help=f"方案二的 damping(默认 {OFFICIAL_KD} = 官方示例值)")
    ap.add_argument("--gravity-ff", type=float, default=0.8)
    ap.add_argument("--payload", type=float, default=0.0)
    ap.add_argument("--stiffness", type=float, default=None,
                    help="方案一的等效关节刚度 N·m/rad(实测标定值),默认取 mc 的 40")
    ap.add_argument("--no-reenter", dest="reenter", action="store_false",
                    help="mc 已经在上肢遥操 action 时不再重走进入序列。"
                         "默认会重走 —— action 名留在 URS 但实际没 armed 时,"
                         "不重走就表现成'全程没报错、手臂一动不动'。")
    ap.add_argument("--hand-mode", type=int, default=1, choices=[0, 1, 2, 3],
                    help="方案一的 hand_sub_mode。0=NONE(只吃 head_pos,手臂不动!) "
                         "1=夹爪 2=灵巧手关节 3=手势。想动手臂必须非 0,默认 1。")
    ap.add_argument("--bias-limit", type=float, default=None,
                    help="方案一重力偏置的夹子 deg(默认 8)。姿态需要的偏置 = "
                         "重力力矩/等效刚度,直臂侧平举要 11 deg,夹在 8 就会残 3 deg 下垂。"
                         "lateral 子命令在你没显式给这个值时会自动抬到够用为止")
    ap.add_argument("--gravity-source", default=None,
                    choices=["chest", "pelvis", "static"],
                    help="重力向量来源,默认 chest(除非 SN 标定文件另有指定)。"
                         "chest=胸腔 IMU(挂在 torso_link,首选);"
                         "pelvis=基座 IMU+腰关节反馈;static=写死 [0,0,-9.81]")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="在执行子命令的同时录制轨迹到 CSV(仅对 home/pose/cartesian/lateral/mdi 有效)")
    ap.add_argument("--record-tcp", action="store_true",
                    help="录制时同时记录 TCP 位姿(额外计算成本)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("state", help="读反馈,打印关节顺序与当前 action")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("order", help="实测 arm_pos[14] 的下标语义(方案一必做)")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--threshold", type=float, default=0.02)
    p.set_defaults(func=cmd_order)

    p = sub.add_parser("home", help="回待机位")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--settle", type=float, default=2.0)
    p.set_defaults(func=cmd_home)

    p = sub.add_parser("pose", help="走到一个笛卡尔位姿(xyz + 可选 RX/RY/RZ)")
    p.add_argument("target", type=float, nargs=3, help="目标 xyz,torso 系,m")
    p.add_argument("--rpy", type=float, nargs=3, default=None, help='目标姿态 RX RY RZ,deg。torso 系固定轴 X-Y-Z 欧拉角,R = Rz(RZ)@Ry(RY)@Rx(RX),和 URDF 的 <origin rpy> 同一约定。不给 --rpy 也不给 --drpy 就用待机姿态')
    p.add_argument("--drpy", type=float, nargs=3, default=None, help='在**待机姿态**基础上、绕 **torso 系**的轴再转 RX RY RZ,deg。R = ΔR @ R_home。和 --rpy 互斥。想绕 TCP 自身的轴转请用 mdi 的 t 命令')
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--settle", type=float, default=2.0)
    _add_converge_args(p)
    p.set_defaults(func=cmd_pose)

    p = sub.add_parser("cartesian", help="多点到位测试(7 个位置点 + 3 个纯姿态点 + 2 个位姿复合点)")
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--target", type=float, nargs=3, default=None,
                   help="只测这一个点(xyz,m);不给就跑内置的 12 点表")
    p.add_argument("--rpy", type=float, nargs=3, default=None, help='目标姿态 RX RY RZ,deg。torso 系固定轴 X-Y-Z 欧拉角,R = Rz(RZ)@Ry(RY)@Rx(RX),和 URDF 的 <origin rpy> 同一约定。不给 --rpy 也不给 --drpy 就用待机姿态')
    p.add_argument("--drpy", type=float, nargs=3, default=None, help='在**待机姿态**基础上、绕 **torso 系**的轴再转 RX RY RZ,deg。R = ΔR @ R_home。和 --rpy 互斥。想绕 TCP 自身的轴转请用 mdi 的 t 命令')
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--settle", type=float, default=2.0)
    _add_converge_args(p)
    p.set_defaults(func=cmd_cartesian)

    p = sub.add_parser("mdi", help="交互式手动输入笛卡尔位姿(边输边动)")
    p.add_argument("--side", default="right", choices=["left", "right"],
                   help="先操作哪条臂,进去以后可以用 `side left/right` 换")
    p.add_argument("--duration", type=float, default=2.5, help="每次运动用时 s,进去可用 dur 改")
    p.add_argument("--settle", type=float, default=1.0, help="到位后稳定 s,进去可用 settle 改")
    p.add_argument("--no-home-first", dest="home_first", action="store_false",
                   help="进入时不先回待机位(默认会先回,免得从奇怪姿势起步)")
    p.add_argument("--dry", action="store_true",
                   help="启动即只预览；HOME、关节运动和等待保持均不发送运动帧")
    p.add_argument("--relax", type=float, default=RELAX_DEFAULT_DEG,
                   help="只给位置(裸 xyz / d)时,姿态最多允许降级多少 deg;0 = 不降级,"
                        "拧不过去就报无解。进去可用 `relax N` 改")
    p.set_defaults(func=cmd_mdi, home_first=True)

    p = sub.add_parser("viz", help="把 torso 轴 / 关节骨架 / TCP 位姿 / 目标点发给 rviz2(只读)")
    p.add_argument("--rviz", action="store_true", help="顺手把 rviz2 也起起来")
    p.add_argument("--rate", type=float, default=30.0, help="发布频率 Hz")
    p.add_argument("--no-reach", action="store_true", help="不画肩心可达球")
    p.add_argument("--target", type=float, nargs=3, default=None,
                   help="钉一个静态目标点 xyz(m)。也可以什么都不给,由 mdi/pose 推过来")
    p.add_argument("--rpy", type=float, nargs=3, default=None, help='目标姿态 RX RY RZ,deg。torso 系固定轴 X-Y-Z 欧拉角,R = Rz(RZ)@Ry(RY)@Rx(RX),和 URDF 的 <origin rpy> 同一约定。不给 --rpy 也不给 --drpy 就用待机姿态')
    p.add_argument("--drpy", type=float, nargs=3, default=None, help='在**待机姿态**基础上、绕 **torso 系**的轴再转 RX RY RZ,deg。R = ΔR @ R_home。和 --rpy 互斥。想绕 TCP 自身的轴转请用 mdi 的 t 命令')
    p.add_argument("--side", default="right", choices=["left", "right"],
                   help="--drpy 的待机姿态基准取哪条臂")
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("lateral", help="侧平举:手臂向左右抬起并保持水平")
    p.add_argument("--elbow", type=float, default=0.0,
                   help="肘屈角 deg(URDF 限位 [-135, 0],只能取负)。"
                        "0 = 直臂 T 位(默认);-90 = 上臂侧平 + 前臂朝前")
    p.add_argument("--side", default="both", choices=["both", "left", "right"],
                   help="抬哪条臂,另一条留在待机位。默认两条一起")
    p.add_argument("--duration", type=float, default=3.0, help="抬起/放下用时 s")
    p.add_argument("--hold", type=float, default=5.0, help="到位后保持 s")
    p.add_argument("--cycles", type=int, default=1, help="抬起-放下循环次数")
    p.add_argument("--no-return", action="store_true", help="最后停在侧平举,不回待机位")
    p.set_defaults(func=cmd_lateral)

    # ⚠ 这里**不能**再挂 aliases=["calibrate"] —— 下面 9-14 合并进来的
    # `calibrate`(标定结果存取)是独立子命令,argparse 对重名**不报错**,
    # 后注册的会静默顶掉前面的别名。实测 `ros calibrate --side right` 会得到
    # "invalid choice: 'right' (choose from 'save','show')"。
    # 而且 x2ik.py 里 `rest[0] == "calibrate"` 会把它路由到不加载 rclpy 的那条路,
    # 重力标定在那条路上根本跑不起来。重力标定只认全名 gravity-calibrate。
    p = sub.add_parser("gravity-calibrate",
                       help="多姿态采样，估计位置环重力偏置刚度(默认 dry-run)")
    p.add_argument("--side", default="right", choices=["left", "right", "both"])
    p.add_argument("--pose", default="all", choices=["home", "lateral", "forward", "all"])
    p.add_argument("--elbow", type=float, default=-5.0,
                   help="侧平举肘角 deg；默认 -5，避免 0 deg 硬限位")
    p.add_argument("--forward-pitch", type=float, default=55.0,
                   help="前伸姿态 shoulder_pitch deg")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--settle", type=float, default=5.0)
    p.add_argument("--sample-interval", type=float, default=0.05)
    p.add_argument("--output", default=None, help="保存完整 JSON 结果")
    p.add_argument("--execute", action="store_true", help="实际运动；默认只打印计划")
    p.set_defaults(func=cmd_gravity_calibrate)

    p = sub.add_parser("imu", help="检查 IMU 通路,量化重力补偿带来的力矩差")
    p.add_argument("--seconds", type=float, default=3.0)
    p.set_defaults(func=cmd_imu)

    p = sub.add_parser("action", help="读/写 mc 状态机(不带参数只读)")
    p.add_argument("name", nargs="?", default=None,
                   help="枚举全名或缩写,如 SD / STAND_DEFAULT / URS")
    p.add_argument("--yes-unsafe", action="store_true",
                   help="确认切到卸力/阻尼态(JD/DD/PD)。这类 action 会让机器人失稳摔倒,"
                        "只在已挂吊架/已坐下/有人扶时用。")
    p.set_defaults(func=cmd_action)

    p = sub.add_parser("passive",
                       help="⚠ 切 DAMPING_DEFAULT(阻尼卸力,会摔)。不是常规收尾 —— "
                            "跑完停在 URS 就行")
    p.add_argument("--yes-unsafe", action="store_true",
                   help="确认要卸力。不加这个参数命令会直接拒绝执行。")
    p.set_defaults(func=cmd_passive)

    p = sub.add_parser("record", help="录制轨迹 CSV(只旁听反馈,不下发)")
    p.add_argument("--seconds", type=float, default=10.0, help="录制时长")
    p.add_argument("--hz", type=float, default=100.0, help="采样率")
    p.add_argument("--output", default="trajectory.csv", help="输出 CSV 路径")
    p.add_argument("--tcp", action="store_true", help="同时记录 TCP 位姿(额外计算成本)")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("calibrate", help="标定结果的存取(按 SN 落盘/回填,不连仿真/真机)")
    csub = p.add_subparsers(dest="calib_cmd", required=True)
    ps = csub.add_parser("save", help="把一组标定参数存进 calibration/<SN>.json")
    ps.add_argument("--sn", default=None, help="不给就用环境变量 X2_ROBOT_SN")
    ps.add_argument("--stiffness", type=float, required=True)
    ps.add_argument("--bias-limit", type=float, required=True, help="deg")
    ps.add_argument("--gravity-source", required=True, choices=["chest", "pelvis", "static"])
    ps.add_argument("--note", default="")
    ps.set_defaults(func=cmd_calibrate_save)
    pw = csub.add_parser("show", help="打印某 SN(或当前环境 SN)的标定文件内容")
    pw.add_argument("--sn", default=None)
    pw.set_defaults(func=cmd_calibrate_show)

    args = ap.parse_args(argv)
    if args.cmd == "mdi":
        import x2_mdi
        try:
            x2_mdi.validate_args(args)
        except ValueError as exc:
            ap.error(str(exc))
    if args.cmd in ("pose", "cartesian"):
        try:
            _validate_converge(args.converge, args.converge_tol / 1000.0,
                               math.radians(args.converge_step),
                               math.radians(args.converge_total))
            if args.converge and (not math.isfinite(args.duration) or args.duration <= 0
                                  or not math.isfinite(args.settle) or args.settle < 0):
                raise ValueError("闭环要求 --duration 为有限正数，--settle 为有限非负数")
            if args.converge:
                for name in ("target", "rpy", "drpy"):
                    values = getattr(args, name, None)
                    if values is not None and not all(math.isfinite(v) for v in values):
                        raise ValueError(f"闭环目标 {name} 必须全部为有限数")
        except ValueError as exc:
            ap.error(str(exc))

    # calibrate 是纯文件操作,不该要求 rclpy/aimdk_msgs 在场,更不该构造客户端。
    if args.cmd == "calibrate":
        return args.func(args)
    try:
        import rclpy  # noqa: F401
        import aimdk_msgs  # noqa: F401
    except ImportError as exc:
        print(f"缺 {exc.name}。本脚本要在配好 rclpy + aimdk_msgs 的 python3.10 下跑。\n"
              f"  宿主机: ./x2ik.py ros <子命令>   (先 ./x2ik.py doctor 看缺什么)\n"
              f"  容器  : sudo docker exec -it x2-sim bash -l", file=sys.stderr)
        return 2

    order = (tuple(s.strip() for s in args.arm_order.split(",")) if args.arm_order
             else DEFAULT_ARM_ORDER)
    # 标定回填。优先级:CLI 显式值 > calibration/<SN>.json > 内置默认值。
    # 不设 X2_ROBOT_SN、也没有标定文件时,行为和加这段之前完全一致。
    sn = os.environ.get("X2_ROBOT_SN")
    calib = load_calibration(sn) if sn else None
    if sn and calib is None:
        print(f"[warn] 未找到 SN={sn} 的标定文件,用内置默认值", file=sys.stderr)
    elif calib is not None:
        print(f"[info] 已加载 SN={sn} 标定: stiffness={calib.get('stiffness')} "
              f"bias_limit={calib.get('bias_limit_deg')}deg "
              f"gravity_source={calib.get('gravity_source')}")

    def _pick(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        return calib[key] if calib and key in calib else default

    stiffness_val = _pick(args.stiffness, "stiffness", None)
    bias_limit_deg = _pick(args.bias_limit, "bias_limit_deg", 8.0)
    gravity_source = _pick(args.gravity_source, "gravity_source", "chest")

    stiffness = None if stiffness_val is None else np.full(7, stiffness_val)
    cli = X2ArmClient(args.mode, order, args.gravity_ff, args.kp, args.kd,
                      stiffness, args.payload, gravity_source,
                      hand_mode=args.hand_mode,
                      bias_limit=math.radians(bias_limit_deg))
    # lateral 的"偏置夹子自动抬高"只在夹子还是内置默认值时才该动手。
    cli.reenter = args.reenter
    cli.bias_limit_explicit = (args.bias_limit is not None
                               or bool(calib and "bias_limit_deg" in calib))
    if args.record is not None:
        from x2_record import JointRecorder
        cli.recorder = JointRecorder(Path(args.record), tcp=args.record_tcp)
    try:
        return args.func(cli, args)
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            if cli.recorder is not None:
                cli.recorder.close()
        finally:
            try:
                cli.close()
            finally:
                if cli.rclpy.ok():
                    cli.rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
