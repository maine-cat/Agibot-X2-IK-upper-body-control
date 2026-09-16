#!/usr/bin/env python3
"""方案一的**最小复现器** —— 用来把"控制通路"和"我们的轨迹/IK/重力代码"解耦。

这个文件**只依赖 rclpy + aimdk_msgs**,不 import 本目录任何其它模块:
没有 IK、没有正解、没有重力补偿、没有五次多项式、没有状态机跳转、没有 URDF。
发布路径逐行照抄官方 `py_examples/upper_body_control.py`(同一个 `create_timer(0.02)`
+ `rclpy.spin`、同样的默认 QoS、同样的 header 填法),唯一的区别是
**`arm_pos` 可以给非零值** —— 官方那四个模式的 `arm_pos` 全是 14 个 0,
照它跑永远看不出手臂能不能动。

用法(先确认 mc 已在 UPPERBODY_REMOTE_SPLIT,本脚本**不会**碰状态机):

    # 1) 先只动一个关节,最小幅度。右肩 roll 抬 20 度:
    ./x2_upper_raw.py --joint right_shoulder_roll --deg -20

    # 2) 阶跃 vs 斜坡:默认 5s 线性斜坡(安全)。想看阶跃响应用 --ramp 0
    ./x2_upper_raw.py --joint right_shoulder_roll --deg -20 --ramp 0

    # 3) 什么都不动,纯看通路(等价于官方示例的 claw 模式)
    ./x2_upper_raw.py

判读:
  * 手臂动了 -> 控制通路和 mc 都正常,问题在我们的轨迹/重力/测量代码
  * 手臂不动 -> 问题在通路或 mc 侧,和我们的算法无关。看本脚本打的三项自检:
      订阅者数、**同一话题上的其它发布者数**、反馈是否在变
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import UpperBodyCommandArray, MessageHeader, JointStateArray
from aimdk_msgs.srv import GetCurrentInputSource, SetMcInputSource
from aimdk_msgs.msg import McInputAction

# source 字段进 mc 的 InputManager 做优先级仲裁。mc.yaml 里列了一张表:
#     - name: rc                 priority 80
#     - name: remote_teleop_pc   priority 81   # 上肢遥操
#     - name: vr / app_proxy / interaction / pnc
# ⚠ 但表里没有的名字**不等于**被拒:官方示例填 `upper_body_example`(不在表里)
#   实测能把手臂拉到全 0(垂直向下)。所以这张表是**优先级表**,不是白名单;
#   反倒是填了表里的 `remote_teleop_pc` 之后,要和同优先级/更高优先级的现役
#   输入源(rc=80,机器上 soc0_rc 一直在跑)抢 —— 抢不到就是"消息收到、不执行"。
#   这就是本工具留 --source 的原因:这是唯一需要做 A/B 的变量。
OFFICIAL_SOURCE = "upper_body_example"   # 官方示例用的,实测能动
PROJECT_SOURCE = "remote_teleop_pc"      # 我们项目用的(x2_sim_ros.py:473)
INPUT_SRC_SRV = "/aimdk_5Fmsgs/srv/GetCurrentInputSource"
SET_INPUT_SRV = "/aimdk_5Fmsgs/srv/SetMcInputSource"

# 待验证的假设(本工具就是为了判定它):
#   mc.yaml 预声明的源(rc / remote_teleop_pc / vr / ...)默认是**未激活**的,
#   要先 SetMcInputSource ADD(1001) 或 ENABLE(2001) 才会被仲裁采纳;
#   而表里没有的名字(upper_body_example)不走仲裁,直接通过。
#   —— 这能同时解释"官方示例能动"和"我们填 remote_teleop_pc 不动"。
#   官方 keyboard.py / mc_locomotion_velocity.py 发指令前都先 ADD→ENABLE,
#   只有 upper_body_control.py 不注册,恰好也只有它用了表外的名字。

UPPER_BODY_TOPIC = "/mc/upper_body_command"
ARM_STATE_TOPIC = "/aima/hal/joint/arm/state"

ARM_JOINT_SUFFIX = ("shoulder_pitch_joint", "shoulder_roll_joint",
                    "shoulder_yaw_joint", "elbow_joint",
                    "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")
# arm_pos[14] 的下标语义。和 x2_sim_ros.py 的 DEFAULT_ARM_ORDER 一致:左 7 + 右 7。
ARM_ORDER = tuple([f"left_{s}" for s in ARM_JOINT_SUFFIX]
                  + [f"right_{s}" for s in ARM_JOINT_SUFFIX])
SHORT = tuple(s.replace("_joint", "") for s in ARM_JOINT_SUFFIX)


class RawUpper(Node):
    def __init__(self, target_deg, hand_mode, ramp, report_hz,
                 source=PROJECT_SOURCE, priority=81, stamp_offset="auto"):
        super().__init__("x2_upper_raw")
        self.target = [math.radians(v) for v in target_deg]
        self.hand_mode = hand_mode
        self.source = source
        self.priority = priority
        # 只读:问 mc 现在到底认谁当输入源。能直接看出我们抢到没抢到。
        self._srcq = self.create_client(GetCurrentInputSource, INPUT_SRC_SRV)
        self._src_fut = None
        self._src_last = None
        self.ramp = float(ramp)
        self.state_pos = {}
        self.state_count = 0
        self.q_first = None
        self.q_span = [0.0] * 14          # 每个关节反馈走过的最大行程
        self._seq = 0
        self.t0 = None

        # ---- 时钟偏差补偿 ----
        # mc.yaml(真机) expiration_time = 200 ms,mc_sim.yaml(仿真) = 1000 ms。
        # mc 拿 header.stamp 和自己的"现在"算帧龄,超过阈值整帧丢弃 —— 丢在仲裁
        # **之前**,所以 source 填什么、hand_sub_mode 填什么都救不回来,表现就是
        # 「订阅者=1、发布者=1、指令正常斜坡,但手臂一动不动,mc 持续输出它内置的
        # 默认姿态(shoulder_pitch=0.4, elbow=-1.2)」。
        #
        # 实测本机(WSL, NTP 已同步)比机器人慢 336 ms ±3 ms —— 卡在 200 和 1000
        # 中间,这正是「仿真好用、真机不动」的根因。
        #
        # auto: 从 /aima/hal/joint/arm/state 的 header.stamp 现场量。
        #   offset = 机器人戳 - 本机收到时刻。这个估计天然偏小一个单程传输延迟,
        #   而偏小是**安全方向**:补完之后帧龄 ≈ 单程延迟(正的、很小),不会变成
        #   未来帧。取中位数而不是均值,免得个别抖动样本把补偿量带跑。
        self.stamp_mode = stamp_offset            # "auto" | "off" | 固定秒数
        self.stamp_offset = 0.0                   # 实际加到发出戳上的秒数
        self._off_samples = []

        # 反馈:只读,不影响下发。QoS 必须和机器人侧一致(BEST_EFFORT),
        # 写成 RELIABLE 的症状是"话题在、发布者在、回调一次都不进"。
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               DurabilityPolicy)
        self.create_subscription(
            JointStateArray, ARM_STATE_TOPIC, self._on_state,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST,
                       durability=DurabilityPolicy.VOLATILE))

        # 下发:和官方示例一字不差 —— 默认 QoS(RELIABLE/KEEP_LAST(10))、depth 10
        self.pub = self.create_publisher(UpperBodyCommandArray, UPPER_BODY_TOPIC, 10)
        self.timer = self.create_timer(0.02, self.publish)      # 50 Hz
        self.rep = self.create_timer(1.0 / report_hz, self.report)

    # ---- 只读反馈 ----
    def _on_state(self, msg):
        # 现场量时钟差。本机收到的时刻用 time.time(),和 header.stamp 同为
        # world time(见 MessageHeader.msg 注释),可直接相减。
        if self.stamp_mode == "auto":
            rob = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if rob > 0.0:                          # 戳为 0 的桩消息不参与
                self._off_samples.append(rob - time.time())
                if len(self._off_samples) > 200:
                    self._off_samples.pop(0)
                v = sorted(self._off_samples)
                self.stamp_offset = v[len(v) // 2]
        for j in msg.joints:
            self.state_pos[j.name] = float(j.position)
        self.state_count += 1
        q = self.q()
        if self.q_first is None:
            self.q_first = q
        else:
            for i in range(14):
                self.q_span[i] = max(self.q_span[i], abs(q[i] - self.q_first[i]))

    def q(self):
        return [self.state_pos.get(n, 0.0) for n in ARM_ORDER]

    # ---- 下发,照抄官方示例 ----
    def publish(self):
        if self.t0 is None:
            self.t0 = time.time()
            self.q_start = self.q()        # 斜坡起点用实测,避免一上来就阶跃
        a = 1.0 if self.ramp <= 0 else min(1.0, (time.time() - self.t0) / self.ramp)
        arm = [self.q_start[i] + (self.target[i] - self.q_start[i]) * a
               for i in range(14)]

        msg = UpperBodyCommandArray()
        # 官方示例直接用本机 now();我们额外加上 stamp_offset,把戳搬进机器人的
        # 时钟域。stamp_offset=0 时这里和官方示例逐位相同。
        ns = self.get_clock().now().nanoseconds + int(self.stamp_offset * 1e9)
        msg.header = MessageHeader()
        msg.header.stamp.sec = ns // 1_000_000_000
        msg.header.stamp.nanosec = ns % 1_000_000_000
        msg.header.frame_id = "mc_upper_body"
        msg.header.sequence = self._seq
        self._seq += 1
        msg.source = self.source
        msg.hand_sub_mode = self.hand_mode
        msg.head_pos = [0.0, 0.0]
        msg.arm_pos = arm
        msg.hand_pos = {1: [0.0, 0.0], 2: [0.0] * 20, 3: [0.0] * 4}.get(self.hand_mode, [])
        self.pub.publish(msg)
        self._last_arm = arm

    def register_source(self) -> bool:
        """照 keyboard.py 的路子把 self.source 注册成活跃输入源:ADD 失败再试 ENABLE。

        ADD 失败的常见原因是"已经注册过"(mc.yaml 预声明的就属于这种),
        所以 ADD 失败**不算失败**,要接着试 ENABLE —— 官方示例就是这么写的。
        """
        cli = self.create_client(SetMcInputSource, SET_INPUT_SRV)
        if not cli.wait_for_service(timeout_sec=8.0):
            print(f"[err] 等 {SET_INPUT_SRV} 超时,无法注册输入源")
            return False

        def call(action_value, label):
            req = SetMcInputSource.Request()
            req.action.value = action_value
            req.input_source.name = self.source
            req.input_source.priority = self.priority
            req.input_source.timeout = 1000          # ms,和官方示例一致
            fut = None
            for _ in range(8):   # 官方示例的重试:远端 peer 首调常常丢
                req.request.header.stamp = self.get_clock().now().to_msg()
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=0.25)
                if fut.done():
                    break
            if fut is None or not fut.done():
                print(f"      {label}: 超时")
                return False
            r = fut.result()
            code = r.response.header.code if r is not None else -1
            print(f"      {label}: code={code} "
                  f"({'OK' if code == 0 else '失败'})")
            return code == 0

        print(f"[注册] source={self.source!r} priority={self.priority}")
        if call(McInputAction.INPUTACTION_ADD, "ADD  (1001)"):
            return True
        print("      ADD 失败(大概率是已预声明/已注册),改试 ENABLE")
        return call(McInputAction.INPUTACTION_ENABLE, "ENABLE(2001)")

    def current_input_source(self):
        """只读:mc 当前生效的输入源。

        ⚠ 本方法是从 timer 回调里调的,**绝不能** spin_until_future_complete ——
        那是在执行器线程里再要执行器转一圈,直接重入死锁(症状就是恒为
        "读不到")。所以这里改成:发一次请求就走,结果留到下一次 report 再收。
        """
        # 先收上一轮的结果
        fut, self._src_fut = self._src_fut, None
        out = self._src_last
        if fut is not None and fut.done():
            r = fut.result()
            src = getattr(r, "input_source", None) if r is not None else None
            if src is not None:
                out = (getattr(src, "name", ""), getattr(src, "priority", -1))
                self._src_last = out
        elif fut is not None:
            self._src_fut = fut          # 还没回来,继续等下一轮
            return out
        # 再发下一轮
        if self._srcq.service_is_ready():
            req = GetCurrentInputSource.Request()
            try:
                req.request.header.stamp = self.get_clock().now().to_msg()
            except AttributeError:
                pass
            self._src_fut = self._srcq.call_async(req)
        return out

    # ---- 三项自检 + 实时对比 ----
    def report(self):
        n_sub = self.count_subscribers(UPPER_BODY_TOPIC)
        n_pub = self.count_publishers(UPPER_BODY_TOPIC)
        moved = max(self.q_span) if self.q_first is not None else 0.0
        print(f"[{self._seq:5d} 帧] 订阅者={n_sub}  同话题发布者={n_pub}(含本节点)"
              f"  反馈帧={self.state_count}  最大行程={math.degrees(moved):6.2f} deg")
        if n_sub == 0:
            print("      [err] 没有节点订阅 —— mc 没在听,发了也没人收")
        if n_pub > 1:
            print("      [err] 还有别的节点在发同一条话题!两个发布者会交替生效。")
            print("            官方示例 upper_body_control 的 arm_pos 是 14 个 0,")
            print("            它要是还在后台跑,会把手臂一直按回零位 —— 先把它停掉。")
        if self.state_count == 0:
            print(f"      [err] {ARM_STATE_TOPIC} 一帧都没收到,无法判断有没有动")
        # 帧龄 = mc 收到时它认为这帧有多老。补偿后应该 ≈ 单程延迟(几 ms)。
        # 未补偿时就等于本机与机器人的时钟差,>200ms 即被真机 mc 丢弃。
        if self._off_samples:
            raw_skew = -sorted(self._off_samples)[len(self._off_samples) // 2]
            age_ms = (raw_skew - self.stamp_offset) * 1000.0
            tag = "" if abs(age_ms) < 200 else "  ← 超过真机 200ms 阈值,整帧会被丢!"
            print(f"      时钟差(本机-机器人)={-raw_skew*1000:+7.1f} ms  "
                  f"补偿={self.stamp_offset*1000:+7.1f} ms  "
                  f"到 mc 时帧龄≈{age_ms:+6.1f} ms{tag}")
        cur = self.current_input_source()
        if cur is None:
            print(f"      本消息 source={self.source!r};mc 当前输入源=(读不到)")
        else:
            nm, pri = cur
            hit = "← 就是我们" if nm == self.source else "← **不是我们**"
            print(f"      本消息 source={self.source!r};"
                  f"mc 当前输入源={nm!r}(priority={pri}) {hit}")
            if nm and nm != self.source:
                print(f"      [err] mc 认的是 {nm!r},我们发的 {self.source!r} 被仲裁掉了 ——")
                print(f"            这就是'消息收到但不执行'。换 --source {nm} 或停掉那个源。")
        q = self.q()
        cmd = getattr(self, "_last_arm", [0.0] * 14)
        for side, off in (("left", 0), ("right", 7)):
            d = "  ".join(f"{SHORT[i]}={math.degrees(cmd[off+i]):7.2f}/"
                          f"{math.degrees(q[off+i]):7.2f}" for i in range(7))
            print(f"      {side:5s} 指令/反馈 deg: {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="方案一最小复现器:只有 rclpy + aimdk_msgs,不含本目录任何算法代码")
    ap.add_argument("--joint", default=None,
                    help="只动一个关节,如 right_shoulder_roll(可省 _joint 后缀)。"
                         "其余 13 个保持当前实测值")
    ap.add_argument("--deg", type=float, default=0.0, help="--joint 的目标角度 deg")
    ap.add_argument("--arm-pos", default=None,
                    help="直接给 14 个角度(deg,逗号分隔),顺序=左7+右7。"
                         "和 --joint 二选一")
    ap.add_argument("--hand-mode", type=int, default=1, choices=[0, 1, 2, 3],
                    help="hand_sub_mode。0 时 mc 只读 head_pos、arm_pos 整段丢弃,"
                         "手臂必然不动 —— 留着是为了做对照实验")
    ap.add_argument("--ramp", type=float, default=5.0,
                    help="从实测起点线性走到目标的时间 s。0 = 阶跃(慎用)")
    ap.add_argument("--seconds", type=float, default=20.0, help="总运行时长 s")
    ap.add_argument("--report-hz", type=float, default=1.0, help="打印频率")
    ap.add_argument("--register-source", action="store_true",
                    help="发指令前先 SetMcInputSource ADD→ENABLE 激活 --source。"
                         "官方 keyboard.py 就是这么做的,但 upper_body_control.py 没做")
    ap.add_argument("--priority", type=int, default=81,
                    help="注册时用的优先级。mc.yaml: rc=80 remote_teleop_pc=81")
    ap.add_argument("--source", default=PROJECT_SOURCE,
                    help=f"消息的 source 字段。默认 {PROJECT_SOURCE}(我们项目用的);"
                         f"官方示例用的是 {OFFICIAL_SOURCE}。"
                         "这是本工具最重要的 A/B 变量")
    ap.add_argument("--stamp-offset", default="auto",
                    help="把 header.stamp 搬进机器人时钟域的补偿量(秒)。"
                         "auto=现场从 /aima/hal/joint/arm/state 量(默认);"
                         "off=不补,和官方示例逐位相同;也可直接给秒数。"
                         "真机 mc 的 expiration_time 只有 200ms,本机时钟慢一点"
                         "就会让每帧过期被丢 —— 这是最重要的 A/B 变量")
    args = ap.parse_args(argv)

    if args.stamp_offset not in ("auto", "off"):
        try:
            float(args.stamp_offset)
        except ValueError:
            print(f"[err] --stamp-offset 要 auto / off / 秒数,给了 {args.stamp_offset!r}")
            return 2

    if args.arm_pos and args.joint:
        print("[err] --joint 和 --arm-pos 二选一"); return 2

    target = [0.0] * 14
    picked = None
    if args.arm_pos:
        vals = [float(v) for v in args.arm_pos.split(",")]
        if len(vals) != 14:
            print(f"[err] --arm-pos 需要 14 个值,给了 {len(vals)}"); return 2
        target = vals
    elif args.joint:
        name = args.joint if args.joint.endswith("_joint") else args.joint + "_joint"
        if name not in ARM_ORDER:
            print(f"[err] 不认识关节 `{args.joint}`。可选:\n      "
                  + "\n      ".join(ARM_ORDER)); return 2
        picked = ARM_ORDER.index(name)
        target[picked] = args.deg

    rclpy.init()
    node = RawUpper(target, args.hand_mode, args.ramp, args.report_hz,
                    source=args.source, priority=args.priority,
                    stamp_offset=args.stamp_offset)
    if args.stamp_offset not in ("auto", "off"):
        node.stamp_offset = float(args.stamp_offset)

    # 起点用实测值,所以先转一会儿把反馈收进来,再开始发
    print(f"等 {ARM_STATE_TOPIC} 反馈 ...")
    end = time.time() + 5.0
    while time.time() < end and node.state_count == 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.state_count == 0:
        print(f"[err] 5s 内没有反馈。mc / HAL 起了吗?");
        node.destroy_node(); rclpy.shutdown(); return 1
    if args.register_source:
        if not node.register_source():
            print("[err] 输入源注册失败 —— 继续发也大概率不执行,但还是跑完让你看数据")
    node.q_start = node.q()
    q0 = node.q()

    if picked is not None:
        tgt = list(q0); tgt[picked] = math.radians(args.deg)
        node.target = tgt
        print(f"只动 {ARM_ORDER[picked]}: {math.degrees(q0[picked]):.2f} -> "
              f"{args.deg:.2f} deg,其余 13 个保持实测值")
    else:
        node.target = [math.radians(v) for v in target]
        print(f"arm_pos 全给: {[round(v,2) for v in target]} deg")
    print(f"hand_sub_mode={args.hand_mode}  斜坡 {args.ramp}s  共跑 {args.seconds}s  "
          f"50 Hz(和官方示例同一条路)")
    if args.hand_mode == 0:
        print("[warn] hand_sub_mode=0:mc 会丢弃整段 arm_pos,手臂必然不动(这是对照组)")
    print()

    node.t0 = None
    t_end = time.time() + args.seconds
    try:
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass

    span = max(node.q_span)
    print()
    print(f"===== 结论:整个过程中反馈的最大行程 = {math.degrees(span):.3f} deg =====")
    if math.degrees(span) < 0.5:
        print("手臂**没有动**。这条路上没有我们的 IK/重力/插值代码,")
        print(f"所以问题不在算法侧(本次 source={args.source!r})。依次排:")
        print("  1. 订阅者=0        -> mc 没在听")
        print("  2. 同话题发布者>1  -> 有别的进程在抢(官方示例发的是 14 个 0)")
        print(f"  3. 换 source 再跑一遍 —— 这是目前最可疑的一项:")
        print(f"       ./x2ik.py raw --source {OFFICIAL_SOURCE} <同样的参数>")
        print(f"     官方示例用 {OFFICIAL_SOURCE!r} 实测能动;我们项目用的是")
        print(f"     {PROJECT_SOURCE!r},它在 mc.yaml 的 input_sources 表里(priority 81)。")
        print("     换成官方那个名字就动 -> 根因是 source 走了仲裁且没被采纳,")
        print("     改 x2_sim_ros.py:473 即可;两个都不动 -> 再查 armed/hand_sub_mode。")
        print("  4. 都排完还不动 -> 重走 URS -> STAND_DEFAULT -> URS 再试")
    else:
        print("手臂**动了** -> 控制通路正常。问题在我们的轨迹/重力/测量代码,")
        print("   下一步用 `./x2ik.py ros --bias-limit 0 home` 对照,差异就在算法侧。")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
