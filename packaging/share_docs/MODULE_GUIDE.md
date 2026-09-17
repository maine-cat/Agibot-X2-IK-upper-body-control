# V2.1 机器人模块与 Python MoveJ

本模块只提供 MDI 和 MoveJ 两类公开入口。运行依赖为 Python 3.10 或以上、NumPy，
以及机器人现有的 ROS 2 / AimDK 消息环境。ROS、SDK 和机器人运行配置由现场提供。

**使用前提：仅在 URS 模式下支持运动。客户须通过机器人原有工具自行切换至 URS；
MDI 和 MoveJ 均不带模式切换功能。**

## 安装

以下路径均为占位符，须替换为接收者自己的目录。在交付目录安装 wheel：

```bash
python3 -m pip install module/x2ik-2.1.0-py3-none-any.whl
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
python3 -m x2ik movej
```

最后一条仅打印右臂 HOME 关节目标，不连接或运动。安装前需准备目标机器的运行配置，
核对型号、模型及反馈。模块内置固定补偿，无需另建补偿文件。
Python 业务程序运行前须加载本机 ROS / AimDK 环境。
MDI 与命令行 MoveJ 会按配置加载相应环境。

如果使用独立目录部署，可用 runtime 压缩包替代 wheel：

```bash
tar -xzf module/x2ik-runtime-2.1.0.tar.gz -C /path/to/robot
cd /path/to/robot/runtime
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
```

`runtime/MANIFEST.json` 是模块文件校验清单。独立目录方式需要从 `runtime/` 运行命令，
或将该目录加入业务程序的 Python 导入路径。

## 更换机器人与现场标定

MDI 和 MoveJ 统一使用内置固定补偿：`40 N·m/rad / ±12° / pelvis`。
这些参数适用于模块的重力位置补偿，未针对每台机器人单独测定。

### 补偿参数

| 参数 | 固定值与作用 |
| --- | --- |
| 等效刚度 | `40 N·m/rad`，两臂各关节共用，用于将模型重力力矩换算为位置偏置 |
| 偏置限幅 | 每关节 `±12°`，限制叠加的位置补偿量 |
| 重力来源 | `pelvis` IMU，经坐标变换取得模型坐标系中的重力方向 |

每个发送周期，根据该帧目标关节角 `q`、模型和最新有效重力方向计算重力力矩 `τ_g(q)`：

```text
Δq = clip(τ_g(q) / 40, −12°, +12°)
q_send = joint_limit_clip(q + Δq)
```

力矩单位为 N·m；除以等效刚度得到 rad，限幅值为 12° 对应的弧度数。
实际偏置随目标姿态和重力方向逐帧变化。
补偿作用于模块发送的位置指令，不调整机器人底层伺服参数。

桌面 MDI、终端 MDI、Python `Robot.moveJ()` 和命令行 MoveJ 使用同一套固定参数。
`Robot(robot_sn=...)` 用于记录机器标识，连接目标须人工核对。
运行时要求机器人处于 URS、关节反馈与 `pelvis` IMU 有效，且模块独占上肢指令发布；
`pelvis` 数据不可用时停止发送运动指令。

### 更换机器人需要做什么

1. 核对目标机器型号、关节顺序、限位、HOME 与所用模型匹配，并按机器人原有维护要求确认机械状态及零位。
2. 为新机器准备自己的 ROS / AimDK 环境、`x2ik.conf` 和连接配置，核对机器标识及连接目标。
3. 安装模块，先以只读方式确认双臂反馈和 `pelvis` IMU 有效。
4. 通过机器人原有工具进入 URS，再按现场审核过的路径分别验证两臂与业务点位，并通过外部测量验收实际 TCP 定位精度。
5. 保存模块版本、机器人/固件、工具与负载、使用范围及验证记录。换工具、负载或影响控制行为的固件后，重新核验。

可选择 `none / hand / gripper / custom` 四种 TCP；`none` 使用腕原点。
灵巧手为近似杯子抓取点，夹爪为模型名义中心；实际工具外参可通过自定义文件填写完整平移与旋转。
具体位置、标定采样与文件格式见 [TCP 标定指南](TCP_CALIBRATION_GUIDE.md)。
工具模式不录入质量/重心、不改变固定补偿、不校正装配误差，也不控制手指或夹爪开合。
目标机器人须与模块模型匹配。
仅更换运行软件的电脑、机器人与工具不变时，不会因此要求重新几何标定，但仍需核对连接目标与配置。

### 自定义 TCP 标定

按 [TCP 标定指南](TCP_CALIBRATION_GUIDE.md) 准备腕姿态样本后，使用交付包中的
`source/tools/calibrate_tcp.py` 离线拟合 TCP 平移；工具旋转需独立测定并填写。
也可将测得的平移与旋转直接填入自定义 TCP 文件。该工具不连接机器人或自动采集运动。

## Python MoveJ

最简文件为 [movej_minimal.py](../examples/movej_minimal.py)。在模块可导入的环境中：

```bash
python3 examples/movej_minimal.py             # 默认离线打印目标
python3 examples/movej_minimal.py --execute   # 实际执行右臂 HOME
```

在线使用要求机器人已处于 `UPPERBODY_REMOTE_SPLIT`（URS），路径已检查，
其他 MDI / MoveJ / 上肢控制会话已退出。API 不自动切换状态。

业务调用如下，执行时会实际运动：

```python
from x2ik import Robot, HOME

with Robot(tcp_mode="gripper") as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
    print(result["position_m"], result["tcp"])  # 所选 TCP 的反馈 FK 与定义
    print(result["err_max"])  # 结束瞬间最大关节误差，rad
```

| 目标 | 调用 |
| --- | --- |
| 右臂 | `robot.moveJ(q, side="right", duration=8, settle=2)` |
| 左臂 | `robot.moveJ(q, side="left", duration=8, settle=2)` |
| 双臂同时 | `robot.moveJ(q_left=q_left, q_right=q_right, duration=8, settle=2)` |

单臂 `q` 为七个有限弧度数；省略 `q` 使用 HOME，省略 `side` 选择右臂。
双臂调用必须同时提供 `q_left`、`q_right`，不能再混用 `q` 或 `side`。
`duration` 为有限正秒数，`settle` 为有限非负秒数。
关节顺序和单位见 [坐标说明](COORDINATES.md)。

单臂返回 `{q, err, err_max, tau, stale, tcp, position_m, rotation_matrix, pose_frame}`：`q` 为末帧反馈，`err` 为反馈减目标，
两者均为七维 rad；`err_max` 为最大绝对关节误差 rad，`tau` 为反馈 effort，`stale` 成功时为 `False`。
`position_m` / `rotation_matrix` 是结束反馈对应的 TCP 位姿，`pose_frame="torso_link"`；
`tcp` 保存本侧模式、工具变换、来源及估计标记。双臂返回 `{"left": 单臂结果, "right": 单臂结果}`。反馈不满足条件时抛异常。

构造参数为 `Robot(tcp_mode="none", tcp_file=None, robot_sn=None, verbose=True)`。
可调用 `robot.moveJ(q, side="right", tcp_mode="hand")` 顺序更换工具；省略覆盖参数时沿用当前选择。
`custom` 需 `tcp_file`，该路径由运行 Python 的机器读取；其他模式不接受文件。
TCP 只改变反馈位姿和笛卡尔解释，不改变 `q`、HOME 或关节插值，也不增加抓取和负载功能。

一个实例使用同一通路控制两臂，方法顺序阻塞执行，返回后不后台保持。
连接与动作包含 URS、反馈、IMU、发布独占和限位等检查；异常后停止当前序列，
不要自动重试或追加 HOME。MoveJ 不提供碰撞规划。

## MDI 入口

推荐人工调试使用 [桌面 MDI](DESKTOP_MDI_GUIDE.md)。终端启动示例为
[mdi_minimal.sh](../examples/mdi_minimal.sh)：

```bash
sh examples/mdi_minimal.sh                           # 帮助，不连接
sh examples/mdi_minimal.sh --dry --side right        # 连接和反馈预览，不发布运动命令
sh examples/mdi_minimal.sh --execute --side right    # 在线终端 MDI，可能持续发送保持
```

在线终端的 `q!` 停止发送退出；普通 `q` 会先执行双臂 HOME。
`--dry` 允许反馈订阅，不等于离线仿真。桌面窗口关闭则不自动 HOME。

入口为 `python3 -m x2ik mdi`；桌面使用 `python3 -m x2ik mdi --stdio`。
二者支持启动参数 `--tcp-mode none|hand|gripper|custom`，自定义追加
`--tcp-file /path/to/robot/my_tcp.json`。例如 `python3 -m x2ik mdi --dry --tcp-mode gripper`。
TCP 在 MDI 会话启动时确定，更换需退出/断开后重新启动；MDI 的笛卡尔位置与姿态都指所选 TCP。
命令行 MoveJ 也支持这两个参数，默认离线，显式 `--execute` 才运动。
它们与 Python MoveJ 共用同一机器人指令通路，不能同时发运动。

业务程序通过 `Robot.moveJ()` 集成关节运动，通过 MDI 完成人工调试。
更新模块时保留本机运行配置、自定义 TCP 文件和 SDK 环境。
