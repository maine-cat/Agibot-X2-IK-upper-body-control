# V2.1 机器人模块与 Python MoveJ

本模块只提供 MDI 和 MoveJ 两类公开入口。运行依赖为 Python 3.10 或以上、NumPy，
以及机器人现有的 ROS 2 / AimDK 消息环境。交付包不包含 ROS、SDK、现场配置或标定。

**使用前提：仅在 URS 模式下支持运动。客户须通过机器人原有工具自行切换至 URS；
MDI 和 MoveJ 均不带模式切换功能。**

## 安装

以下路径均为占位符，须替换为接收者自己的目录。在交付目录安装 wheel：

```bash
python3 -m pip install --upgrade module/x2ik-2.1.1-py3-none-any.whl
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
python3 -m x2ik movej
```

最后一条仅打印右臂 HOME 关节目标，不连接或运动。维护者需准备该机器自己的运行配置，
核对型号、模型及反馈。客户入口内置固定补偿，不要求创建标定文件。
Python 业务程序运行前须加载本机 ROS / AimDK 环境。
MDI 与命令行 MoveJ 会按配置加载相应环境。

如果使用独立目录部署，可用 runtime 压缩包替代 wheel：

```bash
tar -xzf module/x2ik-runtime-2.1.1.tar.gz -C /path/to/robot
cd /path/to/robot/runtime
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
```

`runtime/MANIFEST.json` 是模块文件校验清单。独立目录方式需要从 `runtime/` 运行命令，
或将该目录加入业务程序的 Python 导入路径。不要在存在旧版 `x2ik.py` 的目录中验证 wheel 导入，
以免该文件遮蔽已安装模块。

## 更换机器人与现场标定

**当前版本尚未提供经过可靠验证的自动标定流程，MDI 和公开 MoveJ 统一使用内置固定补偿：
`40 N·m/rad / 12° / pelvis`。** 换机不需要生成或复制 `calibration/<SN>.json`，
也不需要设置 `X2IK_CALIB_DIR`。这些数值是当前采用的工作参数，不是每台机器的实测标定结果。

### 固定补偿具体怎么工作

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

力矩单位为 N·m；除以等效刚度得到 rad，代码将 12° 转为 rad 后限幅。
**固定的是参数，实际偏置随目标姿态和重力方向逐帧变化**，不是给所有关节固定加 12°。
这属于应用层位置指令补偿，不修改机器人底层伺服刚度、增益或 DDS。
源码中固定参数集中在 `x2_compensation.py`；`x2_arm_dynamics.py` 计算力矩与偏置，
`x2_sim_ros.py` 在发送位置指令前叠加偏置并施加关节限位。

桌面 MDI、终端 MDI、Python `Robot.moveJ()` 和命令行 MoveJ 使用同一套参数，
不会读取 SN 标定文件；旧文件缺失、损坏或含有其他数值均不改变这套固定补偿。
`X2_ROBOT_SN` / `Robot(robot_sn=...)` 仅保留机器标识，不用于选择补偿或自动核验硬件身份。
客户接口不提供补偿参数覆盖入口。URS、关节反馈、IMU 有效性和发布独占检查保持有效；
`pelvis` 数据不可用时不能靠固定参数继续运动，也不会自动改用静态重力。

### 更换机器人需要做什么

1. 核对目标机器型号、关节顺序、限位、HOME 与所用模型匹配，并按机器人原有维护要求确认机械状态及零位。
2. 为新机器准备自己的 ROS / AimDK 环境、`x2ik.conf` 和连接配置。若填写 SN，须人工核对真实机器，清除旧环境中的 SN 覆盖值。
3. 安装当前模块，先以只读方式确认双臂反馈和 `pelvis` IMU 有效。无需另建补偿 JSON，也不要运行历史运动采样命令来完成安装。
4. 客户通过机器人原有工具进入 URS，再按现场审核过的路径分别验证两臂与业务点位；固定参数不替代新机验证。
   若要求真实工具 TCP 精度，须另行使用外部测量。反馈 FK 到点不表示外部定位误差达到 1 mm。
5. 保存模块版本、机器人/固件、工具与负载、使用范围及验证记录。换工具、负载或影响控制行为的固件后，重新核验。

V2.1 提供 `none / hand / gripper / custom` 四种 TCP，默认 `none` 为腕原点。
灵巧手为近似杯子抓取点，夹爪为模型名义中心；实际工具外参可通过自定义文件填写完整平移与旋转。
具体位置、标定采样与文件格式见 [TCP 标定指南](TCP_CALIBRATION_GUIDE.md)。
工具模式不录入质量/重心、不改变固定补偿、不校正装配误差，也不控制手指或夹爪开合。
模型不匹配仍需维护侧适配，不能只复用固定补偿就认定精度满足要求。
仅更换运行软件的电脑、机器人与工具不变时，不会因此要求重新几何标定，但仍需核对连接目标与配置。

### 自动标定和旧工程工具的边界

V2.1 的 `tools/calibrate_tcp.py` 支持已采集腕姿态的离线枢轴拟合，只求 TCP 平移；
工具旋转必须独立给定。它不连接机器人、不采集运动、不做自动刚度或负载标定。

自动标定可以继续开发，但当前缺少完整的采样保持、有效样本判定、异常退出和独立实机验证流程。
仅靠模型力矩与关节误差拟合会混入摩擦、零位、装配及负载影响，尚不能作为客户一键换机标定功能。
如要标定外部 TCP 或装配几何，还需适当的外部测量与辨识流程。

源码里的 `calibrate save/show` 只保存/查看人工参数；历史 `gravity-calibrate` 含运动采样，
不是当前客户操作流程。旧 `X2Arm` 默认配置、非 MDI 工程 CLI 及现场验收脚本仍保留按 SN 文件加载的独立流程，
不属于公开 MDI / `Robot.moveJ()` 接口。修改这些旧文件不会改变客户入口的固定补偿，历史测试结论也不自动覆盖新机。

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
不要自动重试或追加 HOME。MoveJ 没有碰撞规划，也不包含 TCP 1 mm 闭环。

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

模块内部文件不作为业务接口承诺。更新模块时保留本机配置、标定、SDK 和已有结果；
原始开发脚本和历史验收报告不需要随业务程序部署。
