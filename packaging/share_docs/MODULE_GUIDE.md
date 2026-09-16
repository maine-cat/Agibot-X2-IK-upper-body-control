# 机器人模块与 Python MoveJ

本模块只提供 MDI 和 MoveJ 两类公开入口。运行依赖为 Python 3.10 或以上、NumPy，
以及机器人现有的 ROS 2 / AimDK 消息环境。交付包不包含 ROS、SDK、现场配置或标定。

**使用前提：仅在 URS 模式下支持运动。客户须通过机器人原有工具自行切换至 URS；
MDI 和 MoveJ 均不带模式切换功能。**

## 安装

以下路径均为占位符，须替换为接收者自己的目录。在交付目录安装 wheel：

```bash
python3 -m pip install module/x2ik-0.2.0-py3-none-any.whl
export X2IK_CONFIG=/path/to/robot/x2ik.conf
export X2IK_CALIB_DIR=/path/to/robot/calibration
python3 -m x2ik --help
python3 -m x2ik movej
```

最后一条仅打印右臂 HOME 关节目标，不连接或运动。维护者需准备该机器自己的配置、
身份和标定；不要复制其他机器的参数。Python 业务程序运行前须加载本机 ROS / AimDK 环境。
MDI 与命令行 MoveJ 会按配置加载相应环境。

如果使用独立目录部署，可用 runtime 压缩包替代 wheel：

```bash
tar -xzf module/x2ik-runtime-0.2.0.tar.gz -C /path/to/robot
cd /path/to/robot/runtime
export X2IK_CONFIG=/path/to/robot/x2ik.conf
export X2IK_CALIB_DIR=/path/to/robot/calibration
python3 -m x2ik --help
```

`runtime/MANIFEST.json` 是模块文件校验清单。独立目录方式需要从 `runtime/` 运行命令，
或将该目录加入业务程序的 Python 导入路径。不要在存在旧版 `x2ik.py` 的目录中验证 wheel 导入，
以免该文件遮蔽已安装模块。

## 更换机器人与现场标定

**当前版本没有完整的自动换机标定流程。** 已实现的是按机器人 SN 读取应用层重力补偿参数，
再由维护人员做点位验证。文件放在 `calibration/` 并不表示其中数值已经经过实测辨识。
开发现场用过的 `40 N·m/rad / 12° / pelvis` 是当时的测试起始参数，不能作为另一台机器的标定结果。

### 哪些项目要重新确认

| 项目 | 换机或换工具时的工作 | 当前模块的边界 |
| --- | --- | --- |
| 机器人自身零位、编码器与机械状态 | 按机器人原有维护流程确认；机械维修后重新检查 | 本模块不执行官方零位校准，不修改底层参数或 DDS |
| 型号、关节顺序与几何模型 | 核对本机与所用 URDF、左右臂定义、限位和 HOME 相符 | 同型号且几何一致时可复用模型；SN 不会自动选择新模型或校正装配偏差 |
| 重力补偿 | 在当前固件、工具和负载下核验等效刚度、偏置限幅及 IMU 来源 | 读取人工确认的参数；没有可靠的一键自动辨识与验收流程 |
| 工具 TCP、工具姿态与负载 | 更换手、夹爪或工件后测量相应外参、质量和质心 | 客户接口目前使用腕 roll 连杆原点，无公开的工具外参/负载配置入口；需要维护侧适配 |
| 外部坐标和精度 | 核对 torso 与工位/相机坐标的变换，按需求外部测量 | 关节反馈 FK 和闭环到点结果不等于外部 TCP 绝对精度 |

仅搬动运行软件的电脑、机器人和工具未改变时，不必因此重新做几何标定，但仍需确认加载的是同一台机器的配置。

### 当前参数怎么生效

机器人配置文件中设置实际序列号和 URS 限制，例如：

```ini
X2_ROBOT_SN=YOUR_VERIFIED_ROBOT_SN
X2_URS_ONLY=1
```

序列号由维护者核对后填写；程序不会读取硬件序列号来自动证明这就是当前连接的机器人。
启动前设置 `X2IK_CONFIG` 和 `X2IK_CALIB_DIR`，路径示例见上面的安装步骤。
加载关系为：

```text
X2IK_CONFIG → 本机 x2ik.conf → X2_ROBOT_SN
X2IK_CALIB_DIR/<SN>.json → 应用层补偿参数 → MDI / MoveJ
```

未设置 `X2IK_CALIB_DIR` 时，通常从 `X2IK_CONFIG` 指向的配置文件旁的 `calibration/` 读取。
模块命令行未设置 `X2IK_CONFIG` 时以当前工作目录的 `x2ik.conf` 为入口；直接 Python 调用可能落到
内部模块目录。部署时应显式导出这两个路径，不依赖随入口变化的默认值。
环境中的 `X2_ROBOT_SN` 优先于配置文件，`Robot(robot_sn=...)` 又优先于该环境值，换机时要清除旧会话的覆盖值。
环境变量应在启动 Python、导入模块前设置；`X2IK_CONFIG` / `X2IK_CALIB_DIR` 应通过环境导出，
不是写入 `x2ik.conf` 的字段（该文件只加载 `X2_` 开头的键）。
桌面使用机器人端的文件和环境，本机终端的 export 不会自动传给 SSH 后端；通常把参数目录放在机器人配置文件旁。

JSON 的主要字段如下。下面仅说明结构：`null` 是未完成值，**不能直接作为运行配置**。

```json
{
  "sn": "YOUR_VERIFIED_ROBOT_SN",
  "stiffness": null,
  "bias_limit_deg": null,
  "gravity_source": "现场确认 chest / pelvis / static 之一",
  "saved_at": "记录确认日期",
  "note": "记录固件、工具、负载、参数依据和验收记录位置"
}
```

| 字段 | 实际作用 |
| --- | --- |
| `stiffness` | 等效刚度，有限正数，单位 N·m/rad；用于 `重力力矩 / 刚度` 的位置补偿计算，不修改机器人伺服增益 |
| `bias_limit_deg` | 应用层补偿角度的绝对限幅，单位 deg；0 关闭该补偿，不代表任何负载下都适宜运行 |
| `gravity_source` | `chest` 或 `pelvis` 使用相应 IMU 及坐标变换；`static` 为固定重力假设，不是自动替代失效 IMU |
| `sn` | 本次参数对应的机器人身份，要求填写并与所选 SN 一致 |

当前格式只支持一组公共标量刚度和偏置限幅，两臂七关节共用；不支持分别录入左右臂的七关节辨识表。
在该 JSON 中增加 `tcp_offset`、零偏或 `payload` 并不会使客户 MDI/MoveJ 自动采用这些值。

**当前缺失配置的处理并不统一，不能把“程序启动成功”当成标定通过：**

| 入口 | 当前行为 |
| --- | --- |
| 桌面 MDI 桥接、Python `Robot`、命令行 MoveJ | 已选 SN 而对应文件缺失/无法解析会拒绝；文件中显式 SN 不一致会拒绝。未选 SN 或部分字段缺失时仍可能使用默认值 |
| 终端 MDI | 沿用内部 CLI 的兼容加载逻辑，缺失文件会提示并回退默认值，未统一执行文件内 SN 一致性检查 |

补偿默认值为 `40 N·m/rad / 8° / chest`，不是新机验收结果。部署要求仍是完整填写并核验本机文件，
不能靠省略 SN、缺失字段或默认值绕过现场确认。改动文件后须退出旧控制会话并重新连接才会重新加载。

### 换机交接流程

1. 现场维护者核对机器人真实 SN、型号/固件、机械状态、工具与负载，完成机器人自身要求的基础校准。
2. 为该 SN 单独准备 `x2ik.conf` 与参数目录；保留参数来源和原始记录，不把旧文件改名就当成新机标定。
3. 在不启用发送的条件下核对双臂反馈、关节定义、坐标和所选重力来源；模型或工具不相符时，先完成适配。
4. 由维护者在独立安排的现场调试中确认补偿参数，并明确允许的姿态、工具、负载与误差范围。
   当前版本没有可交给客户直接执行的通用自动估参命令；这一环节尚需工程调试，不能只靠填写 JSON 完成。
5. 客户使用机器人原有工具进入 URS，再按审核过的路径分别验证两臂及业务目标。记录关节跟踪误差、反馈 FK 误差；
   若要求真实工具定位精度，还须使用外部测量。只验证右臂不能代替左臂，闭环达标也不能反推静态参数正确。
6. 将确认后的参数、模块版本、固件、工具/负载及验证记录按 SN 归档。现场数据不进入通用 Git 仓库或交付包；
   换机、换工具、改变负载或影响控制行为的固件更新后，重新核验相应项目。

### 现有开发标定工具的限制

源码中的 `calibrate save/show` 只保存/查看人工给定的参数，不测量机器人、不生成标定结论。
历史 `gravity-calibrate` 会执行姿态采样，但不自动写入启用参数；其无截距 `k=τ/角度误差` 统计
会混入零偏、摩擦和模型误差，旧逐关节拟合表已不作为参数依据。
该工具还保留双臂 HOME 与异常收尾回 HOME 的行为，采样阶段的持续保持、有效样本判定也不完整，
不能作为客户换机必跑流程。精简模块没有公开该命令，也没有自动工具 TCP/装配几何标定功能。

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

with Robot() as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
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

单臂返回 `{q, err, err_max, tau, stale}`：`q` 为末帧反馈，`err` 为反馈减目标，
两者均为七维 rad；`err_max` 为最大绝对关节误差 rad，`tau` 为反馈 effort，`stale` 成功时为 `False`。
双臂返回 `{"left": 单臂结果, "right": 单臂结果}`。反馈不满足条件时抛异常。

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

底层等价入口为 `python3 -m x2ik mdi`；桌面使用 `python3 -m x2ik mdi --stdio`。
它们与 Python MoveJ 共用同一机器人指令通路，不能同时发运动。

模块内部文件不作为业务接口承诺。更新模块时保留本机配置、标定、SDK 和已有结果；
原始开发脚本和历史验收报告不需要随业务程序部署。
