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
