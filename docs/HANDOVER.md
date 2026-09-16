# X2 MDI / MoveJ 交付说明

本说明面向接收模块的使用者。公开入口只有机器人端 MDI 和 Python `Robot.moveJ()`；
桌面客户端是 Linux x86_64 AppImage，通过 SSH 标准输入/输出连接机器人，不新增 HTTP 服务。

首次了解项目请先阅读[项目整体介绍与逆运动学原理](../packaging/share_docs/PROJECT_OVERVIEW.md)。
仅在 URS 模式下支持运动，客户须自行切换；MDI 不带模式切换功能。

## 交付内容

项目中的交付目录为 `delivery/x2ik-0.2.0/`。`SHA256SUMS` 列出本次实际提供的文件；
在该目录执行 `sha256sum -c SHA256SUMS` 可检查完整性。

```text
delivery/x2ik-0.2.0/
├── README.md
├── module/
│   ├── x2ik-0.2.0-py3-none-any.whl
│   └── x2ik-runtime-0.2.0.tar.gz
├── desktop/
│   ├── X2-MDI-0.2.0-x86_64.AppImage
│   └── APPIMAGE_BUILD_INFO.json
├── examples/
│   ├── movej_minimal.py
│   └── mdi_minimal.sh
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── MODULE_GUIDE.md
│   ├── DESKTOP_MDI_GUIDE.md
│   └── COORDINATES.md
├── source/                       桌面源码与构建脚本
├── licenses/                     随包组件的许可文件
├── THIRD_PARTY_NOTICES.txt
└── SHA256SUMS
```

wheel 与 runtime 压缩包提供的是同一模块，按部署方式选用一个即可。
模块不附带 ROS、AimDK、SSH 凭据、机器人现场配置或标定数据。
桌面客户端包含自身运行所需的 Python/Qt，机器人端仍需现有 ROS / AimDK 消息环境。

## 安装机器人模块

要求 Python 3.10 或以上、NumPy，以及适配的 ROS 2 / AimDK 消息环境。在交付目录可使用 wheel：

```bash
python3 -m pip install module/x2ik-0.2.0-py3-none-any.whl
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
python3 -m x2ik movej
```

最后一条默认只打印 HOME 目标，不连接或运动。需由现场维护者提供该机器的 `x2ik.conf`，
按机器人身份准备对应标定，并加载 ROS / AimDK 环境。不要直接沿用另一台机器的配置。

若使用独立目录：

```bash
tar -xzf module/x2ik-runtime-0.2.0.tar.gz
cd runtime
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
```

## Python MoveJ

最简调用如下，执行时会实际运动：

```python
from x2ik import Robot, HOME

with Robot() as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
```

`HOME` 为七关节目标，关节顺序是肩 pitch / roll / yaw、肘、腕 yaw / pitch / roll。
目标用 rad，时长用 s；左右臂由 `side="left"` / `"right"` 指定。
同时控制双臂使用同一个实例：

```python
with Robot() as robot:
    result = robot.moveJ(q_left=HOME, q_right=HOME, duration=8, settle=2)
```

最简示例 `examples/movej_minimal.py` 默认离线，只有加 `--execute` 才连接并运动。
在线运动要求机器人已处于 URS，其他控制会话已退出，现场路径已检查。
MoveJ 方法返回后不后台保持；返回的 `err_max` 是关节误差 rad，不是 TCP 毫米误差。

## MDI

安装模块后，启动示例的默认行为是显示帮助，不连接机器人：

```bash
sh examples/mdi_minimal.sh
sh examples/mdi_minimal.sh --dry --side right
```

`--dry` 可以建立 ROS 连接并订阅机器人反馈，但不发布运动命令。显式使用 `--execute` 才进入在线终端控制，
该会话可能持续发送保持命令。在线终端 `q!` 直接停止发送退出，普通 `q` 会先让双臂 HOME 再退出。
这与关闭桌面窗口的行为不同。

桌面端由 AppImage 启动：

```bash
chmod +x desktop/X2-MDI-0.2.0-x86_64.AppImage
./desktop/X2-MDI-0.2.0-x86_64.AppImage --demo
```

`--demo` 离线展示界面。真实连接时在应用中设置自己的 SSH 目标、机器人项目目录与密钥。
先用普通 SSH 验证主机身份并配置公钥认证。桌面默认连接后只读显示双臂 3D 骨架及数值反馈，
显式启用发送后才允许下发。MDI 支持绝对位置/姿态、相对平移/旋转、关节目标与双臂 HOME。
桌面 MDI 位置输入可选 m / mm，角度用 deg；这与 Python MoveJ 的 rad 不同。

本产品仅在 URS（`UPPERBODY_REMOTE_SPLIT`）模式下支持运动。客户须通过机器人原有
操作工具自行切换到 URS。MDI 不带模式切换功能，也不提供任何配置开关开启该能力。
HOME 会移动双臂，桌面停止下发、断线或关闭不会自动 HOME 或切换状态。
停止下发不是硬件急停；SSH 和本地客户端仍有控制能力，应用形态不构成运动安全保证。
当前模块没有碰撞规划，显示的 TCP 来自关节反馈的模型 FK，不是外部位置测量。

## 维护边界

业务使用者依赖 `Robot`、`HOME` 和 MDI 入口即可。包内其他模块是内部实现，
旧 `X2Arm`、测试脚本、开发机路径及历史验收报告不属于该交付接口。
完整项目的 `results/`、`backups/`、SDK、现场配置与标定不需要复制给一般使用者。
桌面源码及构建脚本在 `source/`，随包组件的许可文件和说明在 `licenses/`、
`THIRD_PARTY_NOTICES.txt`，重新分发时应保留这些内容。
