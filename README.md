# X2 MDI / MoveJ V2.1 交付说明

本说明面向接收模块的使用者。公开入口只有机器人端 MDI 和 Python `Robot.moveJ()`；
桌面客户端是 Linux x86_64 AppImage，通过 SSH 标准输入/输出连接机器人，不新增 HTTP 服务。

首次了解项目请先阅读[项目整体介绍与逆运动学原理](docs/PROJECT_OVERVIEW.md)。
仅在 URS 模式下支持运动，客户须自行切换；MDI 不带模式切换功能。
产品版本 V2.1，机器人模块版本 `2.1.1`，桌面客户端版本 `2.1.0`，源码工作分支 `V2.0`。

2.1.1 将位姿复核、原位保持和局部连续求解并入原有 IK，MDI 直接使用更新后的求解器，
无需另装验证模块。Python `Robot.moveJ()` 的调用方式及返回字段保持不变，关节目标仍走关节空间插值。

## 交付内容

以下命令均在交付根目录执行，`SHA256SUMS` 列出实际提供的文件，
执行 `sha256sum -c SHA256SUMS` 可检查完整性。

```text
交付目录/
├── README.md
├── module/
│   ├── x2ik-2.1.1-py3-none-any.whl
│   └── x2ik-runtime-2.1.1.tar.gz
├── desktop/
│   ├── X2-MDI-2.1.0-x86_64.AppImage
│   └── APPIMAGE_BUILD_INFO.json
├── examples/
│   ├── movej_minimal.py
│   ├── mdi_minimal.sh
│   └── tcp_tool.example.json
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── MODULE_GUIDE.md
│   ├── DESKTOP_MDI_GUIDE.md
│   ├── COORDINATES.md
│   ├── TCP_CALIBRATION_GUIDE.md
│   └── IK_VALIDATION.md
├── source/
│   ├── x2ik/                     完整机器人模块、模型及 TCP 预置
│   ├── tests/                    离线回归测试和固定目标样本
│   ├── pyproject.toml
│   ├── tools/                    模块构建、包验证、桌面构建与 TCP 标定
│   ├── desktop/                  桌面客户端源码
│   └── x2_tcp.py / assets/        桌面及离线标定工具依赖
├── verification/                 安装包离线测试及源码哈希记录
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
python3 -m pip install --upgrade module/x2ik-2.1.1-py3-none-any.whl
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
python3 -m x2ik movej
```

最后一条默认只打印 HOME 目标，不连接或运动。需由现场维护者提供该机器的 `x2ik.conf`，
并加载 ROS / AimDK 环境。客户 MDI / MoveJ 内置固定 `40 N·m/rad / 12° / pelvis` 补偿，
不读取或要求创建 SN 标定文件。当前没有自动运动标定流程，固定参数也不代表新机精度验收。
V2.1 支持四种 TCP 与离线平移拟合，见 [TCP 标定与文件填写](docs/TCP_CALIBRATION_GUIDE.md)。
换机仍需核对型号、模型、有效反馈与 IMU，并分别验证两臂，具体见
[模块指南](docs/MODULE_GUIDE.md#更换机器人与现场标定)。

若使用独立目录：

```bash
tar -xzf module/x2ik-runtime-2.1.1.tar.gz
cd runtime
export X2IK_CONFIG=/path/to/robot/x2ik.conf
python3 -m x2ik --help
```

升级后可用 `python3 -c "import x2ik; print(x2ik.__version__, x2ik.__file__)"`
确认实际加载的是 2.1.1。独立目录部署应先解压到新目录，再切换运行路径，避免混用旧模块。
桌面 2.1.0 通过 SSH 调用机器人侧模块，升级机器人模块后即可使用新的 IK，桌面协议保持兼容。

## Python MoveJ

最简调用如下，执行时会实际运动：

```python
from x2ik import Robot, HOME

with Robot(tcp_mode="gripper") as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
```

`HOME` 为七关节目标，关节顺序是肩 pitch / roll / yaw、肘、腕 yaw / pitch / roll。
目标用 rad，时长用 s；左右臂由 `side="left"` / `"right"` 指定。
同时控制双臂使用同一个实例：

```python
with Robot(tcp_mode="gripper") as robot:
    result = robot.moveJ(q_left=HOME, q_right=HOME, duration=8, settle=2)
```

默认 `tcp_mode="none"` 为腕原点。可选 `hand` 灵巧手近似杯子中心、`gripper` 夹爪名义中心，
或 `custom` 配合 `tcp_file`。`moveJ(..., tcp_mode="hand")` 可顺序覆盖，省略时沿用当前选择。
TCP 不改变七关节目标和 HOME；结果 `position_m/rotation_matrix/tcp` 报告所选工具的反馈 FK。
手指/夹爪开合与工具质量适配不在此接口内。

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
chmod +x desktop/X2-MDI-2.1.0-x86_64.AppImage
./desktop/X2-MDI-2.1.0-x86_64.AppImage --demo
```

`--demo` 离线展示界面。真实连接时在应用中设置自己的 SSH 目标、机器人项目目录与密钥。
先用普通 SSH 验证主机身份并配置公钥认证。连接前选择 TCP；自定义填写机器人上的绝对 JSON 路径，
`--demo` 才读取本机文件。更换工具需断开重连。桌面默认连接后只读显示双臂 3D 模型及数值反馈，
显式启用发送后才允许下发。MDI 支持绝对位置/姿态、相对平移/旋转、关节目标与双臂 HOME。
桌面 MDI 位置输入可选 m / mm，角度用 deg；这与 Python MoveJ 的 rad 不同。
MDI 笛卡尔目标统一指所选 TCP；CLI 支持 `--tcp-mode` / `--tcp-file`。
目标预检使用原始位姿求解，返回前按完整模型复核位置、姿态和关节限位，
当前姿态已经满足目标时保留原关节角，连续运动优先从上一帧进行局部求解。
运动中保持解析分支检查和 0.05 rad 单帧关节变化上限，不启用全局备用种子，
连续五帧无合格解时停止推进，单帧上限不等同于速度或加速度规划。
离线测试范围和复现方法见 [IK 验证说明](docs/IK_VALIDATION.md)。
3D 外观来自配套 STL 的简化网格，未发现 STEP 资源；缺少有效资源时回退骨架，不作碰撞模型使用。

本产品仅在 URS（`UPPERBODY_REMOTE_SPLIT`）模式下支持运动。客户须通过机器人原有
操作工具自行切换到 URS。MDI 不带模式切换功能，也不提供任何配置开关开启该能力。
HOME 会移动双臂，桌面停止下发、断线或关闭不会自动 HOME 或切换状态。
停止下发不是硬件急停；SSH 和本地客户端仍有控制能力，应用形态不构成运动安全保证。
当前模块没有碰撞规划，显示的 TCP 来自关节反馈的模型 FK，不是外部位置测量。

## 维护边界

业务使用者依赖 `Robot`、`HOME` 和 MDI 入口即可。包内其他模块是内部实现，
旧 `X2Arm`、测试脚本、开发机路径及历史验收报告不属于该交付接口。
完整项目的 `results/`、`backups/`、SDK、现场配置与标定不需要复制给一般使用者。
机器人模块、桌面源码及构建脚本在 `source/`，随包组件的许可文件和说明在 `licenses/`、
`THIRD_PARTY_NOTICES.txt`，重新分发时应保留这些内容。

源码测试和机器人模块构建不需要 ROS、SDK 或 Isaac Sim，在独立 Python 环境中执行：

```bash
cd source
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "numpy>=1.21" "setuptools>=61" wheel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/build_module.py
python tools/verify_release.py --output ../verification/2.1.1-local.json
python tools/update_checksums.py
```

构建同时生成 `module/` 下的 wheel 和 runtime，二者来自同一份 `source/x2ik/`，
包验证会在临时目录安装 wheel、解压 runtime，分别运行离线测试与命令入口，
不连接机器人、不改变当前 Python 环境的安装内容，验证结果文件已存在时需换一个文件名。
