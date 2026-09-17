# X2 双臂运动控制 V2.1：MDI 与 MoveJ

面向 X2 上肢调试和业务集成的 Python 模块。人工调试使用 Linux AppImage 桌面 MDI，业务程序使用 `Robot.moveJ()`。两者共用模型、运动生成、URS 与反馈检查。

**仅在已处于 URS（`UPPERBODY_REMOTE_SPLIT`）的机器人上支持运动。客户必须通过机器人原有操作工具自行切换到 URS。MDI 和 MoveJ 不带模式切换功能。**

MDI / 公开 MoveJ 统一内置 `40 N·m/rad / 12° / pelvis` 补偿，无需按 SN 准备标定文件。
TCP 提供 `none / hand / gripper / custom`，默认腕原点；手/夹爪为近似或名义抓取中心。
自定义支持完整外参和离线枢轴平移拟合，不自动采集运动、不控制开合或更新负载。
换机需核对模型、有效反馈与 IMU，并验证业务点位。产品/包版本 `2.1.0`，工作分支 `V2.0`。

## 文档入口

| 需要了解 | 文档 |
| --- | --- |
| 整体架构、接口、逆运动学原理和能力边界 | [项目介绍](docs/PROJECT_OVERVIEW.md) |
| 模块安装和 Python MoveJ | [模块指南](docs/MODULE_GUIDE.md) |
| 换机配置、补偿参数与标定能力边界 | [更换机器人与现场标定](docs/MODULE_GUIDE.md#更换机器人与现场标定) |
| 桌面连接、3D 反馈、MDI 输入 | [桌面 MDI](docs/DESKTOP_MDI_GUIDE.md) |
| 坐标、关节顺序、单位与 TCP | [坐标说明](docs/COORDINATES.md) |
| TCP 四模式、标定采样与 JSON 填写 | [TCP 标定指南](docs/TCP_CALIBRATION_GUIDE.md) |
| 源码结构、构建与上传范围 | [开发指南](CONTRIBUTING.md) |
| SSH JSONL 协议 | [桥接协议](desktop/BRIDGE_PROTOCOL.md) |

## Python 示例

```python
from x2ik import Robot, HOME

# 实际运动：客户先自行切换至 URS，并完成现场准备。
with Robot(tcp_mode="gripper") as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
```

`moveJ(..., tcp_mode="hand")` 可顺序覆盖 TCP，关节目标不变，返回位姿以所选 TCP 计算。
七关节目标单位 rad，时长单位 s。双臂同步使用 `moveJ(q_left=..., q_right=...)`。
[最简示例](examples/movej_minimal.py)默认离线打印目标，加 `--execute` 才连接并运动。
机器人需适配的 ROS 2 / AimDK 环境；本仓库不包含 SDK、机器配置、现场标定或 SSH 密钥。

## 构建

```bash
python3 tools/build_runtime.py
python3 -m build --wheel --no-isolation runtime --outdir dist
python3 tools/build_appimage.py --appimagetool /path/to/appimagetool-x86_64.AppImage
python3 tools/build_delivery.py
```

AppImage 构建需要 Linux x86_64、PyQt5、PyInstaller、mksquashfs 和 appimagetool。
产物位于 `dist/`，可交付目录位于 `delivery/`；这些文件不纳入 Git。

`runtime/` 由根目录源码生成，请修改源文件后再构建。仓库保留开发与验收**脚本源码**，
不包含现场测试记录；旧 `x2ik.py` 多命令工具属于内部维护入口，客户集成使用精简模块的 MDI / MoveJ。

MDI 默认只读；3D 图以关节反馈 FK 驱动真实连杆 STL 来源的简化网格，缺资源时回退骨架。
此处不提供 STEP 实体或碰撞规划模型。MoveJ 返回后无后台保持，TCP 不保证走直线。
模块不提供碰撞规划或通用毫米级外部定位保证。不会修改机器人原有 DDS 配置。
