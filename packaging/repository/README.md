# X2 双臂运动控制：MDI 与 MoveJ

面向 X2 上肢调试和业务集成的 Python 模块。人工调试使用 Linux AppImage 桌面 MDI，业务程序使用 `Robot.moveJ()`。两者共用模型、运动生成、URS 与反馈检查。

**仅在已处于 URS（`UPPERBODY_REMOTE_SPLIT`）的机器人上支持运动。客户必须通过机器人原有操作工具自行切换到 URS。MDI 和 MoveJ 不带模式切换功能。**

## 文档入口

| 需要了解 | 文档 |
| --- | --- |
| 整体架构、接口、逆运动学原理和能力边界 | [项目介绍](docs/PROJECT_OVERVIEW.md) |
| 模块安装和 Python MoveJ | [模块指南](docs/MODULE_GUIDE.md) |
| 桌面连接、3D 反馈、MDI 输入 | [桌面 MDI](docs/DESKTOP_MDI_GUIDE.md) |
| 坐标、关节顺序、单位与 TCP | [坐标说明](docs/COORDINATES.md) |
| 源码结构、构建与上传范围 | [开发指南](CONTRIBUTING.md) |
| SSH JSONL 协议 | [桥接协议](desktop/BRIDGE_PROTOCOL.md) |

## Python 示例

```python
from x2ik import Robot, HOME

# 实际运动：客户先自行切换至 URS，并完成现场准备。
with Robot() as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
```

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

MDI 默认只读；3D 图来自关节反馈的 FK。MoveJ 返回后无后台保持，TCP 不保证走直线。
模块不提供碰撞规划或通用毫米级外部定位保证。不会修改机器人原有 DDS 配置。
