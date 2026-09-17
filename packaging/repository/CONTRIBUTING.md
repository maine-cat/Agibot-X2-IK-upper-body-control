# 开发与构建

产品 V2.1 使用 Python/产物版本 `2.1.0`。本轮 Git 工作分支 `V2.0`，保留原 `main/v1.0`。

## 源码结构

| 路径 | 职责 |
| --- | --- |
| `x2_movej.py` | 公开 `Robot.moveJ()` 薄封装 |
| `x2_mdi.py` / `x2_mdi_bridge.py` | 终端 MDI / 桌面 JSONL 桥接 |
| `x2_api.py` / `x2_sim_ros.py` | 内部运动接口、反馈检查与 ROS 通路 |
| `x2_arm_model.py` / `x2_frames.py` / `x2_arm_dynamics.py` | 模型、坐标和重力相关计算 |
| `x2_tcp.py` / `assets/tcp_presets.json` | 四种工具、完整 SE(3) 校验与预置来源 |
| `tools/calibrate_tcp.py` | 人工采样后的离线 TCP 平移拟合；不连接、不运动 |
| `config/tcp_tool.example.json` | 自定义工具示例，包含演示值，非现场标定 |
| `x2_compensation.py` | 客户 MDI / MoveJ 的内置固定补偿参数 |
| `x2_srs_ik.py` / `x2_srs_batch.py` | SRS + SEW 候选搜索、局部精修与连续跟踪 |
| `desktop/` | 桌面输入、SSH 通信与 3D 显示 |
| `packaging/` / `tools/` | 包入口、文档模板及显式清单构建 |
| `examples/` / `tests/` | 示例及自动化测试源码 |
| `x2_*test.py` / `verify_x2_arm.py` | 开发与现场验证脚本；非客户接口 |
| `docs/` | 客户使用说明与技术介绍 |

根目录兼容工具仍保留内部维护功能；对外运行模块 CLI 只支持 `mdi` / `movej`。
不要把内部工具的旧状态管理或直控命令作为客户使用方法。
客户桌面/终端 MDI 和 `Robot.moveJ()` 不读取 SN 补偿标定文件，统一使用
`40 N·m/rad / 12° / pelvis`；SN 只保留机器标识。旧 `X2Arm` 的默认行为、非 MDI 工程 CLI
及现场验收脚本仍保留按 SN 参数文件的独立流程，不能据此推断客户接口会加载同一文件。

## 环境与验证

机器人侧基于 Python 3.10+、NumPy 和现有 ROS/AimDK 消息环境。开发机可安装 NumPy
运行模型和大部分离线测试；桌面测试另需 PyQt5。wheel 构建需 `build` 和 `setuptools>=61`。

```bash
python3 -m unittest discover -s tests -q
python3 tools/build_runtime.py
python3 -m build --wheel --no-isolation runtime --outdir dist
python3 desktop/x2_mdi_desktop.py --smoke-test
```

测试目录包含单元测试代码，不含实机记录。运行这些单元测试不要求连接机器人。
现场验收脚本存在机器身份、配置和姿态约束，执行前须另行准备；不应通过删除约束使测试继续。
`x2_converge_test.py` 和 `x2_random_reach_test.py` 的在线模式还要求显式设置
`X2_TEST_EXPECTED_SN`，作为本次独立核对后的目标机器人序列号。该值必须与本机
`X2_ROBOT_SN` 和标定文件中的 `sn` 一致；缺失或不匹配会拒绝测试。离线模型检查无需此变量。

AppImage 与交付包构建见根 README。`runtime/`、`dist/`、`delivery/` 均为生成物，不直接编辑。
客户文档源文件在 `packaging/share_docs/`，构建时复制到交付 `docs/`；仓库 `docs/` 为同源发布副本。

## 不纳入仓库的内容

`.gitignore` 排除完整第三方 SDK、构建产物、现场配置、标定、结果、回滚备份、缓存和密钥。
`x2_ultra.urdf` / `x2_ultra.xml` 为求解所需的模型资源，沿用仓库已有版本并保留来源说明；
完整外部 SDK 与原始 mesh 树不在此仓库。桌面额外提供所需双臂 STL 的减面显示资源，
来源及转换信息随资源保留；它们不是 STEP 或碰撞模型。
工具预置提取必要固定变换及估计说明，不复制整套手/夹爪模型，不改变已有手臂几何。
配置示例见 `config/x2ik.conf.example`，部署时创建机器自己的 `x2ik.conf`。

不要提交 `.env`、SSH 私钥/公钥、证书私钥、访问令牌或完整本机目录。
发布前按显式清单导出并审查：`python3 tools/export_source.py --output /tmp/x2ik-source-review`。
此脚本只整理可审阅文件，不包含 Git 提交或推送操作。

本项目不修改机器人原有 DDS 配置；机器人部署需由维护者独立安排。
