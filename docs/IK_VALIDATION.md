# IK 2.1.1 离线验证

2.1.1 的求解改进已并入 `source/x2ik/x2_srs_ik.py`，原 API、MDI 预检、桌面桥接和笛卡尔运动循环使用该求解器，运行包不依赖外部 `ik_validation` 目录。

## 验证结果

2026-09-28 在 Ubuntu x86_64 上完成离线验证，Python 3.10.12 / NumPy 2.2.6 和 Python 3.11.13 / NumPy 1.26.0 分别检查 wheel 与 runtime，每种安装方式均通过 18 项测试。

随包记录：[Python 3.10](../verification/2.1.1-python310.json)、[Python 3.11](../verification/2.1.1-python311.json)，包含安装包 SHA256、源码哈希、测试输出及离线入口检查结果。

| 检查 | 结果 |
| --- | --- |
| 原始双臂回归目标，含左臂四个旧失败样本 | 112/112 通过 |
| 新随机目标，每臂 100 个，随机种子 20260928 | 200/200 通过 |
| none、hand、gripper、自定义旋转和平移工具 | 96/96 通过 |
| 单目标求解器沿连续路径求解，每臂 121 帧 | 242/242 通过，满足指定的 5° 单步上限 |
| 原 track 入口连续跟踪，每臂 121 帧 | 242/242 通过，解析分支不变，单步不超过 0.05 rad，全局恢复未调用 |
| 两臂六方向远不可达目标 | 全部拒绝 |
| 非法输入、残差伪报、错误分支和超限步长 | 按预期拒绝 |
| 原 API、MDI 后台求解、桌面协议预览 | 通过 |
| 原笛卡尔运动循环，内存关节反馈 | 双臂短路径、原位保持、失败后停止推进通过 |
| wheel RECORD、runtime MANIFEST 与维护源码 | 文件内容一致 |
| 安装包的帮助、离线 MoveJ 和桌面协议 demo | 通过 |

回归目标保存在 `source/tests/fixtures/wrist_regression.json`，固定数据记录原始 wheel 的 SHA256，随机目标由独立 URDF 链 FK 生成，生成目标的关节角不传给未知目标求解器，连续路径只传上一帧已接受的解。

每个返回解使用独立 FK 复核，位置残差不超过 0.01 mm、姿态残差不超过 0.001°，并检查关节限位，工具测试同时包含平移和旋转，七轴冗余解不要求还原生成目标的原始关节角。

## 接入行为

当前姿态已满足目标时保留原关节角，单目标求解优先局部精修，再做解析搜索及有限种子恢复，MDI 预检不再仅凭近似肩腕距离拒绝目标，所有成功结果以原始目标和完整 FK 复核为准。

运动循环使用局部跟踪，最多 12 次六维局部迭代后尝试 SEW 邻域搜索，数值候选需能可靠对应解析分支，分支变化、位姿残差超标、关节越限或单步超过 0.05 rad 时拒绝，运动入口不调用全局恢复，连续五帧失败后停止推进。

`IKSolution` 保留 `q`、`psi`、三个分支编号、`in_limits`、`pos_error` 和 `rot_error`，新增 `source` 标记求解路径，原位置误差单位仍为 m，姿态误差仍为 rad，Python `Robot.moveJ()` 的公开接口不变。

## 复现

在交付目录建立独立 Python 环境，按 README 安装 NumPy、setuptools 和 wheel 后执行：

```bash
cd source
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/build_module.py
python tools/verify_release.py --output ../verification/2.1.1-check.json
```

包验证会在临时目录安装 wheel 并解压 runtime，对两种方式分别执行同一组测试，同时核对导入路径和源码哈希，测试不导入 ROS 消息环境、不建立机器人连接，已有结果文件不会被覆盖。

## 验证范围

以上结果只说明模型内求解、程序调用和打包内容通过本次检查，不代表全工作空间成功率或实体定位精度，内存反馈测试没有执行 PhysX、MuJoCo 或真机运动，桌面检查覆盖协议与求解逻辑，未重新构建或启动 AppImage 图形界面。

0.05 rad 是逐帧拒绝阈值，不是按时间计算的关节限速，有限次数恢复和局部迭代不构成硬实时保证，实物 TCP、关节零位、负载、跟踪误差和碰撞约束仍需在对应机器与任务中验证。
