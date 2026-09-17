# V2.1 坐标、单位与 TCP

本说明适用于 [Python MoveJ](MODULE_GUIDE.md) 和 [桌面 MDI](DESKTOP_MDI_GUIDE.md)。

## 坐标系

双臂 TCP 位置、姿态及 MDI 笛卡尔目标都在 `torso_link` 坐标系，两侧共用同一个原点。

| 轴 | 正方向 |
| --- | --- |
| X | 机器人前方 |
| Y | 机器人左方；右肩在 Y < 0，左肩在 Y > 0 |
| Z | 机器人上方 |

坐标轴随躯干转动，并非固定世界坐标。相机、世界或底盘坐标中的目标必须先转换为 torso 坐标。
重力补偿不会自动转换外部目标坐标。

## 单位与七关节顺序

| 数据 | Python MoveJ | 桌面 MDI |
| --- | --- | --- |
| 七关节目标 | rad | deg |
| TCP 位置 | MoveJ 不接收 TCP 目标 | 输入可选 m / mm；反馈显示 m |
| TCP 姿态 | MoveJ 不接收姿态目标 | deg |
| 运动时长、稳定时间 | s | s |

所有单臂目标固定为七个关节，索引从 0 开始：

| 索引 | 关节 |
| --- | --- |
| 0 | 肩 pitch |
| 1 | 肩 roll |
| 2 | 肩 yaw |
| 3 | 肘 |
| 4 | 腕 yaw |
| 5 | 腕 pitch |
| 6 | 腕 roll |

腕部顺序是 yaw → pitch → roll。关节正方向由模型的轴向与右手定则决定，
名称不表示关节始终绕 torso 同名轴转。左右臂分别使用各自模型和限位，
不要把整组角度取负当作另一侧目标。

`HOME = [0.4, 0, 0, -1.2, 0, 0, 0]`，单位 rad，区别于全零关节姿态。
Python 单臂 MoveJ 只选择指定侧；桌面 HOME 始终为双臂实际运动。

## 姿态与相对运动

RPY 为 `[RX, RY, RZ]`，使用固定轴 X-Y-Z 欧拉角：

```text
R = Rz(RZ) @ Ry(RY) @ Rx(RX)
```

`R` 将末端局部向量转换到 torso 坐标。pitch 接近 ±90° 时，欧拉角表示有万向锁。

| 桌面模式 | 关系 |
| --- | --- |
| `rpy` / `pose` 姿态 | 指定绝对 `R_target` |
| `R`，torso 轴旋转增量 | `R_target = ΔR @ R_reference` |
| `t`，TCP 自身轴旋转增量 | `R_target = R_reference @ ΔR` |
| `d`，torso 平移增量 | `p_target = p_reference + Δp_torso` |

参考来自所选侧当前位姿。欧拉角逐分量相加通常不等价于这些矩阵乘法。

## TCP 与误差口径

默认 `none` 的 TCP 位于对应侧 `wrist_roll_link` 原点，平移零、旋转单位阵。
V2.1 可选 `hand`（灵巧手近似杯子抓取中心）、`gripper`（夹爪模型名义中心）或 `custom`。
手预置相对腕平移 `[0.035, 0, -0.150] m`；夹爪 `[0, 0, -0.17608] m`。
这些数值在**本侧腕坐标**中，不能直接加到 torso 位置上；夹爪左右固定旋转不同。
完整旋转参数、标定与文件填写见 [TCP 标定指南](TCP_CALIBRATION_GUIDE.md)。

```text
p_tcp = p_wrist + R_wrist @ translation_m
R_tcp = R_wrist @ rotation_matrix
```

MDI 的 `xyz/pose/d/R/t/rpy` 与反馈显示统一使用所选 TCP。
Python MoveJ 的关节 `q` 和 HOME 不因工具变化，返回的 `position_m/rotation_matrix` 使用所选 TCP。
手和夹爪预置是几何参考点，精确定位须标定实际工具；工具选择不控制手指开合或更新负载。

界面 TCP 由关节反馈经模型 FK 得到，未包含未建模的装配偏差和工具外参误差。
Python `err_max` 是结束瞬间最大关节误差 rad，不是 TCP 毫米误差。
MoveJ 不使用外部 TCP 位置反馈闭环；实际定位精度须通过独立测量验证。
