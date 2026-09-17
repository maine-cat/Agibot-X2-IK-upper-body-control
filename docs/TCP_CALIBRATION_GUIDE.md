# V2.1 TCP 选择、标定与文件填写

MDI 和 MoveJ 使用同一套 TCP 配置。
TCP 是“希望控制或报告位置的工具坐标系”，配置描述它相对于本侧
`left_wrist_roll_link` / `right_wrist_roll_link` 的固定平移与旋转。
按实际安装工具选择 TCP 模式。默认模式为 `none`，TCP 位于腕 roll 连杆原点。

TCP 标定用于确定工具的几何位置与方向。MDI / MoveJ 使用固定重力补偿参数
`40 N·m/rad / ±12° / pelvis`；选择工具不会更新质量、重心、关节零位或补偿参数。
标定由操作者采集数据，离线拟合程序读取采样文件生成配置，不连接或驱动机器人。
执行运动前请自行将机器人切换至 URS 模式；MDI 不提供模式切换。

## 1. TCP 模式与默认参数

以下平移均在**本侧腕 roll 坐标系**中，单位 m，不是在 torso 坐标中直接加数值。

| 界面选项 | `tcp_mode` | 腕到 TCP 平移 | 适用说明 |
| --- | --- | --- | --- |
| 无 | `none` | `[0, 0, 0]` | 默认模式，以腕原点作为目标点 |
| 灵巧手 | `hand` | `[0.035, 0, -0.150]`，左右相同 | 手掌前方的**近似杯子抓取中心**；几何估计，非实测 |
| 夹爪 | `gripper` | `[0, 0, -0.17608]`，左右相同 | **名义抓取中心**；几何参考值，非实测 |
| 自定义 TCP 工具 | `custom` | 从文件分别读取左右侧 | 适用于自己的工具、转接件或经测量修正的抓取点 |

灵巧手与夹爪的默认抓取中心均为近似参考位置，未经实物抓杯标定。
杯径、手指或夹爪的闭合程度、接触位置及转接件会影响实际抓取中心。
需要精确定位时，请测量实际工作点，并使用 `custom` 模式加载标定结果。

旋转矩阵把 TCP 向量转换到对应腕坐标；请按左右侧分别使用以下参数：

```text
none：R = I

hand（左右相同，沿掌参考轴）：
R ≈ [[1,  0,             0           ],
     [0, -0.9999987317, -0.0015926535],
     [0,  0.0015926535, -0.9999987317]]

gripper 左：                 gripper 右：
R = [[ 0, -1,  0],          R = [[0, 1,  0],
     [-1,  0,  0],               [1, 0,  0],
     [ 0,  0, -1]]               [0, 0, -1]]
```

**TCP 模式用于设置工具坐标，不控制手指或夹爪开合，不提供杯子感知或自动抓取。**
HOME 对应固定的一组关节角；切换 TCP 后，同一 HOME 姿态的 TCP 坐标会改变。

## 2. 在 MDI 与 MoveJ 中选择

### 桌面 MDI

连接前选择四种 TCP 模式。自定义模式填**机器人上的绝对路径**，例如
`/path/to/robot/my_tcp.json`；先由维护者把测量文件放到机器人。
桌面不会把本机文件自动上传，也不会把本机路径当成机器人路径。
`--demo` 离线演示时才读取桌面本机文件。
连接期间不能更换 TCP；停止发送并断开，修改后重新连接。连接默认只读。

CLI 对应示例：

```bash
# 只读终端预览，允许反馈连接，不发布运动
python3 -m x2ik mdi --dry --tcp-mode gripper
python3 -m x2ik mdi --dry --tcp-mode custom --tcp-file /path/to/robot/my_tcp.json

# 桌面使用的标准输入/输出会话，启动时只读
python3 -m x2ik mdi --stdio --tcp-mode hand
```

`xyz/pose/d/R/t/rpy` 都解释为当前所选 TCP 的位置与方向；例如 `t` 绕的是工具轴。
程序把工具目标换算为腕目标再做 IK。`j` 和 HOME 是关节目标，不因 TCP 模式改变。
切换 TCP 模式后，请重新核对笛卡尔目标的位置与方向。

### Python MoveJ

以下调用会实际运动；执行前按模块指南完成现场准备：

```python
from x2ik import Robot, HOME

with Robot(tcp_mode="gripper") as robot:
    result = robot.moveJ(HOME, side="right", duration=8, settle=2)
    print(result["position_m"], result["rotation_matrix"])  # torso 中的 TCP 反馈 FK
    print(result["tcp"])  # 本次工具配置、来源与 estimated 标记

    # 可在下一次顺序调用时更换 TCP；不改变 HOME 这组关节目标。
    result = robot.moveJ(HOME, side="left", tcp_mode="hand")

with Robot(tcp_mode="custom", tcp_file="/path/to/robot/my_tcp.json") as robot:
    result = robot.moveJ(HOME, side="right")
```

`Robot` 默认 `tcp_mode="none"`。`moveJ()` 省略覆盖参数时沿用该实例当前选择；
指定新模式后，后续调用继续使用新选择。`tcp_file` 只允许与 `custom` 一起使用。
目标 `q` 始终为七关节 rad，TCP 配置仅影响位姿解释和反馈 FK，不把关节 MoveJ 变为直线 TCP 运动。
双臂结果的 `left/right` 各自包含本侧工具和位姿。

```bash
python3 -m x2ik movej --tcp-mode hand       # 默认只打印，不连接
python3 -m x2ik movej --tcp-mode custom --tcp-file /path/to/robot/my_tcp.json
# 真正执行需操作者显式追加 --execute
```

## 3. 自定义文件逐项填写

配置模板为交付包中的 `examples/tcp_tool.example.json`。
模板的非零偏移仅演示格式，**没有在你的机器人或工具上验证，不能原样当标定结果下发**。

```json
{
  "schema_version": 1,
  "units": "m",
  "tools": {
    "left": {
      "name": "left_my_tool",
      "frame": "left_wrist_roll_link",
      "translation_m": [0.0, 0.0, -0.1],
      "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]],
      "source": "示例；替换为测量方法、工具编号和日期",
      "estimated": true,
      "description": "示例；填写工具工作点及坐标轴定义"
    },
    "right": {
      "name": "right_my_tool",
      "frame": "right_wrist_roll_link",
      "translation_m": [0.0, 0.0, -0.1],
      "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]],
      "source": "示例；替换为测量方法、工具编号和日期",
      "estimated": true,
      "description": "示例；填写工具工作点及坐标轴定义"
    }
  }
}
```

| 字段 | 填写要求 |
| --- | --- |
| `schema_version` | 整数 `1`；这是文件格式版本，不是产品版本 |
| `units` | 固定 `"m"`；100 mm 填 `0.100` |
| `tools.left/right` | 两侧都必须填写；只使用单臂也不省略另一侧 |
| `name` | 非空工具名，最多 200 字符 |
| `frame` | 严格为本侧 `left_wrist_roll_link` 或 `right_wrist_roll_link` |
| `translation_m` | 腕原点指向 TCP 原点的三维向量，在腕坐标中表达 |
| `rotation_matrix` | 3×3 正交、行列式 +1 的矩阵；列分别是 TCP 的 X/Y/Z 轴在腕坐标中的方向 |
| `source` | 可选；测量依据、工具编号和日期，最多 2000 字符 |
| `estimated` | 可选布尔值；几何估计填 `true`，经实测验证的配置填 `false` |
| `description` | 可选；工作点、轴定义与适用条件，最多 2000 字符 |

旋转和平移组成完整的刚体变换：

```text
T_torso_tcp = T_torso_wrist · T_wrist_tcp
p_tcp = p_wrist + R_wrist · translation_m
R_tcp = R_wrist · rotation_matrix

R_wrist_target = R_tcp_target · rotation_matrix.T
p_wrist_target = p_tcp_target − R_wrist_target · translation_m
```

只能在**主动规定 TCP 三轴与腕三轴同向**时填写单位阵；同一点可以定义不同方向，
必须与业务姿态输入一致。JSON 不接受注释、重复键、未知字段、NaN/Infinity 或字符串数值。
矩阵必须是真正旋转，不能用镜像矩阵表示另一侧工具。

## 4. 怎样取得自己的 TCP

### 方法 A：几何测量或 CAD

识别腕 roll 坐标系，测量腕原点到工具工作点的三轴距离；若尺寸来自法兰面，
先把法兰到腕的固定变换合进去，不能把法兰原点当腕原点。
依据安装方向定义工具 X/Y/Z 正方向，形成旋转矩阵。左右安装分别测量。
尺寸估计先标 `estimated=true`，记录适用工具与转接件；正式精度要求使用独立外部测量复核。

### 方法 B：固定点枢轴法，离线求平移

固定工具上的同一个尖点或可重复定位的接触点，让它在多种腕姿态下始终落在同一个外部固定点。
对第 i 次观测，有：

```text
R_i · t + p_i = c
[R_i  −I] · [t, c].T = −p_i
```

`p_i/R_i` 是**腕原点**在同一基准系中的位置/旋转；`t` 是所求腕到 TCP 偏移，
`c` 是外部固定点的位置。堆叠多帧后最小二乘求 6 个未知数。
这只能辨识**平移**，工具旋转必须通过 CAD、轴向测量或外部姿态标定另行提供。
松散接触杯壁、指尖滑动或手指姿态变化不满足同一刚性 TCP 的前提。

最少每臂 6 帧，建议 10–20 个分布良好的腕姿态，绕多个不平行轴改变方向。
只改变位置、不改变姿态，或只绕同一轴旋转，不能完整辨识偏移。
离线工具拒绝秩不足、条件数大于 1000、3D 残差 RMS 大于 1 mm 或最大残差大于 2 mm 的数据。
这些是数据一致性检查门槛，不表示已经达到 1 mm 外部绝对定位精度。

### 最小采集过程：只读腕 FK

1. 先安排并固定外部参考点，确定刚性工具工作点。所有姿态调整由现场人员使用现有、已核验的机器人操作流程完成；本工具不提供自动试探运动。
2. 选择 `none`，保持 MDI 未启用发送。`--stdio --tcp-mode none` 的只读 `state` 包中，`arms.left/right.xyz` 和 `rpy_deg` 是腕原点姿态；不要混入 `hand/gripper/custom` 的已偏移 TCP 数据。
3. 每次同一点可靠接触、机器人稳定后记录同一帧的位置与姿态，记录姿态编号。确认反馈新鲜，并用完整数值而非界面四舍五入的显示文本作拟合。
4. 使用 torso FK 时，采集期间必须保持躯干相对外部参考点固定。若躯干移动，应利用同步外部跟踪把每帧腕姿态统一到固定参考系；只用 torso FK 无法识别躯干移动。
5. 按下一节格式整理数据。FK 含模型/编码器误差；要求较高的实物精度时，使用外部测量腕姿态和独立验证点，不能仅凭拟合残差验收。

RPY 先由 deg 转 rad，再按 `Rz(rz) @ Ry(ry) @ Rx(rx)` 转为矩阵；不要将 RPY 三个数填到 `rotation_matrix`。
取样不需要调用 MoveJ，也不需要发布 `arm`、`mdi` 或 `home` 请求。

## 5. 采样文件与离线运行

下面是**字段完整的合成数据示例，并非任何机器的实测记录**。
为了演示，两臂 TCP 轴主动定义为与腕轴同向，因此 `tcp_rotation_matrix` 使用单位阵。
实际文件中的每帧必须替换为自己测到的腕姿态，旋转也须按实际轴定义填写。

```json
{
  "schema_version": 1,
  "units": "m",
  "samples": {
    "left": [
      {"position_m":[0.38,0.2,0.4],"rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]]},
      {"position_m":[0.38,0.1,0.3],"rotation_matrix":[[1,0,0],[0,0,-1],[0,1,0]]},
      {"position_m":[0.38,0.3,0.3],"rotation_matrix":[[1,0,0],[0,0,1],[0,-1,0]]},
      {"position_m":[0.5,0.2,0.32],"rotation_matrix":[[0,0,1],[0,1,0],[-1,0,0]]},
      {"position_m":[0.3,0.2,0.28],"rotation_matrix":[[0,0,-1],[0,1,0],[1,0,0]]},
      {"position_m":[0.4,0.18,0.4],"rotation_matrix":[[0,-1,0],[1,0,0],[0,0,1]]}
    ],
    "right": [
      {"position_m":[0.42,-0.2,0.4],"rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]]},
      {"position_m":[0.42,-0.3,0.3],"rotation_matrix":[[1,0,0],[0,0,-1],[0,1,0]]},
      {"position_m":[0.42,-0.1,0.3],"rotation_matrix":[[1,0,0],[0,0,1],[0,-1,0]]},
      {"position_m":[0.5,-0.2,0.28],"rotation_matrix":[[0,0,1],[0,1,0],[-1,0,0]]},
      {"position_m":[0.3,-0.2,0.32],"rotation_matrix":[[0,0,-1],[0,1,0],[1,0,0]]},
      {"position_m":[0.4,-0.18,0.4],"rotation_matrix":[[0,-1,0],[1,0,0],[0,0,1]]}
    ]
  },
  "tcp_rotation_matrix": {
    "left": [[1,0,0],[0,1,0],[0,0,1]],
    "right": [[1,0,0],[0,1,0],[0,0,1]]
  }
}
```

安装 Python 与 NumPy 后，在交付包根目录运行离线标定工具：

```bash
python3 source/tools/calibrate_tcp.py --samples wrist_poses.json --output my_tcp.json
```

只标定左臂时，`samples` 与 `tcp_rotation_matrix` 都只保留 `left`，并提供已有双臂配置：

```bash
python3 source/tools/calibrate_tcp.py --samples left_wrist_poses.json \
  --base-file current_tcp.json --output updated_tcp.json
```

另一侧沿用 `current_tcp.json` 的值，不会自动填零。输出文件必须不存在，避免覆盖已有标定。
成功时终端打印各侧 RMS、最大残差、条件数、固定点位置和样本数；输出 JSON 可直接作为 `custom` 文件。
数据不合格则退出且不写配置。残差过大先检查单位、同一点接触、工具刚性、反馈时刻与躯干是否移动；
不要通过放宽门槛把滑动或错误坐标的数据变成标定结果。

最后用未参与拟合的姿态与外部测量复核 TCP；再由现场人员安排经检查的业务点验证。
保存工具编号、左右安装方式、机器人型号、版本、采样与独立验证记录。换工具、转接件或 TCP 定义后重新核验。
