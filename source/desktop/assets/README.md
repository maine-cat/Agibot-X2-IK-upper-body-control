# 双臂显示网格

`arm_meshes.json` 仅用于桌面 MDI 的三维反馈。它不参与 IK、运动指令、碰撞检测、TCP 标定或精度验证。

来源为用户提供的 `omnipicker_omnipicker/x2_ultra_plus_omnipicker_omnipicker.urdf` 及它引用的双臂 14 个 STL。提供的目录中没有 STEP，因此界面称为 **STL 实体**。未引入腿、头、躯干、夹爪或灵巧手指网格。腕 roll 连杆采用该夹爪版 URDF 绑定的 `wrist_roll_extend_link.STL`（约 55 mm 腕端延伸外壳）；它不是工具本体，也不代表其它装配版本的腕端外壳完全一致。末端工具只通过所选 TCP 原点和坐标轴显示，不据此声称已经显示真实工具几何。

- 来源 URDF SHA-256：`d2213f4e1c560ae463f34540d377db29645d7691a14d95b0df724006787bf853`。
- JSON 包含每个源 STL 的文件名、SHA-256、原始面数、简化面数、原始边界边数、简化后的边界边数及包围盒变化。
- 14 个关节的父子顺序、origin xyz/rpy、axis 已与 `x2_ultra.urdf` 逐字段一致性检查；来源 URDF 不替换运动学模型。
- STL 按来源 URDF 的米单位及 `visual/origin` 使用，没有加入毫米缩放。单段外形尺寸约为 35–173 mm。
- 使用二次误差度量的边折叠减面，没有随机删除三角面。为保留外形，将各轴包围盒极值变化控制在 3 mm 以内，并且不增加源模型的单边界边数量。该阈值只是可视化约束，不是表面误差或机器人定位精度保证。源模型原有的开口、非流形结构不作自动修复。
- 保守减面后左臂 11,676 面、右臂 10,488 面。Qt 根据视角剔除背面并按深度绘制。它是简化可视模型，不是 CAD/STEP 原件或遮挡精确的光栅引擎。

实时显示优先使用服务端 `link_transforms`：7 个关节旋转后的子 link 坐标变换，参考系为 `torso_link`。旧反馈未提供该字段时，使用 `q_deg` 和本包校核过的链，并在画布注明“本地模型 FK”。提供了非法变换、缺少资源或反馈过期时不会保留旧实体：降级为有明确标记的骨架或清空画布。TCP 模式及 TCP 标定只改变工具原点/坐标轴，不改变手臂网格变换。

重建（仅开发机，运行时无需这些构建依赖）：

```bash
python3 -m pip install numpy fast-simplification
python3 tools/prepare_arm_meshes.py \
  --urdf /path/to/omnipicker_omnipicker/x2_ultra_plus_omnipicker_omnipicker.urdf
```

所有派生几何仍受原始资产权利约束；提供目录不代表另行授予新的开源许可证。
