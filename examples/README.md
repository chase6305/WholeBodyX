# 可运行示例

**Unitree H1：** 已增加基于 `HumanoidAssets/UnitreeH1/robot.urdf` 的固定骨盆
MPC + WBC 示例，支持原始网格浏览器显示、WBC 单独运行和 CSV 导出。
见 [H1 运行指南](../docs/h1_example.md)。

```bash
.venv/bin/python examples/python/h1_mpc_wbc.py --viewer --port 8081
# 弯腰拾取台面物体，然后回正持物
.venv/bin/python examples/python/h1_mpc_wbc.py --motion pickup --viewer --port 8081
# 原地踏步运动学展示；默认自动运行、循环播放
.venv/bin/python examples/python/h1_mpc_wbc.py --motion march --viewer --port 8081
```

从仓库根目录执行以下命令。基础环境使用 `uv sync --locked`；浏览器示例增加
`--extra viz`，URDF 示例增加 `--extra urdf`。算法运行与数据导出不需要浏览器。

## 学习顺序与验收点

| 顺序 | Python 入口 | 学习内容 | 默认运行与检查点 |
| --- | --- | --- | --- |
| 1 | [dual_arm_tracking.py](python/dual_arm_tracking.py) | 双手位姿 + 弱关节姿态，WBC 或 MPC → WBC | 5 s；可达静态目标收敛 |
| 2 | [task_weight_tradeoff.py](python/task_weight_tradeoff.py) | 冲突任务的加权最小二乘 | 4 组权重；求解结果对照解析解 |
| 3 | [joint_limit_handling.py](python/joint_limit_handling.py) | 软目标不可达、硬约束不可行、求解器恢复 | 目标 2 rad、限位 1 rad；无效状态不产生命令 |
| 4 | [cartesian_circle.py](python/cartesian_circle.py) | 连续末端轨迹、位姿反馈和速度前馈 | 8 s 一圈；保持末端朝向 |
| 5 | [dual_arm_coordination.py](python/dual_arm_coordination.py) | 双手中心移动 + 世界坐标下相对位置任务 | 8 s；检查相对位置误差 |
| 6 | [mpc_retargeting.py](python/mpc_retargeting.py) | 10 Hz MPC、50 Hz WBC、在线换目标 | 8 s；在 2 s、4 s 更新目标并重规划 |
| 7 | [mixed_backends.py](python/mixed_backends.py) | Python/C++ 的 WBC 与 MPC 自由组合 | 完整运行 4 种组合，对比末端误差 |
| 8 | [urdf_tracking.py](python/urdf_tracking.py) | Pinocchio FK/Jacobian → 通用任务 → WBC | 自带 3R URDF；5 s 位姿跟踪 |
| 9 | [disturbance_recovery.py](python/disturbance_recovery.py) | 状态偏移、预测偏差检测、提前重规划 | 8 s；2.06 s、4.14 s 偏移后恢复 |
| 10 | [custom_backends.py](python/custom_backends.py) | 独立注册 WBC/MPC 工厂，共用运行器与 GUI | 8 s；使用自定义参数，检查两次偏移恢复 |
| 11 | [joint_order_adapter.py](python/joint_order_adapter.py) | 模型、WBC、MPC 使用不同关节排列 | 8 s；逐步比较转换前后的关节/末端轨迹和恢复事件 |

第 4–6、9 个示例支持 `--viewer`、`--steps`、`--csv`、`--backend`、`--mpc-backend`。
`--steps` 和 `--csv` 用于无界面运行。其他示例的参数以各自 `--help` 为准。
返回值为 JSON；求解失败会中止并以非零退出码退出。限位示例中的预期不可行状态属于正常验收流程。

```bash
uv run python examples/python/task_weight_tradeoff.py
uv run python examples/python/joint_limit_handling.py
uv run python examples/python/cartesian_circle.py --csv /tmp/wholebodyx-circle.csv
uv run python examples/python/dual_arm_coordination.py --csv /tmp/wholebodyx-coordination.csv
uv run python examples/python/mpc_retargeting.py --csv /tmp/wholebodyx-mpc.csv
uv run python examples/python/disturbance_recovery.py --csv /tmp/wholebodyx-recovery.csv
uv run --extra urdf python examples/python/urdf_tracking.py
uv run python examples/python/custom_backends.py --csv /tmp/wholebodyx-custom.csv
uv run python examples/python/joint_order_adapter.py --backend native --mpc-backend python
uv run python examples/python/joint_order_adapter.py --scenario coordination
```

安装 wheel 后，可以通过 `wholebodyx list` 列出五个内置场景，
`wholebodyx run coordination --csv /tmp/coordination.csv` 在任意目录运行。
自定义工厂的注册方式和命令诊断见 [接入说明](../docs/integration.md)。

CSV 保存每一步的时间、关节位置/速度、名义关节目标、实际/目标末端位置、位置与朝向误差、
相对位置误差、求解耗时和约束违反。`goal_0...5` 是名义关节目标；连续轨迹的即时目标由
`target_0_x_m...target_1_z_m` 给出。没有 MPC 求解的周期，其耗时列留空。
通用场景的 `final_position_error_m` 是两手中较大的误差；原始 `dual_arm_tracking.py` 使用两手误差之和。
CSV 逐步写入，JSON 误差指标在线累计，不把所有步骤积攒在内存中。
重规划时间列表最多保留最近 300 条，超过时 `replan_history_truncated` 为 `true`；总次数仍完整累计。
发生求解异常时，已完成的 CSV 步骤仍可用于诊断。
另外记录重规划原因、计划剩余时间和预测偏差。状态偏移场景用 `offset_*_rad` 解释位置跳变，
`control_stamp_s` 为注入偏移及下发命令的时刻，`time` 为该控制步骤完成时刻。

## 浏览器演示

```bash
uv run --extra viz python examples/python/cartesian_circle.py --viewer
uv run --extra viz python examples/python/dual_arm_coordination.py --viewer --port 8081
uv run --extra viz python examples/python/mpc_retargeting.py --viewer --port 8082
```

也可用一个入口，在“示例场景”菜单切换：

```bash
uv run --extra viz python -m wholebodyx.visualization --scenario coordination
```

切换场景会重置状态、时间和曲线，并选用场景所需的控制链路。圆轨迹和协作示例使用 WBC，
在线换目标示例使用 MPC + WBC；`tracking` 场景允许手动切换链路。
状态偏移恢复也使用 MPC + WBC，偏差超过 0.03 rad 时提前重规划。
橙色标记跟随即时末端参考移动；圆轨迹/协作场景不绘制未经 IK 求解的“目标机器人”。

WBC 和 MPC 后端可分别选择。使用源码构建的 native 扩展：

```bash
bash scripts/build.sh --python
PYTHONPATH=build/native uv run python examples/python/mixed_backends.py
PYTHONPATH=build/native uv run python examples/python/mixed_backends.py --scenario disturbance_recovery
PYTHONPATH=build/native uv run --extra viz python -m wholebodyx.visualization \
  --scenario mpc_retargeting --backend python --mpc-backend native
```

## 算法配方

动态场景的参考生成与任务组合集中在 [experiments.py](../python/wholebodyx/experiments.py)，
无界面运行和 viewer 共用 [TrackingSession](../python/wholebodyx/session.py)。
[runner.py](../python/wholebodyx/runner.py) 负责运行和导出，示例脚本保留简短命令行入口。

**圆轨迹**：半径 0.025 m、周期 8 s，在初始末端附近向内画圆。使用
`v_task = v_reference + K * (p_reference - p)`，朝向误差用世界坐标系下的旋转向量表示。
`FramePoseTask.feedforward` 的顺序是 `[vx, vy, vz, wx, wy, wz]`，单位 m/s、rad/s。
这两条平面 3R 链实际只控制平面位置和绕 z 轴朝向，不能用于验证任意空间 6D 运动。

**双臂协作**：任务一跟踪双手中心，Jacobian 为 `(J_left + J_right) / 2`；任务二跟踪
`p_right - p_left`，Jacobian 为 `J_right - J_left`，权重 20。两个量都在世界坐标系表达。
弱姿态任务用于调节冗余自由度。这个配方不约束末端朝向，不包含物体动力学或刚性抓取约束。
朝向误差仍作为诊断输出，并通过 `orientation_controlled: false` 标明。

**在线 MPC**：保持关节积分预测模型与硬位置/速度/速度变化约束，输出有效期内的
`JointPlan`。WBC 对采样位置做反馈并加入计划速度前馈。目标变化会立即触发重规划，
并清除旧的可视化预测，防止暂停改目标后仍显示旧计划。
调度按上次成功规划的时间计算，计划不足以覆盖完整控制周期时也会立即刷新。
接收校验与失败处理见 [运行模块说明](../docs/runtime.md)。

**状态偏移恢复**：在关节积分器上施加两次位置偏移，保持时钟/速度不变，用于验证参考反馈衔接。
偏移后根据最新测量位置重新规划；此场景不含外力或动力学。默认执行产生 2 次预测偏差重规划。

**自定义后端**：`smooth_wbc` 使用平滑权重 0.02，`long_mpc` 使用 20 步预测和控制权重 0.2。
两个名称分别只注册一种角色，使用已有算法验证工厂接入方式。支持 `--viewer`，菜单中会出现
对应自定义名称；接自己的仓库时替换工厂构造函数。

**权重冲突**：两个任务分别要求同一关节速度为 +1 和 −1。正任务权重为 `w`，
负任务权重为 1，正则项为 0.1，解析解为 `(w - 1) / (w + 1 + 0.1)`。
提高权重会减小该任务残差，但有限权重不提供严格层级优先级。

**限位与失败**：软目标越界时，最优解可以停在边界；硬状态/速度约束相互矛盾时，
返回 `infeasible` 且不输出命令。示例将故障状态单独注入求解器，随后用有效状态验证恢复，
没有把测量状态裁剪后当作真实状态执行。实际硬件的停机/降级策略需要由运行层实现。

## 独立 C++

以下程序不需要 Python、Viser 或 Pinocchio：

```bash
bash scripts/build.sh
./build/native/wbx_joint_control
./build/native/wbx_task_weight_tradeoff
./build/native/wbx_joint_limit_handling
./build/native/wbx_mpc_retargeting
./build/native/wbx_joint_order_adapter
ctest --test-dir build/native --output-on-failure
```

| 入口 | 验证 |
| --- | --- |
| [joint_control.cpp](cpp/joint_control.cpp) | 6 关节 MPC → WBC 静态收敛 |
| [task_weight_tradeoff.cpp](cpp/task_weight_tradeoff.cpp) | 四组加权解与解析结果一致 |
| [joint_limit_handling.cpp](cpp/joint_limit_handling.cpp) | 边界收敛、失败不发命令、恢复 |
| [mpc_retargeting.cpp](cpp/mpc_retargeting.cpp) | 2.06 s、4.14 s 非周期换目标；总计 82 次重规划 |
| [joint_order_adapter.cpp](cpp/joint_order_adapter.cpp) | 不同 WBC/MPC 排列、不对称限位、完整预测节点与命令比较 |

这些可执行程序都有数值验收条件，失败返回非零；构建时启用 `BUILD_TESTING` 后自动加入 CTest。

## 替换自己的 URDF

```bash
uv run --extra urdf python examples/python/urdf_tracking.py \
  --urdf /path/to/robot.urdf --frame tool \
  --q0 0.2 0.6 -0.4 --goal 0.5 0.4 -0.3
```

关节数组使用 `model.joint_names` 的顺序。当前 adapter 支持固定基座、单自由度欧氏关节和有限限位。
脚本允许指定初始姿态和可达关节目标；任意 URDF 的局部 IK 收敛需要独立检查。
自带 [planar_arm.urdf](models/planar_arm.urdf) 只有运动学数据，不适合用于动力学验证。
这个 URDF 示例暂时只输出数值，浏览器仍展示内置平面双臂。

可用 `--backend-joints wrist_joint shoulder elbow` 指定自带 URDF 的另一套后端排列；
初始姿态/目标仍按模型顺序输入。C++/Python 映射、注册方式和不同排列的浏览器示例见
[关节顺序说明](../docs/joint_order.md)。第 11 个示例支持 `--viewer`，同样使用内置双臂几何。

## 后续示例设计

下列场景需要新的算法或运行组件，目前没有放置占位脚本：

| 场景 | 先补齐的功能 | 计划验收 |
| --- | --- | --- |
| 双臂搬运刚性物体 | 相对位姿/闭链约束、物体模型、碰撞约束 | 闭链误差与碰撞距离 |
| 重力补偿与末端受力 | 逆动力学 WBC、力矩接口、动力学仿真 | 动力学残差、力矩边界、受力跟踪 |
| 推扰恢复 | 状态估计/扰动注入、接触动力学 | 恢复时间、接触可行性 |
| 人形双足站立与接触切换 | 浮动基座状态、摩擦锥、接触调度 | 接触滑移、质心和角动量误差 |
| 质心 MPC → 动力学 WBC | 质心预测模型、力参考和多频率运行层 | 预测可行性、超时/失效后的处理 |

先通过现有场景验证自有控制器适配，再按上述依赖顺序增加新的动力学示例。

MPC 参数更新性能可通过 [benchmark_mpc.py](../scripts/benchmark_mpc.py) 测量，
测试协议、初始化开销和运行边界见 [性能记录](../docs/performance.md)。

### H1 动力学站立与重心转移

```bash
.venv/bin/python examples/python/h1_dynamics_wbc.py --viewer --port 8081
```

浮动基座、力矩级 WBC、固定双脚接触正向动力学。
加 `--controller mpc` 使用质心/ZMP 预览 MPC，加 `--push-force 3` 每轮施加短时侧向外力。默认自动循环；
[运行指南与适用范围](../docs/h1_dynamics.md)。
