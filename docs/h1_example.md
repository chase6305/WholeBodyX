# Unitree H1：MPC + WBC 可运行示例

新增：[Pinocchio 浮动基座动力学 WBC](h1_dynamics.md)，支持力矩驱动的站立与重心转移。
本页原有动作仍为运动学演示。

## 弯腰拾取并恢复

```bash
.venv/bin/python examples/python/h1_mpc_wbc.py --motion pickup --viewer --port 8081
```

打开 `http://127.0.0.1:8081`，默认自动运行并循环播放。可使用 0.25–1.5 倍播放速度、单步观察、
重新开始。速度只改变播放节奏，不改变控制 dt。蓝点标记虚拟抓取位置，物体拾起后变绿。
机器人保持初始姿态半秒，然后通过
髋、膝、踝协调前俯约 43°，右前臂末端接近台面上的橙色方块。3–4 秒之间，
手端距离物体小于 1.5 cm 时才执行虚拟附着；4 秒起回正，最终持物抬高约 34 cm。
0.5–2.5 秒平滑前俯，2.5–4 秒对准并保持，4–6 秒平滑回正，6–8 秒持物展示。
使用三次平滑曲线，阶段交界目标速度为零；单独 WBC 使用解析速度前馈。
每 8 秒复位机器人与物体并重复播放；取消“循环播放”可在本轮结束后停止。
`--no-autoplay` 启动时暂停，`--no-loop` 只播放一轮。仅演示弯腰可使用 `--motion bow`。

```bash
.venv/bin/python examples/python/h1_mpc_wbc.py --motion pickup --csv /tmp/h1-pickup.csv
.venv/bin/python examples/python/h1_mpc_wbc.py --motion pickup --backend native
```

该 H1 的 torso_joint 只有 yaw，没有腰部 pitch。弯腰通过双侧髋 pitch=-1.0、
膝=0.4375、踝=-0.1875 rad 的目标协调实现；MPC/WBC 与速度/加速度限制生成过渡。
求解器仍是固定基座关节模型，腿部任务跟踪计划位姿。显示与诊断时，以左脚初始位姿
为锚点，根据当前关节 FK 重建骨盆的世界位姿，并检查右脚偏移；并非求解了浮动基座
动力学或双脚硬接触约束。默认幅度 1.25 的 MPC 模式两脚最大位置偏移约 0.18 mm。持续变化的目标会触发 MPC 提前重规划，
本例 8 秒共约 241 次，不能再按静态目标的 80 次计数理解。

物体放在按可达姿态计算的小台面上，不是地面。URDF 没有手指，本例在右前臂
末端 `right_elbow_link` 局部 `(0.27, 0, 0)` 处定义虚拟抓取点。附着时保存相对
位置和朝向，避免物体跳变；附着后刚性随手移动。没有求解摩擦、接触力、负载、
碰撞或抓手闭合，不能据此判断真实 H1 能稳定拾取。

CSV 新增 pelvis_pitch_rad、object_attached、grasp_distance_m、object_lift_m 和
物体坐标；这些坐标使用脚部锚定坐标系，不含浏览器为地面显示额外添加的高度。
JSON 的 solver_fixed_base 始终为 true；base_model 说明展示采用 fixed 或
support_foot_kinematic 或 floor_aligned_kinematic，fixed_pelvis 仅描述最终展示的骨盆是否固定。

Python/native、MPC/WBC 四种组合均通过 400 步拾取回正测试；另覆盖远离物体
不附着、附着瞬间位置连续和 reset 恢复物体。

入口：`examples/python/h1_mpc_wbc.py`。默认读取相邻仓库
`../HumanoidAssets/UnitreeH1/robot.urdf`，也可通过 `--urdf` 指定路径。
本仓库不复制 H1 URDF 或 GLB 资产。

## 动作幅度与原地踏步

`--amplitude` 控制弯腰、拾取和踏步幅度，默认 1.25，支持 0.5–1.4。
例如 `--motion pickup --amplitude 1.4 --viewer` 前俯目标约 48°；
物体与台面位置随目标姿态重新计算。幅度不改变关节速度和加速度限制。

```bash
.venv/bin/python examples/python/h1_mpc_wbc.py --motion march --viewer --port 8081
.venv/bin/python examples/python/h1_mpc_wbc.py --motion march --csv /tmp/h1-march.csv
```

全身迈步每轮 8 秒，运动阶段约每秒交替一步，最后收步回正。
双腿同时前后配合，包含支撑腿屈膝、踝姿态补偿、髋侧摆、躯干转动、对侧摆臂和屈肘。
默认 MPC 实测抬脚约 11 cm，双脚最大前后间距约 53 cm；这不是实际向前行走的步长。
骨盆显示高度根据较低脚位对齐地面，侧向位置随双脚几何位置调整。
没有前进位移、重心规划、接触力或动态平衡，地面对齐不代表满足物理接触条件。
循环默认开启，仍可暂停、调播放速度或单步观察。

踏步模式速度上限为 URDF 值与 4 rad/s 的较小值，加速度上限为 24 rad/s²，
仅为运动学演示参数。其他动作仍为 0.6 rad/s 和 2 rad/s²。
当前 MPC 使用关节积分模型追踪即时目标，快速周期运动存在明显滞后：
默认 MPC 最大脚位跟踪误差约 14.5 cm，直接 WBC 约 3.5 cm。
可用 `--mode wbc` 对比直接带前馈跟踪；两种模式都不是动力学控制。

CSV 的 left_foot_lift_m/right_foot_lift_m 表示双脚相对初始位置的高度；
left_foot_x_m/right_foot_x_m 和 pelvis_y_m/pelvis_z_m 用于观察迈步和骨盆配合。
该模式 foot_error_m 是相对运动目标（含目标骨盆变换）的位置跟踪误差，
其他模式则是相对初始脚位的偏移。

后续可用 MuJoCo 替换关节积分仿真，逐步加入浮动基座状态、地面接触、
力矩控制与平衡约束，再验证真正的行走与带负载拾取；当前没有接入 MuJoCo。

## 浏览器运行

在 WholeBodyX 根目录、安装 `urdf` 和 `viz` 可选依赖后运行：

```bash
.venv/bin/python examples/python/h1_mpc_wbc.py --viewer --port 8081
```

打开 `http://127.0.0.1:8081`，默认自动运行并循环播放。使用 URDF 中的原始 GLB 网格，
0–2 秒第一组上肢目标，2–4 秒第二组目标，4 秒后回到零位。
每 8 秒自动重复，可关闭“循环播放”。取消勾选可暂停，Ctrl+C 关闭服务。
远程主机可使用 SSH 转发：`ssh -L 8081:127.0.0.1:8081 user@host`。

## 无界面及后端对比

```bash
# Python MPC → WBC，8 秒，保存全部 19 关节状态
.venv/bin/python examples/python/h1_mpc_wbc.py --csv /tmp/h1-mpc.csv

# WBC 直接跟踪目标
.venv/bin/python examples/python/h1_mpc_wbc.py --mode wbc --csv /tmp/h1-wbc.csv

# 已安装 native 扩展时使用 C++ 后端，也可分别指定 WBC/MPC
.venv/bin/python examples/python/h1_mpc_wbc.py --backend native
.venv/bin/python examples/python/h1_mpc_wbc.py --backend python --mpc-backend native
```

`--steps` 和 `--csv` 仅用于无界面运行；`--mode mpc` 表示 MPC + WBC。
CSV 包含仿真时间、关节误差、脚部位置误差、WBC 求解耗时、重规划原因、
约束违反以及所有关节的 q/v。自定义 URDF 路径需指向具有相同命名的 19 关节 H1。

## 控制链路与适用范围

- 固定基座 Pinocchio 求解模型；弯腰/拾取额外重建展示用骨盆位姿；19 个关节按 URDF 模型顺序排列。
- MPC：19 关节积分模型、15 步预测、0.1 秒步长，参考管理器按 10 Hz 重规划。
- WBC：50 Hz 速度级加权 QP；关节参考、两臂 elbow_link 位姿及脚部 ankle_link 位姿任务。
  MPC 模式将计划速度经 Jacobian 转为末端速度前馈。
- 保留 URDF 位置限位；速度上限取 URDF 值与 0.6 rad/s 的较小值，
  加速度采用示例设定 2 rad/s²；march 使用上述单独限制，均非厂家提供参数。
- arms 模式腿部和躯干目标为零，运动发生在双臂。脚部是软位姿任务；不是接触约束。
  显示时整体抬高骨盆以便观看，地面是视觉辅助。

这是运动学示例，不含浮动基座平衡、接触力、力矩输出、碰撞约束或行走。
它没有连接硬件。不要将此处速度积分仿真视为 H1 真实动力学验证。

本机 400 步验证：Python MPC + WBC 最终最大关节误差约 `9.3e-14 rad`，
WBC 约 `3.5e-7 rad`；脚部位置偏移为零。Python/native MPC 均完成 80 次重规划，
最大约束违反低于 `1e-6`。外部资产不存在时相关测试自动跳过。
