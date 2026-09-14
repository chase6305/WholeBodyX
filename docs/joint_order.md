# 模型与算法库的关节顺序适配

`JointOrder` 用关节名建立模型顺序与算法库顺序之间的一一映射，Python 和独立 C++ 均已实现。
WBC、MPC 可以使用不同排列；任务构造、仿真状态、参考管理、CSV 和浏览器仍使用模型顺序。

例如模型顺序为 `[shoulder, elbow, wrist]`，WBC 使用 `[wrist, shoulder, elbow]`。
初始位置 `[0.1, 0.3, -0.7]` 进入 WBC 时为 `[-0.7, 0.1, 0.3]`，返回命令按逆映射还原。
映射要求两侧非空、名称唯一且集合完全相同，缺失、多余或重复关节会在初始化时被拒绝。
不会按数组长度猜测对应关系，也不会静默丢弃关节。

## 必须一起转换的数据

设 `P` 将模型顺序排列为后端顺序，则 `q_backend = P q_model`。
模型中的任务关系为 `J_model v_model = b`，所以后端必须接收
`J_backend = J_model P.T`，返回速度按 `v_model = P.T v_backend` 恢复。

| 数据 | 转换 |
| --- | --- |
| 状态 `q/v`、关节目标 | 模型顺序 → 后端顺序 |
| 位置上下界、速度和速度变化率限位 | 在构造后端前排列为后端顺序 |
| 任务 Jacobian | 只排列关节列；任务行、目标速度和任务权重保持原意 |
| 速度命令 | 后端顺序 → 模型顺序；时间戳与有效期保持不变 |
| `JointPlan` | 每个位置节点和区间速度都恢复到模型顺序；预测时间轴不变 |
| `QPResult.x` | 保留原求解器的决策变量排列，复制快照，不猜测辅助变量或 slack 的布局 |

`QPResult.x` 是诊断数据；执行使用 `ControlResult.command`，参考使用 `PlanResult.plan`。
特别是 MPC 的 QP 向量可能按时间堆叠，也可能包含其他变量，不能把它直接当作关节向量重排。
这个适配器不转换单位、关节正负方向、零点、Jacobian 坐标系或旋转表示；这些需由具体库的 adapter 处理。

## Python 与后端注册

```python
from wholebodyx import BackendRegistry, JointOrder, ReorderedController, ReorderedPlanner
from wholebodyx.model import PlanarDualArm
from wholebodyx.runner import run

names = PlanarDualArm.joint_names
wbc_order = JointOrder(names, tuple(names[i] for i in (4, 2, 1, 5, 0, 3)))
mpc_order = JointOrder(names, tuple(names[i] for i in (2, 0, 5, 1, 4, 3)))
source = BackendRegistry()
registry = BackendRegistry()
registry.register("my_wbc", controller=lambda limits: ReorderedController(
    source.create_controller("native", wbc_order.limits_to_backend(limits)), wbc_order))
registry.register("my_mpc", planner=lambda limits: ReorderedPlanner(
    source.create_planner("python", mpc_order.limits_to_backend(limits)), mpc_order))

result = run("disturbance_recovery", "my_wbc", "my_mpc", registry=registry)
```

先按后端顺序构造其限位和配置，再用 `ReorderedController` / `ReorderedPlanner` 包装。
工厂每次创建新实例，`reset()` 转发给内部算法。映射对象可以共享，求解器实例由对应循环独立使用。
逐关节的增益、惯量或其他自有参数也需要按实际语义排列；适配器不会检查任意外部配置对象。

Python 中 `to_backend()` / `to_model()` 接收关节向量或最后一轴为关节的二维矩阵，返回独立只读数组。
`task_to_backend()`、`state_to_backend()`、`limits_to_backend()`、`command_to_model()`、
`plan_to_model()` 提供类型明确的转换。控制器/规划器包装只实现公共 `step/solve/reset` 契约，
不将任意外部 QP 的 `formulate()` 自动转换为模型坐标。

适配器拒绝不完整或矛盾的求解结果，例如失败状态携带命令、成功状态没有计划、输出关节维度错误。
合法失败保留状态和原因且不产生执行数据。模型顺序的参考仍需通过
`MPCReferenceManager(planner, model.limits)` 校验；模型顺序命令仍需由执行层检查限位和时间覆盖。

## 独立 C++

```cpp
#include <wholebodyx/joint_order.hpp>

using namespace wholebodyx;
JointOrder order({"shoulder", "elbow", "wrist"}, {"wrist", "shoulder", "elbow"});
JointLimits limits{Vector::Constant(3, -2), Vector::Constant(3, 2), Vector::Ones(3), {}};
ReorderedController controller(
    std::make_unique<KinematicWBC>(order.limits_to_backend(limits)), order);
ReorderedPlanner planner(
    std::make_unique<JointMPC>(order.limits_to_backend(limits)), order);
```

新接口随 Conan 包安装，继续链接 `wholebodyx::wholebodyx`。
C++ `to_backend()` / `to_model()` 操作 `Vector`，矩阵使用
`columns_to_backend()` / `columns_to_model()`；其他类型转换方法与 Python 同名。
C++ 包装器通过 `unique_ptr` 持有算法实例；Python 包装器同样要求单一循环拥有该实例。
Python 可以包住 `NativeWBC/NativeMPC`，转换完成后再进入 C++ 算法，不要求 Python 使用 C++ 映射类。

## 示例与验收

```bash
uv run python examples/python/joint_order_adapter.py
uv run python examples/python/joint_order_adapter.py --scenario coordination
uv run python examples/python/joint_order_adapter.py --backend native --mpc-backend python
uv run --extra viz python examples/python/joint_order_adapter.py --viewer
./build/native/wbx_joint_order_adapter
```

Python 示例同时运行原始顺序和两套不同排列，逐步比较关节位置、关节速度、末端位置与恢复事件。
无界面默认执行 400 步，浏览器显示实际使用重排适配器的场景。
C++ 示例使用不对称限位，执行 250 步，比较全部预测节点和 WBC 命令。

不对称限位、耦合 Jacobian、三关节循环排列用于发现只交换状态却遗漏限位/任务列、
或者误把正向排列当逆向排列的问题。测试同时覆盖缺失/重复关节、错误输出维度、时间戳、
采样、失败与重置、原始决策向量包含辅助变量的情况。

闭环比较阈值为关节位置 `2e-6 rad`、关节速度 `2e-5 rad/s`、末端位置 `1e-6 m`。
同一数学问题的变量顺序可能影响 OSQP 的缩放及数值求解路径。协作场景使用 `1e-6` 正则化，
本次 Python 测量最大关节速度差约 `1.55e-5 rad/s`、最大关节位置差约 `3.09e-7 rad`，
出现在第一步；因此不要求浮点结果逐位相同。单次耦合任务与不对称约束使用更强正则化，
其解比较阈值为 `2e-7`。

## URDF 的后端排列

```bash
uv run --extra urdf python examples/python/urdf_tracking.py \
  --backend-joints wrist_joint shoulder elbow
```

`--backend-joints` 按后端顺序列出 URDF 中全部活动关节名。`q0`、`goal` 仍按
`model.joint_names` 输入；JSON 同时返回 `joints` 和 `backend_joint_order`。
自带 3R URDF 的 Python/native 控制器均验证原始排列和不同排列下的位姿收敛。
用户提供 URDF 时必须填写其实际活动关节名，当前仍限于固定基座、每关节 `nq=nv=1` 的模型。
浮动基座、缺省关节、mimic 关系或把一部分关节锁定的模型不是简单排列，需要专门的状态/模型适配。
