# 接入自有 WBC / MPC 与统一运行入口

Python 安装包现在包含场景执行、CSV 导出、诊断命令和可选浏览器入口。
同一份 `TrackingSession` 驱动无界面实验与可视化；无需从 `examples/` 导入运行代码。

## 在安装后的环境中运行

```bash
wholebodyx list
wholebodyx run tracking --mode WBC --steps 250
wholebodyx run disturbance_recovery --mpc-backend native --csv /tmp/recovery.csv
wholebodyx doctor --require native urdf viz
wholebodyx view mpc_retargeting --backend python --mpc-backend native
```

源码环境可以加 `uv run`，也可以使用 `python -m wholebodyx`。默认 `tracking` 使用
MPC + WBC；其他场景指定自己的控制链路，传入冲突的 `--mode` 会报错。
`list`、`run` 和 `doctor` 向标准输出写 JSON；运行失败返回非零退出码。
CSV 逐步写入，失败前完成的步骤会保留。`run` 的 `mpc_backend` 表示所选配置；
`mode: WBC` 时不构造、不调用该规划器。

`doctor` 实际执行 Python/native 的小规模 MPC → WBC 求解，并检查
Pinocchio、Viser uPlot 能否导入。缺少可选依赖仍可通过基础检查；使用 `--require`
声明当前应用必须具备的能力，缺失时返回 1。动态库缺失和扩展接口不匹配会保留具体原因。
它不安装依赖、不启动服务器；URDF/viz 检查仅证明导入成功，不等同于真实机器人或浏览器验收。
基础 NumPy/SciPy/OSQP 必须先能正常导入，命令才能启动。

## 后端注册

每个应用拥有自己的 `BackendRegistry`。内置 `python`、`native`，分别提供 WBC 与 MPC。
内置 `sparse` 只提供 MPC，可通过 `--mpc-backend sparse` 配合任一 WBC。
`wholebodyx doctor --require sparse` 会检查稀疏 MPC 与 Python WBC 的组合求解。
内置 `native-sparse` 同样只提供 MPC，使用 C++ 未消元稀疏问题；
可通过 `--mpc-backend native-sparse` 与两种 WBC 组合。
`wholebodyx doctor --require native-sparse` 实际检查该组合，旧扩展缺少此能力时需要重建。
注册名不能重复；新后端可以只提供其中一项。工厂接收 `JointLimits`，每次调用必须创建
独立实例，使求解器工作区、warm start 和重置状态不会跨控制循环共享。

下面的示例可以直接运行，使用现有算法的不同配置：

```python
from functools import partial
from wholebodyx import BackendRegistry, JointMPC, KinematicWBC
from wholebodyx.runner import run

registry = BackendRegistry()
registry.register("my_wbc", controller=partial(KinematicWBC, smoothing=0.02))
registry.register("my_mpc", planner=partial(JointMPC, horizon=20))

metrics = run("disturbance_recovery", backend="my_wbc", mpc_backend="my_mpc",
              registry=registry, csv_path="/tmp/custom.csv")
```

接入已有仓库时，将工厂替换为你自己的 adapter 构造函数。注册本身不加载第三方库；
需要延迟导入时，在工厂内部 import。可选 `check` 是无参数的加载检查函数，失败时抛异常，
供 `registry.status()` 和浏览器过滤选项使用。它应快速返回，避免初始化硬件或打开连接。
没有 `check` 的后端在菜单中视为可选，实际能否创建仍由工厂调用决定。

`TrackingSession` 复制注册表，之后应用注册新名字不影响已有 session。工厂本身不会深拷贝，
因此不要通过闭包返回同一个有状态实例。显式请求的后端加载失败会直接报错，不切换算法。
未指定 `mpc_backend` 时，优先采用 WBC 同名的规划器；仅注册 WBC 的名字默认搭配 Python MPC。
WBC 模式不需要调用任何 MPC 工厂。

`TrackingSession` 是自带场景的运行器；接入其他模型或驱动时使用
[`VelocitySession`](execution.md)，直接注入模型、驱动、控制器、规划器和任务工厂。
浮动基座力矩接口独立于速度级注册表，使用 `DynamicsController` 与 `TorqueControlLoop`。

浏览器也接收同一个注册表，并按 controller/planner 能力分别填充两个菜单：

```python
from wholebodyx.visualization.viewer import ControlViewer

ControlViewer(backend="my_wbc", mpc_backend="my_mpc", registry=registry).run()
```

纯 WBC 模式禁用 MPC 菜单；切换回 MPC + WBC 后启用。具体使用方式见
[custom_backends.py](../examples/python/custom_backends.py)，支持 `--viewer`。
此示例演示注册、配置和运行复用，算法仍是已有 WBC/MPC，不代表已接入你的私有仓库。
CLI 的 `--backend` 提供内置名字，自定义注册通过 Python 应用入口完成。

### 稳定任务模型的可选 Hessian 复用

`KinematicWBC(limits, cache_task_hessian=True)` 适用于 Jacobian 和权重经常保持不变、
目标速度持续变化的任务。它逐项精确比较当前 Jacobian、权重、维度和基础对角项，
只复用最近一批已验证任务的 Hessian；目标、状态、梯度、速度边界和命令检查每周期重算。
即使调用方重新打开数组的写入标志并修改内容，也会重新构造和验证模型。
失败的模型或动态向量不会替换健康缓存。reset 与已有 QP 模板一样保留这份模型，
下一次调用仍须满足精确匹配条件；缓存不保存历史任务链。

```python
registry.register("stable_task_wbc",
                  controller=partial(KinematicWBC, cache_task_hessian=True))
```

该选项默认关闭：较大且稳定的任务矩阵可能受益，Jacobian 持续改变时，比较和快照有额外成本。
新增内存约为一份 Hessian、最近一批 Jacobian 与权重的数值存储，以及 Python 对象开销。
这只影响 Python `KinematicWBC`，不是原生后端选项，也不提供实时调度保证。
可运行 `scripts/benchmark_wbc_hessian_cache.py` 对自身维度做默认 / 开启选项的交错测量，
再决定是否启用。基准会核对每一步的 QP 或完整求解结果一致，包含动态 Jacobian 对照。

## Adapter 必须转换的内容

| 边界 | 当前契约 |
| --- | --- |
| 控制器 | `reset()`；`step(state, tasks, dt) -> ControlResult` |
| 规划器 | `reset()`；`solve(state, goal) -> PlanResult` |
| WBC 成功 | `QPResult("solved", x, ..., violation=...)` 与 `VelocityCommand`；时间戳等于输入状态 |
| MPC 成功 | `QPResult` 与 `JointPlan`；起点、维度、积分一致性和覆盖范围通过参考管理器校验 |
| 失败 | 明确状态和原因，命令/计划为 `None`；不要把旧结果标记为成功 |
| 关节和空间约定 | 按模型关节名建立索引；SI 单位；世界方向的 `[linear; angular]` Jacobian |

关节顺序转换已有 C++ / Python 的 `JointOrder`、`ReorderedController` 和 `ReorderedPlanner`。
它们统一处理状态、限位、任务列、命令和预测节点，支持 WBC 与 MPC 的排列各不相同。
具体使用方式、URDF 参数与数值验收见 [关节映射说明](joint_order.md)。
原始 `QPResult.x` 保留求解器的决策变量排列；执行使用已恢复为模型顺序的 command/plan。

注册表检查方法是否存在；运行层继续检查返回值。场景会拒绝非 `ControlResult`、
错误命令类型、时间戳不一致、无效约束违反数值和积分器无法执行的命令。
MPC 使用 [参考管理模块](runtime.md) 检查计划。后端配置失败时，保留原控制器和原参考；
执行步骤失败时，清除旧预测并报告错误。第三方数值最优性、模型转换正确性仍需单独验收。

会话层对适配器 I/O 异常和取消执行也清除旧参考，并原样抛出异常；取消场景切换时恢复
之前的场景与算法实例。求解或命令校验失败发生在积分提交前，保留原仿真状态；
积分提交后的诊断故障保留已推进的状态和步骤计数，并在下次成功执行时重新建立参考。
`reset()` 先复位外部控制器和规划器，再提交新的仿真状态；组件复位失败或被取消时，
保留原状态、步数与历史，清除旧参考并记录失败原因。修复组件后可以再次调用 `reset()`。
场景切换先构造并复位新组件；任一阶段失败或被取消时恢复原会话绑定，
包括原场景、后端、目标、仿真状态、计数、历史、计划与缓存，原会话可以继续执行。

## 仓库与包的维护边界

| 内容 | 维护方式 |
| --- | --- |
| 自有纯 C++ WBC/MPC | 在各自仓库提供 CMake target / Conan recipe；WholeBodyX adapter 链接这些 target |
| 自有纯 Python WBC/MPC | 在各自仓库提供 `pyproject.toml`；开发时安装到相同 uv 环境，通过工厂构造 adapter |
| C++ 算法供 Python 使用 | 独立绑定包；adapter 只交换数值数组和明确的数据类型 |
| 应用配置和机器人映射 | 应用创建注册表、工厂配置、模型与任务；算法库无需依赖 Viser |

把确定版本的外部依赖写进项目配置并更新对应锁文件；Conan 管 C++ 依赖与构建配置，
uv 管 Python 包环境。注册机制不下载代码，也不替代依赖版本管理。
当前没有填入尚未提供的自有仓库地址或版本。

`TrackingSession` 和五个内置场景仍针对平面双臂积分器。通用 URDF 入口使用同一
`BackendRegistry.create_controller()`，但任务和控制循环按该模型构造；浏览器暂不展示任意 URDF。
动力学 WBC 的力矩/接触接口需要独立契约，不能直接作为速度控制后端注册后执行。
