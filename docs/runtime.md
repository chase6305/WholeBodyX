# MPC 与 WBC 的参考衔接

`MPCReferenceManager` 负责参考的接收校验、有效期和重规划时机，独立于机器人几何、
仿真与 Viser。它使用现有 `Planner` / `JointPlan` 契约，可以接 Python 或 native MPC；
采样结果可交给任一 WBC。Python 使用 `runtime.py` 同步调度，独立 C++ 程序可使用
`wholebodyx/reference.hpp`。带执行门控的通用组合见 [执行接口](execution.md)。

## 重规划条件

默认重规划间隔为 0.1 s，关节预测偏差阈值为 0.03 rad。偏差取
`max(abs(q_measured - q_plan(t)))`，不包含速度或动力学状态误差。

| 原因 `reason` | 行为 |
| --- | --- |
| `initial` / `forced` | 初次运行或显式失效后，从当前测量状态规划 |
| `goal_changed` | 目标改变，丢弃旧参考并立即规划 |
| `tracking_error` | 关节预测偏差超过阈值，提前规划 |
| `periodic` | 到达上次成功规划时间加重规划间隔 |
| `coverage` | 当前时刻仍有效，但计划不足以覆盖完整控制区间 |
| `expired` | 参考已过期，获取新计划后才能继续 |
| `reused` | 复用有效参考；本周期不调用规划器 |
| `retry` | 示例发生控制错误后，下一次手动运行重新获取参考 |
| `clock_rollback` | 时钟回退，丢弃计划；必须 `reset()` 后恢复 |

周期按状态时间戳判断，容忍本次规划以来多次积分累积的 ULP 舍入差异，
容差上限为四分之一控制周期，避免低分辨率时钟明显提前触发。例如 0.08 s 换目标后，
下一次周期规划为 0.18 s，而不会在全局第 5 个控制步骤的 0.10 s 再次规划。
目标变化、预测偏差和有效期检查仍可提前触发。

`RobotState`、`VelocityCommand`、`JointPlan` 和 `CoMPlan` 的时间标量保存为 Python
`float`；计划采样时也先转换时间参数，避免 NumPy 单精度运算使大时间戳下的有效区间折叠。
转换保留输入已经表示的数值，不能恢复调用方先前丢失的时间精度；无法分辨步长的计划仍会被拒绝。

## 接收自有 MPC 的结果

接入自己的仓库时，adapter 返回 `PlanResult(QPResult(...), JointPlan(...))`。
运行模块在采用新计划前检查：

- 求解状态成功且包含计划，输入数组满足 `JointPlan` 的维度、有限值和积分关系。
- 计划时间戳等于本次传入的测量时间，起始位置匹配测量位置。
- 有效期覆盖整个 `[state.stamp, state.stamp + control_dt]` 控制区间。
- 预测位置、速度及可选速度差满足运行实例的限位。

位置起点及约束检查使用 `1e-6` 的绝对容差。接受时复制计划，避免后续 adapter 工作区变化
影响已接收的参考。目标可为限位外的软目标；预测状态本身仍需要满足硬约束。

一次刷新失败时，`ReferenceUpdate.success` 为假，位置/速度均为空，旧计划被清除。
`reset()` 在调用外部规划器前先使旧参考失效；外部复位抛错时继续保持失效，
时钟回退锁存只在复位成功后清除。取消异常原样传给调用方。
运行模块不自动退回旧轨迹，不会把陈旧或未来时间戳改写成当前时间。
参数错误抛出 `ValueError`；规划器的 `ValueError` / `RuntimeError` 转换成失败结果供运行层处理。

下面演示一次参考更新。连续运行时，每个 WBC 控制周期执行 `update()`，调度器按条件决定是否调用 MPC：

```python
import numpy as np
from wholebodyx import JointMPC, MPCReferenceManager, RobotState
from wholebodyx.model import PlanarDualArm

model = PlanarDualArm()
planner = JointMPC(model.limits)
references = MPCReferenceManager(planner, model.limits,
                                  replan_period=0.1, tracking_tolerance=0.03)
state = RobotState(np.zeros(6), np.zeros(6), 0.0)
update = references.update(state, np.full(6, 0.3), control_dt=0.02)
if not update.success:
    raise RuntimeError(update.solver.detail)
# update.position / update.velocity 用于姿态反馈与速度前馈任务。
```

完整的任务构造、WBC 状态检查和积分执行见
[TrackingSession](../python/wholebodyx/session.py)。WBC 失败时该 session 同样清除参考，
控制步骤不推进仿真时间。`VelocitySession` 将同一参考管理器与 `VelocityControlLoop`
组合，故障时清除参考并通过驱动 `stop()` 锁存故障；规划也纳入周期预算。
真实硬件的停机行为由驱动实现；同步模块没有后台 worker、
超时中断、跨进程参考交换或硬实时保证。

## 可选独立进程规划

同步 `MPCReferenceManager` 保留本页所述刷新策略。需要隔离规划耗时时，可显式选择
`ProcessMPCWorker` 和 `AsyncMPCReferenceManager`，通过 `VelocitySession(reference_manager=...)`
接入相同的控制与命令门控。异步版本要求启动种子、预测未来状态并按原时间戳接续；
目标更新期间允许继续使用仍然有效的旧轨迹，失败时清空参考并锁存故障。
启动顺序、延迟代价、控制周期匹配与可运行例子见 [独立进程关节 MPC](execution.md#独立进程关节-mpc)。

## 速度驱动边界与执行期限

`VelocityControlLoop` 串行执行 `read → task_factory → controller.step → gate → write`。
驱动实现 `read()`、`write(command)` 和 `stop()`；控制器继续使用现有 `Controller` 契约。
`CommandGate` 保存测量状态的独立快照，下发前核对时间、关节位置和速度，
防止相同时间戳下换入不同状态，或任务/后端修改输入数组后仍通过校验。
命令时钟与有效期比较最多容忍 4 ULP，同时限制在控制周期的百万分之一以内，
避免大时间戳的低分辨率将明显偏移或过短的命令判为有效。C++ 使用相同规则。

实际驱动接入时可显式配置执行预算，单位为秒：

```python
from wholebodyx import VelocityControlLoop

# driver、limits、controller、task_factory 由应用提供。
loop = VelocityControlLoop(driver, limits, controller, control_dt=.005,
                           max_cycle_time=.004)
cycle = loop.step(task_factory)
print(cycle.control_ms, cycle.cycle_ms)
```

预算从调用 `read()` 前开始，覆盖读取、任务构造、求解、校验和下发。
超期会锁存故障并调用一次 `stop()`；下发前超期则不写命令，下发阻塞超期则在返回后停机。
即使预算较长，也不允许本次执行耗时超过命令自身的有效区间长度。
`control_ms` 只计控制器调用，新增的 `cycle_ms` 包含整个周期。
默认 `max_cycle_time=None` 保留原同步行为；C++ 保留原四参数构造函数，
可使用第五个正有限秒数配置相同预算。

故障后调用 `reset()` 清除控制器和门控历史，重新建立状态时钟起点。
控制器复位失败也会尝试一次 `stop()` 并锁存故障；成功复位后才允许再次读写驱动。
Python 取消执行时先停机，再原样抛出 `KeyboardInterrupt` / `SystemExit`；
C++ 非标准异常也进入停机路径。
执行预算不能抢占阻塞中的 I/O 或求解器；状态时间戳的时钟映射、传感数据年龄、
周期调度和设备端看门狗仍由驱动/应用负责。

## 状态偏移恢复示例

新增 [disturbance_recovery.py](../examples/python/disturbance_recovery.py)：保持同一关节目标，
在 2.06 s 和 4.14 s 注入关节位置偏移，最大单关节偏移分别为 0.12 rad、0.10 rad。
两次事件都发生在原周期规划点之外，由预测偏差检查触发额外规划。

```bash
uv run python examples/python/disturbance_recovery.py --csv /tmp/recovery.csv
uv run --extra viz python examples/python/disturbance_recovery.py --viewer
PYTHONPATH=build/native uv run python examples/python/mixed_backends.py --scenario disturbance_recovery
```

这是运动学状态跳变，注入时保持时钟与速度不变；没有力、惯量或接触模型。
`JointIntegrator.inject_configuration_offset()` 在提交前检查维度、有限值和位置范围，越界会拒绝，
不会把偏移静默裁剪到限位。默认 8 s 示例中，四种后端组合均应检测两次偏移、触发两次
`tracking_error` 重规划，并恢复至位置误差 `1e-4 m` 以下。

CSV 同时保留控制区间起点 `control_stamp_s` 与区间终点 `time`。
状态偏移写在该控制步骤的 `offset_0_rad...offset_5_rad` 列，因此有
`q[k] - q[k-1] = offset[k] + dt*v[k]`。事件在 2.06 s 注入，对应完成于 2.08 s 的那一行。

## 可视化与诊断

- `tracking_error_rad`：本周期规划前，相对上一份计划的最大关节位置偏差；没有旧参考时为空。
- `plan_age_s` / `plan_remaining_s`：完成本次控制步骤后，参考年龄和剩余时间。
- `replan_reason`：本次更新原因；`disturbances`、`tracking_replans` 为累计次数。
- `solve_ms` / CSV `mpc_ms`：规划调用加接收校验的墙钟耗时；复用时为空。

在定时目标变化的显示边界，旧预测立即失效；不用等下一个 MPC 控制步骤才隐藏。
暂停会保留最新参考诊断，错误信息会保持到成功执行或重置。求解失败时清除排队单步并隐藏旧预测。
误差曲线仍按完成步骤的时间绘制，新目标刚出现时的误差跳变属于正常诊断信息。
