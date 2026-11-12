# H1：Pinocchio 浮动基座动力学、质心 MPC 与 WBC

```bash
.venv/bin/python examples/python/h1_dynamics_wbc.py --viewer --port 8081
```

打开 `http://127.0.0.1:8081`，默认自动开始并循环展示双脚站立下的左右重心转移。
绿色标记为实际质心，橙色标记为目标，地面小球显示二者投影。
蓝框显示假设足底接触范围，绿色竖线显示支撑力（500 N/m）。
面板显示双脚法向力、最大关节力矩、质心与接触位置误差；可关闭受力显示。
取消“运行”暂停，取消“循环播放”在本轮结束停止；支持 `--no-autoplay` 和 `--no-loop`。
循环跨过 8 秒边界时保留机器人状态、速度和控制器，连续积分，不再自动复位。
支持 0.25–2 倍播放速度、单步 5 ms 和“复位并暂停”；播放速度不改变物理积分步长。
关闭循环在下一轮边界暂停；再次运行从当前状态继续。
需要与原 H1 示例相同的 `urdf`、`viz` 可选依赖和外部 H1 资产。

```bash
# 固定目标站立，默认运行 1600 步（8 秒）
.venv/bin/python examples/python/h1_dynamics_wbc.py --motion stand
# 左右重心转移，CSV 保存接触力、力矩与误差
.venv/bin/python examples/python/h1_dynamics_wbc.py --csv /tmp/h1-dynamics.csv
# 目标侧向幅度，单位为米，范围 0–0.05
.venv/bin/python examples/python/h1_dynamics_wbc.py --amplitude 0.025 --viewer
```

## 质心预览 MPC 与轻推

```bash
# 10 Hz 质心/ZMP MPC → 200 Hz 力矩 WBC
.venv/bin/python examples/python/h1_dynamics_wbc.py --controller mpc --viewer
# 每轮第 2 秒在骨盆原点施加世界 Y 方向 3 N 外力，持续 0.2 秒
.venv/bin/python examples/python/h1_dynamics_wbc.py --controller mpc --push-force 3 --csv /tmp/h1-mpc-push.csv
```

默认 `--controller wbc` 保留解析目标跟踪。`--controller mpc` 增加恒定质心高度、忽略
角动量变化的线性倒立摆近似：`ZMP = CoM_xy - height / g * CoM_acceleration_xy`。
优化变量为分段恒定的二维质心 jerk，预测 20 步、每步 0.1 秒，包含速度、加速度、jerk 限制。
双脚矩形接触区的凸包提供 ZMP 支撑范围，向内保留 5 mm 裕量；
用多项式 Bernstein 控制点约束保证区间内部的 ZMP 与速度也满足限制，
并要求预测末端的捕获点落在支撑范围内。这是简化模型的可行性约束，不是全身稳定性证明。
紫色显示规划 ZMP 与未来 2 秒预览，浅蓝色显示双脚支撑多边形。

`CoMPreviewMPC` 默认启用 `unconstrained_fast_path=True`：若固定严格凸代价的无约束
极小值满足全部 QP 边界及独立轨迹验收，直接返回该最优计划；否则调用 OSQP。
可在构造时设置 `unconstrained_fast_path=False`，用于活动约束较多的负载或性能对照。
显式替换求解器及其求解方法仍会被正常调用。reset 清除计划和求解历史，保留固定代价分解。

MPC 每次用当前质心状态重规划，再将位置、速度和加速度参考交给全身 WBC。
位置和速度使用实际反馈；加速度作为参考状态，沿用上一段规划在当前时刻的值，
避免每 0.1 s 被实际加速度覆盖而产生参考跳变。若该初值无法通过 MPC 可行性检查，
仅回退一次，使用实际加速度重新规划；仍失败则暂停。CSV/JSON 的
`mpc_continuity_fallbacks` 和面板记录回退次数，复位清零。
这保证正常重规划时的参考加速度连续，不保证 WBC 实际加速度、支撑力或参考位置/速度连续。
它优先保持简化模型的支撑可行性，当前轨迹跟踪比直接 WBC 保守，不代表误差必然更小。
尚没有全身非线性 MPC、接触切换或落脚点优化。

`scripts/report_h1_dynamics.py` 比较两个 CSV 前，会检查采样时间、目标轨迹和实际施加的
外力是否一致；不同实验会报错，避免误比。HTML 与 JSON 同时报告质心 RMSE、
单步法向支撑力最大变化和变化 RMS，以及采样间隔范围。
支撑力变化按全部原始样本统计，包括轻推启停；图形抽样不参与这些指标计算。

界面“侧向轻推 3 N · 0.2 s”可在运行中施加一次外力；暂停时点击会在继续运行后施加。
外力只进入正向动力学，控制器在下一步根据测得状态反馈响应，不直接获知外力。
轻推模式在控制器中预留 50% 接触裕量，并保持到复位；实际接触力仍按原物理界限验证。
`--push-force` 接受 0–5 N；建议 3 N。更大重心幅度与 5 N 组合可能超出支撑条件并暂停。
这不是通用抗扰保证，也没有模拟外力导致的滑移或抬脚。
CSV 的 `applied_push_y_n` 是前一个积分区间实际施加的外力；有外力时，
WBC 预测加速度与正向动力学加速度可以不同，应查看包含外力的 `forward_dynamics_residual`。

## 实际实现

- Pinocchio free-flyer：26 维配置（含四元数）、25 维速度，19 个执行器。
- CRBA 质量矩阵和非线性项，包括重力与速度相关项。
- WBC 优化变量是广义加速度、19 个关节力矩、两个足底局部六维接触力/力矩。
- 硬约束包括浮动基座动力学、双脚接触加速度、URDF 力矩/位置限位、示例速度及加速度限制。
- 足底采用耦合矩形接触力锥：四个角点的保守摩擦棱锥 `|fx| + |fy| <= 0.6 fz`
  合成足底六维受力可行域，同时约束剪切力、压力中心和扭转力矩。额外保留 `|mz| <= 0.015 fz` 的示例上限。
- 软任务包括带解析速度/加速度前馈的质心跟踪、姿态保持、骨盆朝向及左右脚载荷分配。
- 控制周期 5 ms。将求解出的关节力矩送入独立的约束正向动力学 KKT 求解，
  然后积分加速度，并用 Pinocchio 的流形积分更新浮动基座配置；没有直接回放目标关节角。

双脚初始微屈膝、各外转约 4.6°，避免平行对称姿态的接触 Jacobian 退化。
足底平面使用 ankle_link 下方 7 cm，接触矩形半长 10 cm、半宽 3.5 cm；
摩擦、接触面积、扭转约束和加速度限制都是示例假设，不是厂家标定参数。
KKT 求解采用秩判定，处理冗余约束；接触力在冗余情况下可能不唯一。
每步还检查正向动力学实际求得的接触力是否满足相同的摩擦、压力中心和扭转约束；
若不满足则在积分前报错，浏览器暂停，不仅检查控制器规划的接触力。
当前示例使用非退化站姿。力锥由 16 条角点摩擦射线的归一化凸包构造，
约束剪切力、压力中心和扭转力矩的耦合；测试使用独立角点力线性规划验证其可实现性。
这仍是矩形刚性足底与线性化摩擦假设，没有接触顺应性或真实足底压力建模。

## 动力学接口

`FloatingBaseDynamics.terms(state)` 返回 `DynamicsTerms`，可通过 `terms.mass`、
`terms.contact_jacobian`、`terms.com`、`terms.com_jacobian` 等字段访问。
原来的九项元组解包、切片和下标访问保持兼容；结果数组与后续 Pinocchio 工作区计算隔离。
浮动基座时间戳要求有限且非负；无法表示时间推进的积分区间会被拒绝。
积分时先将时间戳和步长转为 Python `float`，支持 NumPy 单精度时间参数。
计算产生非有限速度、位移或配置时抛出 `ValueError`，保留传入状态；正常积分不增加幅值裁剪。
前向动力学也显式拒绝非有限解和残差，避免数值溢出产生的 NaN 绕过残差比较；
仍使用原有的秩感知求解和 `1e-5` 残差阈值。

`ContactWBC.formulate(state, target_com, dt, target_velocity=..., target_acceleration=...)`
返回 `ContactFormulation(problem, terms, contact_rhs)`，其中 QP 使用物理单位，
不含内部数值缩放。调用它不会推进 warm start、创建求解器或更新求解计数，
返回的 QP 是独立快照，可用于检查问题或接入其他求解器。
`solve()` 返回 `DynamicsControlResult`，支持 `result.command`、`result.acceleration` 等字段。
原有键读取、`get()` 和 `dict(result)` 保留；结果是只读 Mapping，不再是可修改字典。
`DynamicsState`、命令和结果的加速度/wrench/RHS 均为独立只读快照；更新状态时构造新对象。
`DynamicsTerms` 保持原有独立模型快照和元组访问语义，其模型项数组仍可用于本地诊断修改。
`ContactWBC.step(state, CoMReference(...), dt)` 提供带参考有效期校验的统一动力学入口。

`ContactSpec` 描述接触帧、局部放置、矩形尺寸和扭转摩擦；默认值复现 H1 示例假设，
支撑多边形与接触力约束共用这些几何配置。`CoMReferenceManager` 管理质心参考的
接受、覆盖、重规划和连续性，失败时清除旧参考。
独立的 `TorqueControlLoop` 与 `FixedContactSimulator` 组合示例见
[`torque_control.py`](../examples/python/torque_control.py)，使用仓库自带惯性 URDF。
完整类型、执行与故障语义见 [执行接口](execution.md)。

`ContactWBC.reset()` 清除求解工作区、warm start 和求解计数，保留标称姿态、
接触锚点、摩擦和接触裕量；需要重设锚点时重新构造控制器。
固定拓扑的矩阵块和变量缩放在实例内复用，修改 `contact_reserve` 会同步刷新约束模板，
外部修改一次返回的 QP 不会污染后续构造。
动力学接口仍独立于固定基座的 `Controller` / `VelocityCommand`。
当前接触 QP 使用 `alpha=1.4`，绝对/相对求解容差为 `1e-8`，物理约束验收为 `1e-5`。
参数交替对照见 [性能记录](performance.md)，最新源码与长测验收见 [验证记录](validation.md)。

## 验证结果与边界

早期 MPC、耦合接触约束、扰动交互及长测证据见 [2026-09-14 优化记录](h1_optimization_20260914.md)。
后续数值求解精度细化与对比方法见 [2026-09-16 求解优化](h1_solver_20260916.md)。

连续 3 轮（24 秒）重心转移，本机结果：质心横向范围约 5.55 cm，最小单脚法向力约 240.8 N，
最大关节力矩约 30.1 Nm，最大脚部位置漂移约 0.00064 mm。
动力学等式残差约 `5.7e-13`，正向动力学与 WBC 加速度差约 `1.6e-12`。
这些是同模型、固定接触、无噪声条件下的数值一致性结果，不是现实硬件精度。

同一默认目标运行 8 秒对比：加入前馈后质心位置 RMSE 从约 6.62 mm 降至 4.19 mm，
最大误差从 12.11 mm 降至 8.06 mm。CSV 与面板的目标/实际质心使用同一时间戳。
CSV 逐步写出，不再保留整段轨迹；JSON 新增最大质心误差、实际接触约束违反量和平均 WBC 耗时。
耗时包含 QP 构建与求解，不包括正向动力学、积分或绘制。

测试覆盖质量矩阵正定性、RNEA 一致性、接触 Jacobian 差分、无外力自由落体、
静态重力平衡、力矩扰动对加速度的影响、摩擦/压力中心限制，以及连续两轮重心转移、复位、非法实际接触力拒绝、前馈差分和浏览器循环状态。

**已加入简化质心/ZMP MPC；尚未实现全身非线性 MPC、抬脚/落脚接触切换、冲击、碰撞检测或 MuJoCo。**
正向动力学预先固定双脚接触，不自主判断脱离或滑动；WBC 中的接触力限制用于保证当前方案可行。
骨盆位姿由浮动基座动力学积分得到，不再采用原 march 示例的显示用地面对齐。
这份入口与原 `h1_mpc_wbc.py` 分开，原有 arms/bow/pickup/march 仍是运动学示例。

实现参考：[Pinocchio 动力学算法说明](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_a-features_g-dynamic.html)。

算法参考：[LIPM 行走控制器的预览模型](https://scaron.info/doc/lipm_walking_controller/walkgen.html)、[矩形接触力锥](https://scaron.info/publications/icra-2015.html)、[OSQP 矩阵更新](https://osqp.org/docs/examples/update-matrices.html)。
