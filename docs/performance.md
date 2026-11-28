# 控制与规划性能

性能测量应覆盖实际使用的完整调用路径，并与数值验收分开报告。共享 Linux 主机上的
分位数和最大观测值不能代替最坏执行时间；当前实现不提供 5 ms 硬实时保证。

## 测量范围

- `KinematicWBC.step()` 包含问题构造、求解、速度边界校验和命令构造；任务线性化通常在计时外。
- MPC 的初始化、首次求解、重复更新、冷重试和诊断矩阵展开分别计量。
- `control_ms` 计控制器调用，`cycle_ms` 计包含读取、任务、门控和下发的完整执行周期。
- H1 动力学示例分别报告 MPC、WBC 和正向动力学开销；墙钟与线程 CPU 时间用于区分计算和等待。
- `fixed_matrix_bytes` 只计规划器固定 H/A 的数值和索引，不包括求解器分解、动态向量及进程 RSS。

比较版本时使用相同依赖、输入、预热次数和 BLAS 线程数，交替运行顺序；记录 CPU affinity、
源码哈希及实际加载的扩展。同时报告 P50、P95、最大观测值、失败和重试次数。
数值结果需独立检查物理约束，不能只比较耗时或求解器状态。

## 结构与取舍

固定 QP 模板保存经过校验的不可写 H/A，动态梯度和边界每次验证。变化的 Hessian
重新校验凸性；普通公开数组被修改后会重新检查，不依赖可逆的只读标志。
稠密上三角矩阵打包可复用非零索引，变化稀疏结构仍重新构建。

求解前的 QP 缓冲区已由构造器转换为 NumPy 数组或 SciPy 稀疏矩阵。复数存储检查
直接读取 `dtype.kind`，减少进入通用类型分派的开销；有限性与维度继续独立校验。

求解器对 bytes 支撑的 H/A 快照共享不可写数据，各自保留矩阵与数组元数据；
普通公开数组仍复制数值，以检测后续写入。缩放后的求解器矩阵独立保存，物理残差继续
使用原始坐标的约束矩阵。该优化减少 Python 快照存储，不代表 OSQP 分解或进程 RSS 同比下降。

[SciPy 稀疏校验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.check_format.html)
可能重新分配索引存储。适配器在实际校验后重算矩阵存储凭据；转换后可写的缓冲区保存
独立快照，后续修改仍会触发重新校验。CSC/CSR 的格式本身也参与矩阵比较。

固定 CSC 通过浅复制矩阵封装并为 data/indices/indptr 建立独立数组视图，避免每个周期
重复运行 SciPy 的构造检查；校验凭据仍覆盖实际缓冲区和视图元数据。动态稠密 Hessian
只在非零结构保持不变时复用空 CSC 封装及不可写索引，每次重新提取数值，不保留旧数值。

C++ 速度边界在关节远离位置限位时，先用保守停止距离排除无需收紧的制动边界。
接近限位及速度差下溢时仍使用离散制动公式；停止距离以扩展精度计算，避免快捷判断溢出。

关节 MPC 的积分直接写入输出缓冲区，速度差分包含测量速度与首条命令之间的变化，
无需先拼接历史数组。C++ 计划验收使用列视图归约，避免逐预测步分配临时向量；
公开计划仍需检查积分一致性，计算溢出不能作为一致性证据。

同步参考管理器在一次复用更新中共用跟踪误差的采样结果；成功重规划后重新采样新轨迹。
C++ 返回速度边界、周期状态和求解结果时移动已拥有的缓冲区；门控与规划接收边界的
独立快照继续保留，避免诊断结果或调用方后续修改影响内部状态。
同步 C++ 参考管理器会验收规划器返回的完整轨迹，并接管该结果的缓冲区。复用时只插值
当前区间，开销随关节数增长。公开的 `JointPlan::sample()` 仍会验证完整轨迹。

Linux/GCC 12 的 x86_64 profile 仅为 OSQP 设置 `-falign-functions=64:16:16`，
并把该选项计入 Conan 包标识。该选项优先按 64 字节对齐，最多填充 15 字节，
否则回退到 16 字节对齐，语义见 [GCC 12 说明](https://gcc.gnu.org/onlinedocs/gcc-12.3.0/gcc/Optimize-Options.html)。
在本次构建中，QDLDL 代码字节保持不变，代码节对齐提高，消除了 C++ 重构引起的大 MPC
布局回退；其他平台和工具链仍需分别测量。

Python 与 native 消元 MPC 在默认求解中复用压缩固定模型。调用 `formulate()` 会按需展开
稠密诊断问题；使用自定义 Python 求解器也保留稠密接口。比较内存时应明确是否已创建诊断缓存。
Python 消元 MPC 用重叠预测区间的权重尾和构造时间 Hessian，省去初始化时的稠密矩阵乘法。
`sparse` / `native-sparse` 使用局部耦合的状态和控制变量；固定矩阵存储随关节数和预测长度
线性增长，但额外决策变量和收敛行为可能使小问题或切换负载更慢。
默认稀疏规划器会平移远离零点的窄关节区间，避免绝对位置抬高 OSQP 的相对收敛阈值。
返回的决策量、诊断问题及约束残差使用原关节坐标。平移结果须通过原 QP 的物理残差复核，
构造的轨迹仍独立验收位置、速度和速度差。
预测步长不超过 20 ms、范围至少 30 步且有紧速度差约束时，默认稀疏求解器内部为速度差增加
独立变量。公开问题及解列顺序保持为位置和速度。扩展问题的结果须通过原问题约束复核，
固定矩阵存储仍随关节数和预测长度线性增长。
扩展问题的热启动达到迭代上限或精度不足时，先对同一问题冷启动一次。必要时再用原稀疏问题重试。
相同条件下，默认消元规划器也可调用该稀疏模型。它提取速度决策并按消元 QP 复核原始约束，
诊断接口仍返回速度问题。首次使用后，固定矩阵统计计入稀疏模型。
Python 默认消元求解在未尝试稀疏模型且达到迭代上限时，也会用已保存的同一目标尝试一次稀疏求解。
结果按原消元约束复核，迭代次数计入两次求解。

长预测范围的稀疏 MPC 可先在较短的前段保留约束。位置跟踪目标在整段保持固定时，
后续无约束问题可用一维二次代价递推消去。递推得到的代价系数作为前段终端权重，
之后以对应反馈重建完整轨迹。规划器根据关节到目标的距离，从预设长度中选择能覆盖
估算行程的最短前段。重建逐步检查位置、速度和速度差，其中包括前后段之间的第一步
速度变化。完整原 QP 的残差与从实测初值积分得到的轨迹分别验收。若候选被拒绝，
继续用完整原问题求解。公开 `formulate()` 和求解结果都保留原预测范围和决策顺序。

Python 和原生稀疏规划器也可直接计算整段无约束解。只有整段通过原 QP 和物理轨迹验收，
包括初始速度到首条命令的变化，才返回零次迭代。固定约束矩阵按排序后的列直接写入
压缩存储，原生速度差模型也复用已有列的顺序。原生缩放矩阵和两种后端的速度差模型
在相应 OSQP 求解首次使用时构造。一次前段尝试最多使用 6,000 次迭代，未通过验收时
继续由完整原问题求解。

在独立的四轮交替原生对照中，96 × 200、50 ms 步长的近目标负载，求解线程 CPU
中位数从 `5.629 ms` 降到 `0.354 ms`。相同规模的直接稀疏矩阵构造把初始化中位数
从 `1.032 ms` 降到 `0.386 ms`，固定矩阵和该用例的完整求解决策逐值相同。
Python 的 96 × 200、20 ms 近目标用例中，延迟构造使固定矩阵占用从 `5,371,408 B`
降到 `2,378,504 B`。加上直接写入约束列，该用例不同阶段的初始化中位数分别为
`10.75 ms`、`5.87 ms` 和 `4.14 ms`。

原生初版 `d2aa2de` 与 `41d6b3a` 在同机回放 12 个闭环运动输入，规模为 96 关节 × 200 步，
预测步长为 50 ms。每版四轮交替，以下耗时取各轮分位数的中位数。

| 求解线程 CPU | 基线（s） | 折算后续代价（s） |
| --- | ---: | ---: |
| P50 | 2.124 | 0.626 |
| P95 | 2.905 | 0.904 |
| P99 | 3.259 | 1.014 |

48 次候选的完整 QP 和物理轨迹均通过验收。最大相对目标差为 `1.62e-9`，
最大位置差为 `2.61e-8 rad`，最大速度差为 `2.07e-7 rad/s`。
后来在同样的 50 ms 输入上交替比较初版与 `8f0efde`，求解线程 CPU 的 P50 / P99
从 `616 / 1,000 ms` 进一步降到 `143 / 199 ms`。48 次运动求解全部通过原始约束验收。
其中 5 个运动输入也以 20 ms 预测步长回放。基线与按行程选前段版本的线程 CPU 中位数
分别为 `6.613 s` 与 `0.829 s`，最高观测值分别为 `17.401 s` 与 `0.979 s`。
复现同类目标切换负载可给 `compare_native_mpc.py` 或 `compare_python_mpc.py` 传入
`--workload retarget --dt .05`，同时记录求解线程 CPU 和墙钟分位数。

在同样规模下，使用 `soak_async_mpc.py --backend native-sparse --joints 96 --horizon 200`
与 `--control-dt .05 --lookahead .60 --seconds 90` 各运行一次闭环仿真。两个版本各有
1,800 次控制尝试，完成后均回收子进程。基线和初版记录的结果如下。

| 闭环观测 | 基线 | 折算后续代价 |
| --- | ---: | ---: |
| 接受的控制周期 | 1,796 | 1,800 |
| 采用的参考更新 | 62 | 146 |
| 逾期回复 | 38 | 0 |
| 参考覆盖不足停机 | 4 | 0 |
| 子进程最长求解墙钟时间（ms） | 2,231 | 309 |

`check_async_soak.py` 复核了两个版本记录的事件与捕获轨迹。闭环在采用不同参考后会产生
不同的测量状态，因此求解墙钟时间应结合上面的同输入回放解读。

Python 和 C++ 的 `JointMPC` 默认预计算单关节对目标偏差的无约束速度响应，缓存占
`8*N` 字节。各关节使用各自的实测位置和目标缩放该响应。只有候选值满足原 QP 各行精确
上下界且通过重建轨迹验收时，才返回零迭代的求解结果。位置约束按预测步累加位移，
速度和速度差逐步比较，候选约束检查随关节数和步数线性增长。约束截断候选值时交由 OSQP 求解。
Python 可用 `unconstrained_fast_path=False` 关闭此路径。注入求解器或覆盖问题构造仍使用各自接口。
Python 在直接求解后返回 OSQP 时，可复用同一工作区的原始和对偶迭代量。复用要求当前测量时间
仍处于上次 OSQP 求解的第一个预测区间，并且新梯度更接近该工作区的梯度。

`cache_task_hessian=True` 适用于稳定的大 Jacobian 和权重；默认关闭。缓存逐值核对输入，
只保留最近一批，不缓存目标速度。动态 Jacobian 可能因额外比较与复制增加开销。
任务缓存与 QP 模板共享已验证的 Hessian 字节存储，分别持有数组元数据；Jacobian 和权重
仍保存独立快照。共享只减少这部分矩阵副本，不代表整个控制进程的内存减半。

`ProcessMPCWorker` 把规划放到独立进程，控制线程只提交、取回和使用有效参考。
它改变控制周期中的等待分布，不减少规划问题本身的计算量；启动、IPC、参考接续和停机
都需要单独测量。队列有界与非阻塞锁不构成 Python 或操作系统调度的硬实时保证。
新请求会通知空闲的 relay 线程立即发送。提交端在原有 mailbox 锁内发出通知，
空闲检查进程状态的周期不再决定请求发送时间。

`CoMPreviewMPC` 默认先尝试固定代价的无约束极小值。一次缓存 Cholesky 分解后，
每次只求解动态梯度对应的线性系统；候选值按原 QP 上下界逐值比较，不增加可行性松弛，
并对返回轨迹独立检查 Bernstein 区间约束与终端捕获点。候选不可行时继续使用 OSQP。
替换求解器、修改其求解方法或改变固定模型时，继续调用原求解接口。
可设置 `unconstrained_fast_path=False` 关闭此路径；活动约束频繁时需测量额外尝试的成本。
缓存分解占 `(2*N)^2*8` 字节，N=100 时约 313 KiB；这部分须与固定 H/A 分别报告。

## 小系数约束与精度

绝对约束残差不能直接作为决策变量误差的上界。一个活动约束的系数若只有 `8e-5`，
约 `3e-9` 的约束残差仍可能对应约 `3.5e-5` 的决策偏差。变量缩放和 OSQP 内部均衡
不会消除所有这类情况；单独收紧求解容差也可能先触发迭代上限。

对有此精度要求的问题，可选择 `OSQPSolver(normalize_constraints=True)`，按原始
A 每行最大绝对系数归一化 A 和两侧边界。全零行使用尺度 1；正行缩放保持可行集不变。
适配器保存原始 A，返回结果继续按原始 A/边界复核绝对物理残差，公开 QP 无需改写。
该选项与 `variable_scaling` 可同时使用，先归一化行，再变换变量坐标。
归一化后的矩阵和行除数只在 A 改变时更新；动态边界每次缩放，增加的开销需要按实际负载测量。

`scripts/soak_scaled_qp.py --solver-normalize-rows` 覆盖适配器内归一化，
`--normalize-rows` 保留调用方手动归一化对照，两者不能同时指定。
确定性 KKT 用例见 `tests/data/scaled_qp_small_row.json`。选项默认关闭，
适配器仍要求 `1e-6` 的绝对物理残差，不额外承诺问题无关的决策距离界限；
默认缩放长测的严格决策偏差断言保持不变。行归一化可能改变迭代次数及不可行性判据的
数值表现；大系数行也可能因原始物理残差超限被拒绝。按行相除发生有限值溢出或把非零值
下溢为零时，会在更新工作区前抛出 `ValueError`。近相关的活动约束仍可能在配置的迭代
预算内无法达到严格收敛状态，调用方需按已有失败处理路径拒绝这些结果。

## 复现入口

输出默认写入 `/tmp`；长期记录、截图和阶段报告保存在仓库外或忽略的 `dist/` 中。
固定 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，native 测量另设置 `PYTHONPATH=build/native`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=build/native \
  .venv/bin/python scripts/benchmark_mpc.py --backend all --cases 6x15 30x60 \
  --rounds 4 --iterations 100 --output /tmp/wbx-mpc.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compare_python_wbc.py --baseline-python /path/to/baseline/python \
  --rounds 4 --iterations 6000 --output /tmp/wbx-wbc-comparison.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compare_python_mpc.py --baseline-python /path/to/baseline/python \
  --output /tmp/wbx-mpc-comparison.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compare_scaled_qp.py --baseline-python /path/to/baseline/python \
  --output /tmp/wbx-scaled-comparison.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=build/native \
  .venv/bin/python scripts/soak_h1_dynamics.py --controller mpc --seconds 120 \
  --cpu-timing --output /tmp/wbx-h1-timing.json
```

`benchmark_process_mpc.py` 比较同步规划周期与进程父周期；`benchmark_wbc_hessian_cache.py`
对照固定和变化任务的可选缓存。具体参数以各入口 `--help` 为准。
`process_round_trip` 计从提交到父进程取回结果的时间。该值包含 `--interval` 指定的轮询等待，
`child_solve` 单独报告子进程的求解时间。
`benchmark_qp_row_normalization.py` 串行比较调用方每次归一化、调用方缓存固定 A、
以及适配器内归一化的完整 QP 构造/求解路径；报告墙钟与线程 CPU 分位数及独立数值验收。
调用方缓存模式只在固定 A 的负载中预处理矩阵，变化 A 仍在计时内重新归一化。
`compare_com_preview.py --baseline-python /path/to/baseline/python` 比较质心预览的
无活动约束、活动约束与交替负载，单独报告初始化、首次规划、墙钟/线程 CPU 分位数、
直接求解命中和拒绝次数；数值复核在计时外进行。`--disable-fast-path` 可对照关闭快捷路径，
`factor_bytes` 计分解缓存，`fixed_qp_bytes` 只计固定 H/A。仅比较成功率一致的负载耗时。
`benchmark_native_validation.py --acceleration 4` 测量启用速度差与制动边界的完整 native WBC；
加 `--near-limits` 覆盖靠近位置限位的负载。默认不设置加速度限制，适合检查普通路径是否回退。
通过 `PYTHONPATH` 选择基线或候选扩展，使用 `--trace` 保存输出并独立比较数值。
版本对照中，`compare_python_mpc.py --backend sparse` 与
`compare_native_mpc.py --backend native-sparse` 分别选择稀疏实现；默认仍为消元 MPC。
两者在计时外独立复核导出计划的位置、速度和速度差约束。
两个对照入口都可用 `--workload near` 测量接近目标的周期，或用 `--workload alternating`
交替测量接近目标和目标跳变。报告中的 `zero_iteration_solves` 记录直接求解次数。
耐久与故障恢复的验收方法见 [验证说明](validation.md)，运行边界见 [执行接口](execution.md)。
