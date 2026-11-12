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

求解器对 bytes 支撑的 H/A 快照共享不可写数据，各自保留矩阵与数组元数据；
普通公开数组仍复制数值，以检测后续写入。缩放后的求解器矩阵独立保存，物理残差继续
使用原始坐标的约束矩阵。该优化减少 Python 快照存储，不代表 OSQP 分解或进程 RSS 同比下降。

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

Linux/GCC 12 的 x86_64 profile 仅为 OSQP 设置 `-falign-functions=64:16:16`，
并把该选项计入 Conan 包标识。该选项优先按 64 字节对齐，最多填充 15 字节，
否则回退到 16 字节对齐，语义见 [GCC 12 说明](https://gcc.gnu.org/onlinedocs/gcc-12.3.0/gcc/Optimize-Options.html)。
在本次构建中，QDLDL 代码字节保持不变，代码节对齐提高，消除了 C++ 重构引起的大 MPC
布局回退；其他平台和工具链仍需分别测量。

Python 与 native 消元 MPC 在默认求解中复用压缩固定模型。调用 `formulate()` 会按需展开
稠密诊断问题；使用自定义 Python 求解器也保留稠密接口。比较内存时应明确是否已创建诊断缓存。
`sparse` / `native-sparse` 使用局部耦合的状态和控制变量；固定矩阵存储随关节数和预测长度
线性增长，但额外决策变量和收敛行为可能使小问题或切换负载更慢。

`cache_task_hessian=True` 适用于稳定的大 Jacobian 和权重；默认关闭。缓存逐值核对输入，
只保留最近一批，不缓存目标速度。动态 Jacobian 可能因额外比较与复制增加开销。
任务缓存与 QP 模板共享已验证的 Hessian 字节存储，分别持有数组元数据；Jacobian 和权重
仍保存独立快照。共享只减少这部分矩阵副本，不代表整个控制进程的内存减半。

`ProcessMPCWorker` 把规划放到独立进程，控制线程只提交、取回和使用有效参考。
它改变控制周期中的等待分布，不减少规划问题本身的计算量；启动、IPC、参考接续和停机
都需要单独测量。队列有界与非阻塞锁不构成 Python 或操作系统调度的硬实时保证。

## 小系数约束与精度

绝对约束残差不能直接作为决策变量误差的上界。一个活动约束的系数若只有 `8e-5`，
约 `3e-9` 的约束残差仍可能对应约 `3.5e-5` 的决策偏差。变量缩放和 OSQP 内部均衡
不会消除所有这类情况；单独收紧求解容差也可能先触发迭代上限。

对有此精度要求的问题，可在交给求解器前按每行最大绝对系数归一化 A 和两侧边界，
并继续使用原始 A/边界独立复核物理残差；全零行使用尺度 1。正行缩放保持可行集不变。
`scripts/soak_scaled_qp.py --normalize-rows` 提供该对照，确定性 KKT 用例见
`tests/data/scaled_qp_small_row.json`。适配器默认仍接受 `1e-6` 的绝对物理残差，
不额外承诺问题无关的决策距离界限；默认缩放长测的严格决策偏差断言保持不变。

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
`benchmark_native_validation.py --acceleration 4` 测量启用速度差与制动边界的完整 native WBC；
加 `--near-limits` 覆盖靠近位置限位的负载。默认不设置加速度限制，适合检查普通路径是否回退。
通过 `PYTHONPATH` 选择基线或候选扩展，使用 `--trace` 保存输出并独立比较数值。
版本对照中，`compare_python_mpc.py --backend sparse` 与
`compare_native_mpc.py --backend native-sparse` 分别选择稀疏实现；默认仍为消元 MPC。
两者在计时外独立复核导出计划的位置、速度和速度差约束。
耐久与故障恢复的验收方法见 [验证说明](validation.md)，运行边界见 [执行接口](execution.md)。
