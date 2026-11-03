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

关节 MPC 的积分直接写入输出缓冲区，速度差分包含测量速度与首条命令之间的变化，
无需先拼接历史数组。C++ 计划验收使用列视图归约，避免逐预测步分配临时向量；
公开计划仍需检查积分一致性，计算溢出不能作为一致性证据。

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
版本对照中，`compare_python_mpc.py --backend sparse` 与
`compare_native_mpc.py --backend native-sparse` 分别选择稀疏实现；默认仍为消元 MPC。
两者在计时外独立复核导出计划的位置、速度和速度差约束。
耐久与故障恢复的验收方法见 [验证说明](validation.md)，运行边界见 [执行接口](execution.md)。
