# 验证方法

验证覆盖数值问题、控制契约、故障恢复、Python/native 一致性、分发包和下游 C++ 消费。
测试计数随用例变化；CI 日志报告具体提交的通过、失败和跳过数量。
阶段报告、原始测量与截图保存在仓库外，长期文档只维护方法、验收条件和能力边界。

## 自动检查

基础 CI 在 Python 3.10 和 3.12 上运行测试与纯 Python 包构建。
另在 Python 3.10 上使用声明的最低版本 NumPy 1.24.0、SciPy 1.10.0 和 OSQP 1.0.0 执行回归，
覆盖旧版异常处理和稀疏矩阵元数据隔离；该组合在安装锁定环境后单独覆盖依赖，测试时不重新同步。
native/URDF CI 使用 Python 3.11，构建 C++ 和扩展，运行完整可选依赖测试及 Conan 消费方。
H1 无界面测试使用仓库内固定的关节、限位与惯性模型，不需要下载网格；来源和许可证见
[测试模型](../tests/data/h1/README.md)。示例可视化仍使用用户提供的完整模型资源。

```bash
uv sync --locked --all-extras
bash scripts/build.sh --python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=build/native WBX_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q
ctest --test-dir build/native --output-on-failure
conan create . -pr:h profiles/linux-gcc12-release -pr:b profiles/linux-gcc12-release --lockfile=conan.lock
```

本地运行 H1 用例时，示例默认在相邻 `HumanoidAssets/UnitreeH1/robot.urdf` 查找模型。
缺少外部资源时相应测试会跳过；CI 把 `tests/data/h1/robot.urdf` 复制到这个位置，确保这些用例执行。
`WBX_REQUIRE_NATIVE=1` 把缺少扩展变成失败，避免误把 Python-only 结果当作 native 验收。
关闭 pytest 插件自动加载可隔离系统 ROS 等无关插件。

## 数值与数据契约

- 独立离散制动 rollout 检查位置、速度、速度差和下一步可行性；注入求解残差后仍验证边界。
  C++ 回归另覆盖停止距离快捷判断两侧，以及扩展精度和速度差下溢边界。
- 耦合 WBC 与 QP 使用解析解、KKT 最优解及独立物理残差；缩放前后保持物理单位和验收阈值。
- 可选约束行归一化覆盖小系数活动约束、全零/空行、无穷边界、固定矩阵与结构变化，
  并注入归一化残差很小但物理残差超限的结果，确认最终验收仍使用原始 A/边界。
- 修改公开数组、稀疏索引、dtype、shape 和 strides 后必须重新校验；非法输入不得污染健康工作区。
  CSC/CSR 格式不同也不能只凭相同的压缩缓冲区复用矩阵，因为相同缓冲区可表示不同约束。
  索引类型或字节序转换形成可写缓冲区后，修改数值、索引或指针仍须触发重新校验；
  非法 Hessian 不得替换健康的工作区。
- 稠密和稀疏 QP 构造都拒绝复数矩阵和边界存储，包括虚部为零的复数 dtype；
  输入在浮点转换前检查，避免转换警告后虚部丢失；实数无穷边界继续允许。
- Python/native 对照包含固定和变化 Jacobian、不同关节顺序、稀疏与消元 MPC、失败后恢复。
- 计划检查测量起点、时钟、积分一致性、完整控制区间与限位；旧参考、越代回复和不合格结果不执行。
- 动力学检查质量矩阵、逆/正向动力学、接触约束、力矩限位、四元数、积分溢出和输入快照隔离。

## 故障恢复与耐久

故障注入覆盖读取、任务、求解、下发、复位、取消和停机异常。检查故障锁存、一次停机尝试、
阻止后续 I/O，以及成功复位后从新测量恢复。下发后报错保留已经提交的物理或仿真状态。
执行预算在各阶段边界检测；它不能中断阻塞调用，也不能撤销已经发生的下发。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=build/native \
  .venv/bin/python scripts/soak_execution.py --native --torque --dynamic-tasks \
  --seconds 3600 --output /tmp/wbx-execution.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=build/native \
  .venv/bin/python scripts/soak_wbc.py --seconds 3600 --output /tmp/wbx-wbc.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=build/native \
  .venv/bin/python scripts/soak_async_mpc.py --backend all --joints 30 \
  --horizon 60 --control-dt .1 --lookahead .5 --seconds 3600 \
  --output /tmp/wbx-async.json
```

`soak_mpc.py` 和 `soak_scaled_qp.py` 补充规划与缩放更新检查；`check_async_soak.py`
流式复核异步事件、回复采用和故障恢复。报告须说明实际墙钟时长、模拟时间、拒绝/重试、
源码是否保持不变、父子进程 RSS 采样与子进程回收。并行测试的时长不能相加。
`soak_scaled_qp.py --solver-normalize-rows` 将原始物理 QP 交给适配器归一化，
独立 KKT 最优解、目标差和物理残差阈值保持不变；`--normalize-rows` 则在调用方预先归一化。
两种模式分别报告，不能把调用方模式的结果算作适配器内部归一化的验收。
可加 `--row-exponents -10 1`，把物理约束行放到跨 11 个数量级的单位范围内，
同时反向缩放 KKT 乘子，保持目标函数中活动约束的作用量级；验收仍使用原始物理单位。
共享主机上分批耐久检查需要保留 RNG 与工作区状态，并分别记录实际计算段、等待锁和总墙钟时间。

## 分发与能力边界

构建 wheel 和源码包后，在仓库外的隔离环境安装并运行示例、测试和真实规划子进程，
核对实际模块来源；C++ 变更重新构建下游消费方。Sanitizer 结果应注明本库与依赖的插桩范围。
需要浏览器的变更另用实际页面交互检查加载、执行、暂停、复位、曲线与浏览器异常。

这些检查不证明实机精度、接触切换、碰撞约束、其他操作系统兼容性或硬实时调度。
H1 当前使用持久双脚刚性接触的实验模型；其物理验收与执行耗时应分别报告。
具体接口见 [接入指南](integration.md)、[执行组件](execution.md) 和 [H1 动力学](h1_dynamics.md)。
