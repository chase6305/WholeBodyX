# WholeBodyX

C++ 与 Python 均可实现控制器和规划器的机器人控制框架。
以独立算法库、统一接口和机器人适配为边界，支持逐步接入自有 WBC/MPC 仓库。

- [功能、接口和算法设计](docs/design.md)
- [开源仓库参考与取舍](docs/open_source_references.md)
- [可视化工具与操作](docs/visualization.md)
- [验证结果与界面截图](docs/validation.md)
- [示例目录、算法配方与运行命令](examples/README.md)
- [MPC 性能、缓存和长期运行边界](docs/performance.md)
- [Unitree H1 运动学 MPC + WBC 案例](docs/h1_example.md)
- [Unitree H1 浮动基座动力学、质心 MPC 与力矩 WBC](docs/h1_dynamics.md)
- [参考调度、失败处理与状态偏移恢复](docs/runtime.md)
- [自有仓库接入、后端注册与统一命令](docs/integration.md)
- [模型/驱动注入、稀疏 MPC 与力矩执行](docs/execution.md)
- [C++ / Python 关节顺序适配](docs/joint_order.md)

目前提供 C++ / Python 速度级 WBC、关节空间 MPC、OSQP adapter、
Python 末端/姿态任务、可选 URDF adapter、平面双臂示例，以及独立 C++ 示例。
核心算法可分别运行，Python 不强制安装本项目的 C++ 扩展。

核心双臂案例仍是固定基座运动学基线。另有实验性 Python H1 浮动基座动力学 WBC、
恒定高度质心/ZMP MPC 和固定接触力矩仿真；已提供速度/力矩驱动边界，
实际设备驱动、碰撞与接触切换尚未实现。
双臂测试模型为两条平面 3R 链，仿真后端为关节速度积分器。

## 浏览器可视化

```bash
uv sync --locked --extra viz
uv run --extra viz python -m wholebodyx.visualization
```

打开 `http://127.0.0.1:8080`，点击“运行”。Viser 显示实际双臂、末端目标、
MPC 预测和历史轨迹；内嵌 uPlot 显示误差和求解耗时。
支持暂停、单步、重置、调节关节目标和切换 Python/C++ 后端。
“示例场景”提供静态跟踪、末端圆轨迹、双臂协作和 MPC 在线换目标；
WBC 与 MPC 后端可以分别选择。
另有状态偏移恢复场景，用于观察预测偏差触发重规划。
详细操作及远程访问方式见 [可视化说明](docs/visualization.md)。

## Python

需要 Python >= 3.10 和 uv。在仓库根目录运行：

```bash
uv sync --locked
uv run python examples/python/dual_arm_tracking.py
uv run python examples/python/dual_arm_tracking.py --mpc
uv run pytest -q
```

统一入口包含在 wheel 中，安装后可在仓库外运行：

```bash
uv run wholebodyx list
uv run wholebodyx run disturbance_recovery --csv /tmp/recovery.csv
uv run wholebodyx doctor
uv run --extra viz wholebodyx view tracking
```

`doctor --require native urdf viz` 可检查指定的可选能力。WBC/MPC 的创建统一经过
`BackendRegistry`，支持分别注册自己的工厂并交给运行器或浏览器。

MPC 可显式选择 `python`、`native`、`sparse` 或 `native-sparse`；后两者采用
未消元稀疏问题，适合单独评估较长预测范围。`native-sparse` 需要重新构建匹配的扩展。
问题规模和目标变化方式都会影响收益，见 [性能记录](docs/performance.md)。
可运行的自定义配置示例：`uv run python examples/python/custom_backends.py`。

另有任务权重、限位与失败恢复、URDF 跟踪等示例。动态场景支持 CSV 导出及浏览器展示：

```bash
uv run python examples/python/dual_arm_coordination.py --csv /tmp/coordination.csv
uv run --extra viz python examples/python/cartesian_circle.py --viewer
```

示例覆盖双臂、URDF、H1 和独立力矩闭环，另有 5 个独立 C++ 入口，详见 [示例指南](examples/README.md)。

基础环境没有 native 扩展和 Pinocchio 时，对应的可选测试会跳过。
需要 URDF 时安装 `urdf` extra：

```bash
uv sync --locked --extra urdf
uv run --extra urdf pytest -q tests/python/test_model_and_loop.py
```

`wholebodyx.model.PinocchioModel` 接收 URDF 路径；当前只支持固定基座、每关节
`nq=nv=1` 且具有有限限位的模型。

## C++ 与 Python 绑定

当前已配置 Linux x86_64 / GCC 12，需预先安装 GCC/G++ 12、CMake >= 3.20 和 Conan 2。
使用其他编译器或平台时建立对应 profile，并单独验证。首次构建需要下载 Conan 依赖。

```bash
# 仅 C++，不需要安装 Python 运行包或构建绑定
bash scripts/build.sh
./build/native/wbx_joint_control

# 可选 Python 扩展，先执行 uv sync
bash scripts/build.sh --python
PYTHONPATH=build/native WBX_REQUIRE_NATIVE=1 uv run pytest -q
PYTHONPATH=build/native uv run python examples/python/dual_arm_tracking.py --backend native
PYTHONPATH=build/native uv run python examples/python/dual_arm_tracking.py --backend native --mpc
```

Python 可通过 `wholebodyx.native.NativeWBC` / `NativeMPC` 单独替换任一后端。
两种后端使用相同的状态、任务、参考和求解结果类型。
当模型和算法库的关节排列不同时，使用 `JointOrder` 与
`ReorderedController` / `ReorderedPlanner` 转换，支持 WBC/MPC 各自的顺序。

C++ 使用 `#include <wholebodyx/control.hpp>`，通过
`find_package(WholeBodyX CONFIG REQUIRED)` 和 `wholebodyx::wholebodyx` 消费安装后的库。
可执行以下命令验证 Conan 包及下游消费：

```bash
conan create . -pr:h profiles/linux-gcc12-release -pr:b profiles/linux-gcc12-release --lockfile=conan.lock
```

## Python 分发包

```bash
# 独立 Python 包
uv build --python .venv/bin/python

# 可选 native wheel + 源码包；先运行一次 Conan/CMake 构建
uv run python scripts/build_native.py \
  --toolchain build/conan/build/Release/generators/conan_toolchain.cmake
```

`native/pyproject.toml` 是 staging 模板，应通过上述脚本构建，不能在 `native/` 中直接运行 build。
生成的 native 源码包包含 C++ 源码；重新编译需要先准备 Eigen/OSQP，并向构建器传入相应
CMake toolchain 或依赖查找配置。当前 wheel 为本机构建产物，尚未发布或进行多平台兼容性处理。

## 代码导航

| 内容 | 路径 |
| --- | --- |
| C++ 公共接口与算法 | `include/wholebodyx/control.hpp`、`cpp/` |
| Python 状态与 QP | `python/wholebodyx/types.py`、`qp.py` |
| Python WBC / MPC | `python/wholebodyx/wbc.py`、`mpc.py` |
| 模型、任务、积分仿真 | `python/wholebodyx/model.py`、`tasks.py`、`simulation.py` |
| 跨语言适配 | `python/wholebodyx/native.py`、`bindings/` |
| 关节顺序映射与控制器/规划器包装 | `python/wholebodyx/joint_order.py`、`include/wholebodyx/joint_order.hpp` |
| 后端工厂与加载诊断 | `python/wholebodyx/backends.py`、`diagnostics.py` |
| 场景执行、CSV 与统一命令 | `python/wholebodyx/session.py`、`runner.py`、`cli.py` |
| MPC 参考校验、有效期与重规划 | `python/wholebodyx/runtime.py`、`include/wholebodyx/reference.hpp` |
| 可注入的速度组合运行器 | `python/wholebodyx/execution.py` |
| 稀疏 QP / 关节 MPC | `python/wholebodyx/sparse_qp.py`、`sparse_mpc.py`、`include/wholebodyx/sparse_mpc.hpp` |
| 动力学契约、力矩门控与独立仿真 | `python/wholebodyx/dynamics_types.py`、`torque.py`、`dynamics_simulation.py` |
| 质心 MPC 与参考调度 | `python/wholebodyx/centroidal.py`、`centroidal_runtime.py` |
| 浏览器场景与诊断 | `python/wholebodyx/visualization/` |
| 构建配置 | `conanfile.py`、`profiles/`、`pyproject.toml`、`native/` |
| 数值与闭环测试 | `tests/`、`test_package/` |
